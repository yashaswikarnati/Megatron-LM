# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon multimodal data provider for MIMO.

This module intentionally mirrors the provider used by the previous
``feat/nemotron-moe-vlm-mimo`` branch. Energon's ``MultiModalPackingEncoder``
owns sample cooking, preencoding, and packing; the MIMO-specific adapter only
expands each single ``<image>`` placeholder into one placeholder per image
embedding and remaps the batch to MIMO's forward signature.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from megatron.energon.task_encoder.multimodal import (
    MultiModalPackingEncoder,
    PackingConfig,
    VisionConfig,
)
from megatron.energon.task_encoder.multimodal.sample_types import PackedSample
from megatron.energon.task_encoder.multimodal.vision_tokens import get_num_image_embeddings


class TokenizerAdapter:
    """Wrap Megatron tokenizers for Energon's tokenizer protocol."""

    def __init__(self, megatron_tokenizer) -> None:
        self._tok = megatron_tokenizer
        inner = megatron_tokenizer
        if hasattr(inner, "_tokenizer"):
            inner = inner._tokenizer
        if hasattr(inner, "tokenizer"):
            inner = inner.tokenizer
        self._hf = inner

    @property
    def pad_token_id(self) -> int:
        """Return the tokenizer pad id."""
        return self._tok.pad

    @property
    def eos_token_id(self) -> int:
        """Return the tokenizer EOS id."""
        return self._tok.eod

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text with the wrapped HuggingFace tokenizer."""
        return self._hf.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        """Decode token ids with the wrapped HuggingFace tokenizer."""
        return self._hf.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to ids with the wrapped Megatron tokenizer."""
        return self._tok.convert_tokens_to_ids(tokens)


class MimoMultiModalPackingEncoder(MultiModalPackingEncoder):
    """Remap Energon multimodal packed samples to MIMO batch inputs."""

    def __init__(
        self,
        vision_config: VisionConfig,
        packing_config: PackingConfig,
        tokenizer,
        encoder_name: str = "radio_encoder",
        encoder_input_key: str = "x",
        target_seq_length: Optional[int] = None,
    ) -> None:
        super().__init__(vision_config, packing_config, tokenizer)
        self.encoder_name = encoder_name
        self.encoder_input_key = encoder_input_key
        self._target_seq_length = target_seq_length
        self._embeddings_per_tile = get_num_image_embeddings(
            img_h=vision_config.img_h,
            img_w=vision_config.img_w,
            patch_dim=vision_config.patch_dim,
            class_token_len=vision_config.class_token_len,
            disable_vision_class_token=vision_config.disable_vision_class_token,
            pixel_shuffle=vision_config.pixel_shuffle,
            conv_merging=vision_config.conv_merging,
            use_tile_tags=vision_config.use_tile_tags,
            max_num_tiles=vision_config.max_num_tiles,
            use_image_break_token=vision_config.use_image_break_token,
        )

    def batch(self, samples: list[PackedSample]) -> dict:
        """Expand image placeholders and return a MIMO-compatible batch."""
        image_token_id = self.packing_config.image_token_id
        ignore_index = self.packing_config.ignore_index
        pad_id = self.packing_config.pad_id
        emb_per_tile = self._embeddings_per_tile

        expanded_tokens_list = []
        expanded_labels_list = []
        all_images = []

        for sample in samples:
            tokens = sample.tokens
            labels = sample.labels
            num_tiles = sample.num_tiles
            budget = self._target_seq_length
            new_tokens = []
            new_labels = []
            img_idx = 0
            truncated = False
            truncated_padding_only = False
            kept_tile_count = 0

            for idx, token in enumerate(tokens.tolist()):
                if token == image_token_id:
                    n_tiles = num_tiles[img_idx] if img_idx < len(num_tiles) else 1
                    n_tokens = n_tiles * emb_per_tile
                    if budget is not None and len(new_tokens) + n_tokens > budget:
                        truncated = True
                        break
                    new_tokens.extend([image_token_id] * n_tokens)
                    new_labels.extend([ignore_index] * n_tokens)
                    kept_tile_count += n_tiles
                    img_idx += 1
                else:
                    if budget is not None and len(new_tokens) + 1 > budget:
                        truncated = True
                        truncated_padding_only = _remaining_tokens_are_padding(
                            tokens=tokens,
                            labels=labels,
                            start=idx,
                            pad_id=pad_id,
                            ignore_index=ignore_index,
                        )
                        break
                    new_tokens.append(token)
                    new_labels.append(labels[idx].item())

            if truncated and len(sample.cu_lengths) > 2 and not truncated_padding_only:
                raise RuntimeError(
                    "Packed Energon sample exceeds target sequence length after MIMO image-token "
                    "expansion. Refusing to clamp packed cu_seqlens because that can create "
                    "zero-length packed segments. Increase --total-seq-length or lower image "
                    "tiling/packing settings."
                )

            if truncated and not truncated_padding_only:
                warnings.warn(
                    f"Sample truncated to fit target_seq_length ({self._target_seq_length}): "
                    f"kept {len(new_tokens)} of ~{len(tokens)} original tokens, "
                    f"{img_idx}/{len(num_tiles)} images ({kept_tile_count} tiles). "
                    "Consider increasing --total-seq-length or reducing --max-num-tiles.",
                    stacklevel=2,
                )

            all_images.extend(sample.images[:kept_tile_count])
            expanded_tokens_list.append(torch.tensor(new_tokens, dtype=torch.long))
            expanded_labels_list.append(torch.tensor(new_labels, dtype=torch.long))

        max_len = max(len(tokens) for tokens in expanded_tokens_list)
        if self._target_seq_length is not None:
            max_len = self._target_seq_length

        batch_size = len(samples)
        tokens_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        labels_batch = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)

        for idx, (tokens, labels) in enumerate(zip(expanded_tokens_list, expanded_labels_list)):
            tokens_batch[idx, : len(tokens)] = tokens
            labels_batch[idx, : len(labels)] = labels

        loss_mask = (labels_batch != ignore_index).float()
        loss_mask[labels_batch == image_token_id] = 0.0
        position_ids = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).contiguous()

        result = {
            "input_ids": tokens_batch,
            "labels": labels_batch,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if all_images:
            images = self.tiling_strategy.stack(all_images)[0]
            result["modality_inputs"] = {
                "images": {self.encoder_name: {self.encoder_input_key: images}}
            }

        is_packed = any(len(sample.cu_lengths) > 2 for sample in samples)
        if is_packed:
            if batch_size != 1:
                raise RuntimeError(f"Packing requires micro_batch_size=1, got {batch_size}")
            result["packing_kwargs"] = _build_packing_kwargs(samples[0], max_len)

        return result


def _remaining_tokens_are_padding(
    tokens: torch.Tensor, labels: torch.Tensor, start: int, pad_id: int, ignore_index: int
) -> bool:
    """Return whether truncation only drops right-padding tokens."""
    remaining_tokens = tokens[start:]
    remaining_labels = labels[start:]
    return bool(
        remaining_tokens.numel() > 0
        and torch.all(remaining_tokens == pad_id).item()
        and torch.all(remaining_labels == ignore_index).item()
    )


def _build_packing_kwargs(sample: PackedSample, max_len: int) -> dict[str, torch.Tensor]:
    """Build validated packed-sequence metadata for the MIMO language model."""
    cu_seqlens = sample.cu_lengths.to(dtype=torch.int32)
    if cu_seqlens.numel() < 2:
        raise RuntimeError(f"Packed sample must have at least two cu_lengths, got {cu_seqlens}")
    if torch.any(cu_seqlens[1:] < cu_seqlens[:-1]):
        raise RuntimeError(f"Packed cu_lengths must be monotonic, got {cu_seqlens.tolist()}")

    if cu_seqlens[0] != 0:
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), cu_seqlens])
    if cu_seqlens[-1] > max_len:
        raise RuntimeError(
            f"Packed cu_lengths end at {int(cu_seqlens[-1])}, beyond sequence length {max_len}"
        )
    if cu_seqlens[-1] != max_len:
        cu_seqlens = torch.cat([cu_seqlens, torch.tensor([max_len], dtype=torch.int32)])

    segment_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    if torch.any(segment_lens <= 0):
        raise RuntimeError(
            "Packed cu_lengths must be strictly increasing after MIMO expansion, "
            f"got {cu_seqlens.tolist()}"
        )
    max_seqlen = segment_lens.max()
    return {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_kv": cu_seqlens,
        "cu_seqlens_q_padded": cu_seqlens,
        "cu_seqlens_kv_padded": cu_seqlens,
        "max_seqlen_q": max_seqlen,
        "max_seqlen_kv": max_seqlen,
        "total_tokens": torch.tensor(max_len, dtype=torch.int32),
    }


def build_multimodal_encoder(
    args, tokenizer, encoder_name: str = "radio_encoder", encoder_input_key: str = "x"
) -> MimoMultiModalPackingEncoder:
    """Build the MIMO Energon encoder from train args."""
    target_seq_length = _resolve_target_seq_length(args)
    image_token_id = getattr(args, "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids(getattr(args, "image_token", "<image>"))
    pad_id = getattr(args, "pad_token_id", tokenizer.pad)

    vision_config = VisionConfig(
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        vision_model_type=getattr(args, "vision_model_type", "radio"),
        disable_vision_class_token=getattr(args, "disable_vision_class_token", False),
        pixel_shuffle=getattr(args, "pixel_shuffle", False),
        max_num_tiles=getattr(args, "max_num_tiles", getattr(args, "num_image_tiles", 1)),
        use_tiling=getattr(args, "use_tiling", False),
        use_thumbnail=getattr(args, "use_thumbnail", False),
        class_token_len=getattr(args, "class_token_len", None) or 1,
        conv_merging=getattr(args, "conv_merging", False),
        use_tile_tags=getattr(args, "use_tile_tags", False),
        use_image_break_token=getattr(args, "image_break_token", None) is not None,
        use_area_weighted_aspect_ratio=getattr(args, "use_area_weighted_aspect_ratio", False),
        dynamic_resolution=getattr(args, "dynamic_resolution", False),
    )
    packing_config = PackingConfig(
        seq_length=target_seq_length, pad_id=pad_id, image_token_id=image_token_id
    )
    return MimoMultiModalPackingEncoder(
        vision_config=vision_config,
        packing_config=packing_config,
        tokenizer=TokenizerAdapter(tokenizer),
        encoder_name=encoder_name,
        encoder_input_key=encoder_input_key,
        target_seq_length=target_seq_length,
    )


def _resolve_target_seq_length(args) -> int:
    """Return the sequence length used by Energon and MIMO expansion."""
    target_seq_length = getattr(args, "total_seq_length", None)
    if target_seq_length is None:
        target_seq_length = getattr(args, "seq_length", None)
    if target_seq_length is None:
        raise AttributeError("Energon multimodal provider requires total_seq_length or seq_length")
    return target_seq_length
