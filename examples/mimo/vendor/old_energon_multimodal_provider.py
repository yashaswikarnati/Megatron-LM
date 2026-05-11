# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Previous-branch MIMO Energon encoder fixture for dataloader parity checks.

Bridges energon's ``MultiModalPackingEncoder`` (which produces 1 image-placeholder
per ``<image>`` tag) to MIMO's contract (which requires N placeholders per image,
where N = num_tiles * embeddings_per_tile, for 1:1 ``masked_scatter_`` alignment).
"""

import warnings
from typing import List, Optional

import torch

from megatron.energon.task_encoder.multimodal import (
    MultiModalPackingEncoder,
    PackingConfig,
    VisionConfig,
)
from megatron.energon.task_encoder.multimodal.sample_types import PackedSample
from megatron.energon.task_encoder.multimodal.vision_tokens import get_num_image_embeddings


# ---------------------------------------------------------------------------
# Tokenizer adapter: Megatron tokenizer → energon TokenizerProtocol
# ---------------------------------------------------------------------------
class _TokenizerAdapter:
    """Wraps a Megatron tokenizer to satisfy energon's ``TokenizerProtocol``.

    Handles both HuggingFaceTokenizer (single wrapper) and MultimodalTokenizer
    (double wrapper) by walking the ``_tokenizer`` / ``tokenizer`` chain.
    """

    def __init__(self, megatron_tokenizer):
        self._tok = megatron_tokenizer
        # Walk the wrapper chain to reach the HF PreTrainedTokenizerFast.
        # Chain: DefaultTokenizerVision._tokenizer → MegatronMultimodalTokenizer
        #        MegatronMultimodalTokenizer.tokenizer → HF AutoTokenizer
        # IMPORTANT: Do NOT drill into PreTrainedTokenizerFast.tokenizer — that's
        # the raw Rust tokenizer whose encode() returns tokenizers.Encoding, not list[int].
        inner = megatron_tokenizer
        # Unwrap DefaultTokenizerVision → MegatronMultimodalTokenizer
        if hasattr(inner, '_tokenizer'):
            inner = inner._tokenizer
        # Unwrap MegatronMultimodalTokenizer → HF AutoTokenizer
        if hasattr(inner, 'tokenizer'):
            inner = inner.tokenizer
        self._hf = inner

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad

    @property
    def eos_token_id(self) -> int:
        return self._tok.eod

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        return self._hf.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return self._hf.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        return self._tok.convert_tokens_to_ids(tokens)


# ---------------------------------------------------------------------------
# MIMO-specific MultiModalPackingEncoder subclass
# ---------------------------------------------------------------------------
class MimoMultiModalPackingEncoder(MultiModalPackingEncoder):
    """Subclass that remaps energon batch output to MIMO's forward() signature.

    Key transformation: expand each single ``image_token_id`` placeholder in the
    token stream into ``num_tiles * embeddings_per_tile`` copies so that MIMO's
    ``align_embeddings_by_token_positions`` can do a strict 1:1 scatter.
    """

    def __init__(
        self,
        vision_config: VisionConfig,
        packing_config: PackingConfig,
        tokenizer,
        encoder_name: str = "radio_encoder",
        encoder_input_key: str = "x",
        target_seq_length: Optional[int] = None,
    ):
        super().__init__(vision_config, packing_config, tokenizer)
        self.encoder_name = encoder_name
        self.encoder_input_key = encoder_input_key
        self._target_seq_length = target_seq_length

        # Compute embeddings per tile using the standalone math function.
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

    def batch(self, samples: List[PackedSample]) -> dict:
        """Override to expand image placeholders, build packing_kwargs, and remap to MIMO format.

        Energon's token stream has 1 placeholder per image.
        MIMO needs ``num_tiles * embeddings_per_tile`` placeholders per image
        for 1:1 ``masked_scatter_`` alignment.

        The base class pipeline (preencode → pack_selected_samples) already
        computes ``cu_lengths`` using expanded ``total_len`` values, so the
        cumulative lengths are correct for the MIMO-expanded token stream.

        When ``target_seq_length`` is set, samples whose expanded length would
        exceed the limit are **right-truncated** at image boundaries.

        Returns dict with keys: input_ids, labels, loss_mask, position_ids,
        modality_inputs, and optionally packing_kwargs.
        """
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
            num_tiles = sample.num_tiles  # e.g. [5, 3, 1] for 3 images

            budget = self._target_seq_length  # None means unlimited

            # Expand each single image placeholder → N copies, respecting budget.
            new_tokens = []
            new_labels = []
            img_idx = 0
            truncated = False
            kept_tile_count = 0

            for i, tok in enumerate(tokens.tolist()):
                if tok == image_token_id:
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
                        break
                    new_tokens.append(tok)
                    new_labels.append(labels[i].item())

            if truncated:
                warnings.warn(
                    f"Sample truncated to fit target_seq_length "
                    f"({self._target_seq_length}): kept {len(new_tokens)} of "
                    f"~{len(tokens)} original tokens, {img_idx}/{len(num_tiles)} "
                    f"images ({kept_tile_count} tiles). "
                    f"Consider increasing --total-seq-length or reducing "
                    f"--max-num-tiles.",
                    stacklevel=2,
                )

            all_images.extend(sample.images[:kept_tile_count])
            expanded_tokens_list.append(torch.tensor(new_tokens, dtype=torch.long))
            expanded_labels_list.append(torch.tensor(new_labels, dtype=torch.long))

        # Pad to target length or max length in batch
        max_len = max(len(t) for t in expanded_tokens_list)
        if self._target_seq_length is not None:
            max_len = self._target_seq_length

        B = len(samples)
        tokens_batch = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels_batch = torch.full((B, max_len), ignore_index, dtype=torch.long)

        for i, (t, l) in enumerate(zip(expanded_tokens_list, expanded_labels_list)):
            tokens_batch[i, : len(t)] = t
            labels_batch[i, : len(l)] = l

        loss_mask = (labels_batch != ignore_index).float()
        # Don't train the model to predict <image> tokens — they are special
        # placeholders replaced by vision embeddings, never naturally generated.
        loss_mask[labels_batch == image_token_id] = 0.0
        position_ids = torch.arange(max_len).unsqueeze(0).expand(B, -1).contiguous()

        result = {
            "input_ids": tokens_batch,
            "labels": labels_batch,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        # Only include modality_inputs when there are actual images.
        if all_images:
            imgs = self.tiling_strategy.stack(all_images)[0]  # (total_tiles, C, H, W)
            result["modality_inputs"] = {
                "images": {self.encoder_name: {self.encoder_input_key: imgs}}
            }

        # Build packing_kwargs from base class cu_lengths when packing is active.
        # The base class pipeline computes cu_lengths using expanded total_len,
        # so they match our MIMO-expanded token stream.
        is_packed = any(len(s.cu_lengths) > 2 for s in samples)
        if is_packed:
            # Build per-sample cu_seqlens from PackedSample.cu_lengths.
            # With micro_batch_size=1 (required for packing), B==1.
            assert B == 1, f"Packing requires micro_batch_size=1, got B={B}"
            sample = samples[0]
            cu_seqlens = sample.cu_lengths.to(dtype=torch.int32)

            # Clamp to actual sequence length (cu_lengths are based on expanded
            # total_len which should match, but clamp for safety).
            cu_seqlens = cu_seqlens.clamp(max=max_len)

            # Ensure starts at 0 and ends at max_len.
            if cu_seqlens[0] != 0:
                cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), cu_seqlens])
            if cu_seqlens[-1] != max_len:
                cu_seqlens = torch.cat([cu_seqlens, torch.tensor([max_len], dtype=torch.int32)])

            # Compute per-segment lengths and max segment length.
            segment_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = segment_lens.max()

            result["packing_kwargs"] = {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_kv": cu_seqlens,
                "cu_seqlens_q_padded": cu_seqlens,
                "cu_seqlens_kv_padded": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_kv": max_seqlen,
                "total_tokens": torch.tensor(max_len, dtype=torch.int32),
            }

        return result
