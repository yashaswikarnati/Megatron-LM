# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Energon multimodal data provider for MIMO.

Bridges energon's ``MultiModalPackingEncoder`` (which produces 1 image-placeholder
per ``<image>`` tag) to MIMO's contract (which requires N placeholders per image,
where N = num_tiles * embeddings_per_tile, for 1:1 ``masked_scatter_`` alignment).

Usage from train.py::

    _DATASET_PROVIDERS["energon_multimodal"] = energon_multimodal_dataloader_provider
"""

import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import torch

from megatron.energon.task_encoder.multimodal import (
    MultiModalPackingEncoder,
    PackingConfig,
    VisionConfig,
)
from megatron.energon.task_encoder.multimodal.sample_types import PackedSample
from megatron.energon.task_encoder.multimodal.vision_tokens import (
    get_num_image_embeddings,
)


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
                "images": {
                    self.encoder_name: {self.encoder_input_key: imgs},
                }
            }

        # Build packing_kwargs from base class cu_lengths when packing is active.
        # The base class pipeline computes cu_lengths using expanded total_len,
        # so they match our MIMO-expanded token stream.
        is_packed = any(len(s.cu_lengths) > 2 for s in samples)
        if is_packed:
            # Build per-sample cu_seqlens from PackedSample.cu_lengths.
            # With micro_batch_size=1 (required for packing), B==1.
            assert B == 1, (
                f"Packing requires micro_batch_size=1, got B={B}"
            )
            sample = samples[0]
            cu_seqlens = sample.cu_lengths.to(dtype=torch.int32)

            # Clamp to actual sequence length (cu_lengths are based on expanded
            # total_len which should match, but clamp for safety).
            cu_seqlens = cu_seqlens.clamp(max=max_len)

            # Ensure starts at 0 and ends at max_len.
            if cu_seqlens[0] != 0:
                cu_seqlens = torch.cat(
                    [torch.tensor([0], dtype=torch.int32), cu_seqlens]
                )
            if cu_seqlens[-1] != max_len:
                cu_seqlens = torch.cat(
                    [cu_seqlens, torch.tensor([max_len], dtype=torch.int32)]
                )

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


# ---------------------------------------------------------------------------
# EnergonDataloader wrapper
# ---------------------------------------------------------------------------
def _cyclic_iter(iterable):
    """Endlessly cycle over an iterable."""
    while True:
        for x in iterable:
            yield x


class EnergonDataloader:
    """Wraps an energon dataloader for Megatron's cyclic iteration."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(_cyclic_iter(dataloader)) if dataloader else iter([])

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return self._iter


# ---------------------------------------------------------------------------
# Factory: train_valid_test_dataloaders_provider
# ---------------------------------------------------------------------------
def train_valid_test_dataloaders_provider(train_val_test_num_samples, **kwargs):
    """Build energon multimodal dataloader for MIMO training.

    Registered in train.py as ``_DATASET_PROVIDERS["energon_multimodal"]``.
    """
    from megatron.core import parallel_state
    from megatron.core.parallel_state import get_tensor_model_parallel_rank
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset
    from megatron.training import get_args
    from megatron.training.global_vars import get_tokenizer

    args = get_args()

    # Only build on TP rank 0; other ranks get None (broadcast in get_batch).
    if get_tensor_model_parallel_rank() != 0:
        return None, None, None

    tokenizer = get_tokenizer()
    tok_adapter = _TokenizerAdapter(tokenizer)

    # Derive image_token_id and pad_id from the tokenizer (matches remote reference).
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    pad_id = tokenizer.pad

    vision_config = VisionConfig(
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        vision_model_type=getattr(args, "vision_model_type", "radio"),
        disable_vision_class_token=getattr(args, "disable_vision_class_token", False),
        pixel_shuffle=getattr(args, "pixel_shuffle", False),
        max_num_tiles=getattr(args, "max_num_tiles", 1),
        use_tiling=getattr(args, "use_tiling", False),
        use_thumbnail=getattr(args, "use_thumbnail", False),
        class_token_len=getattr(args, "class_token_len", None) or 1,
        conv_merging=getattr(args, "conv_merging", False),
        use_tile_tags=getattr(args, "use_tile_tags", False),
        use_image_break_token=getattr(args, "image_break_token", None) is not None,
        use_area_weighted_aspect_ratio=getattr(args, "use_area_weighted_aspect_ratio", False),
        dynamic_resolution=getattr(args, "dynamic_resolution", False),
    )

    target_seq_length = args.total_seq_length

    packing_config = PackingConfig(
        seq_length=target_seq_length,
        pad_id=pad_id,
        image_token_id=image_token_id,
    )

    encoder = MimoMultiModalPackingEncoder(
        vision_config=vision_config,
        packing_config=packing_config,
        tokenizer=tok_adapter,
        encoder_name="radio_encoder",
        encoder_input_key="x",
        target_seq_length=target_seq_length,
    )

    worker_config = WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
    )

    dname = args.data_path[0] if isinstance(args.data_path, list) else args.data_path
    # Packing disabled — MIMO model does not support packing yet.
    packing_buffer_size = getattr(args, "packing_buffer_size", None)

    train_ds = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=encoder,
        worker_config=worker_config,
        packing_buffer_size=packing_buffer_size,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
    )
    train_dataloader = get_loader(train_ds)

    return EnergonDataloader(train_dataloader), None, None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def _test_standalone(
    blend_yaml: str,
    batch_size: int = 1,
    num_batches: int = 5,
    seq_length: int = 4096,
    packing_buffer_size: int = 32,
):
    """Test the data pipeline with packing verification.

    Usage::

        python energon_multimodal_provider.py <path-to-blend.yaml> [batch_size] [num_batches] [seq_length]
    """
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset
    from transformers import AutoTokenizer

    tokenizer_model = os.environ.get(
        "TOKENIZER_MODEL",
        "/home/sasatheesh/data/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67",
    )
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        image_token_id = 128256
        print(f"WARNING: <image> token not in vocab, using fallback id {image_token_id}")
    print(f"image_token_id = {image_token_id}")

    vision_config = VisionConfig(
        img_h=512,
        img_w=512,
        patch_dim=16,
        vision_model_type="radio",
        disable_vision_class_token=True,
        pixel_shuffle=True,
        max_num_tiles=12,
        use_tiling=True,
        use_thumbnail=True,
    )

    packing_config = PackingConfig(
        seq_length=seq_length,
        pad_id=tokenizer.pad_token_id or 0,
        image_token_id=image_token_id,
    )

    encoder = MimoMultiModalPackingEncoder(
        vision_config=vision_config,
        packing_config=packing_config,
        tokenizer=tokenizer,
        encoder_name="radio_encoder",
        encoder_input_key="x",
        target_seq_length=seq_length,
    )

    print(f"Embeddings per tile: {encoder._embeddings_per_tile}")
    print(f"Sequence length: {seq_length}")
    print(f"Packing buffer size: {packing_buffer_size}")

    worker_config = WorkerConfig.default_worker_config(0)
    dataset = get_train_dataset(
        blend_yaml,
        batch_size=batch_size,
        task_encoder=encoder,
        worker_config=worker_config,
        packing_buffer_size=packing_buffer_size,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
    )
    loader = get_loader(dataset)

    print(f"\n{'='*60}")
    print(f"Loading batches from: {blend_yaml}")
    print(f"{'='*60}\n")

    errors = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]
        has_images = "modality_inputs" in batch

        B, S = input_ids.shape
        n_image_tokens = (input_ids == image_token_id).sum(dim=1)
        n_loss_tokens = loss_mask.sum(dim=1)
        n_pad_tokens = (input_ids == (tokenizer.pad_token_id or 0)).sum(dim=1)

        print(f"--- Batch {i} ---")
        print(f"  shape: B={B}, S={S}")
        print(f"  image tokens per sample: {n_image_tokens.tolist()}")
        print(f"  loss tokens per sample:  {n_loss_tokens.int().tolist()}")
        print(f"  pad tokens per sample:   {n_pad_tokens.tolist()}")

        # 1. Verify image token / tile consistency
        if has_images:
            imgs = batch["modality_inputs"]["images"]["radio_encoder"]["x"]
            total_tiles = imgs.shape[0]
            total_image_toks = n_image_tokens.sum().item()
            expected_image_toks = total_tiles * encoder._embeddings_per_tile
            ok = total_image_toks == expected_image_toks
            status = "OK" if ok else "MISMATCH"
            print(f"  images: {imgs.shape}  tiles={total_tiles}  "
                  f"img_tokens={total_image_toks}  expected={expected_image_toks}  [{status}]")
            if not ok:
                errors.append(f"Batch {i}: image token mismatch")
        else:
            print(f"  images: text-only batch")

        # 2. Verify packing_kwargs
        if "packing_kwargs" in batch:
            pk = batch["packing_kwargs"]
            cu_q = pk["cu_seqlens_q"]
            cu_kv = pk["cu_seqlens_kv"]
            max_q = pk["max_seqlen_q"]
            max_kv = pk["max_seqlen_kv"]

            num_segments = len(cu_q) - 1
            segment_lens = cu_q[1:] - cu_q[:-1]

            print(f"  packing_kwargs:")
            print(f"    cu_seqlens_q:  {cu_q.tolist()}")
            print(f"    num_segments:  {num_segments}")
            print(f"    segment_lens:  {segment_lens.tolist()}")
            print(f"    max_seqlen_q:  {max_q.item()}")

            # Verify cu_seqlens starts at 0
            if cu_q[0].item() != 0:
                errors.append(f"Batch {i}: cu_seqlens_q doesn't start at 0")
                print(f"    ERROR: cu_seqlens_q doesn't start at 0!")

            # Verify cu_seqlens ends at S
            if cu_q[-1].item() != S:
                errors.append(f"Batch {i}: cu_seqlens_q doesn't end at S={S}, got {cu_q[-1].item()}")
                print(f"    ERROR: cu_seqlens_q ends at {cu_q[-1].item()}, expected {S}!")

            # Verify monotonically increasing
            if not torch.all(segment_lens > 0):
                errors.append(f"Batch {i}: cu_seqlens_q not strictly increasing")
                print(f"    ERROR: cu_seqlens_q not strictly increasing!")

            # Verify max_seqlen matches
            actual_max = segment_lens.max().item()
            if max_q.item() != actual_max:
                errors.append(f"Batch {i}: max_seqlen_q mismatch: {max_q.item()} vs {actual_max}")
                print(f"    ERROR: max_seqlen_q={max_q.item()} but max segment={actual_max}!")

            # Verify q == kv
            if not torch.equal(cu_q, cu_kv):
                errors.append(f"Batch {i}: cu_seqlens_q != cu_seqlens_kv")
                print(f"    ERROR: cu_seqlens_q != cu_seqlens_kv!")

            # Verify content within segments: check each segment has valid tokens
            for seg_idx in range(num_segments):
                start = cu_q[seg_idx].item()
                end = cu_q[seg_idx + 1].item()
                seg_tokens = input_ids[0, start:end]
                seg_loss = loss_mask[0, start:end]
                seg_img_count = (seg_tokens == image_token_id).sum().item()
                seg_loss_count = seg_loss.sum().int().item()
                seg_pad_count = (seg_tokens == (tokenizer.pad_token_id or 0)).sum().item()
                print(f"    segment {seg_idx}: [{start}:{end}] len={end-start}  "
                      f"img_tok={seg_img_count}  loss_tok={seg_loss_count}  pad={seg_pad_count}")
        else:
            print(f"  packing_kwargs: NOT present (single sample, no packing)")

        print()

    print(f"{'='*60}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All checks passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python energon_multimodal_provider.py <blend.yaml> "
            "[batch_size] [num_batches] [seq_length]"
        )
        sys.exit(1)

    blend_yaml = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    num_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    seq_length = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    _test_standalone(blend_yaml, batch_size, num_batches, seq_length)
