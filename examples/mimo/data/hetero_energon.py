# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon multimodal iterator for heterogeneous MIMO training."""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from examples.mimo.model_providers.nemotron_moe_vlm import NEMOTRON_VISION_ENCODER_KEY
from examples.mimo.training.hetero.topology import get_grid_coordinate, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, is_process_group_member


class ExpandedSample(NamedTuple):
    """Token row and expanded-coordinate segment boundaries."""

    tokens: torch.Tensor
    labels: torch.Tensor
    cu_lengths: torch.Tensor
    kept_tiles: int


def build_energon_iterator(args, topology):
    """Build an Energon iterator for the current rank, or return None if unused."""
    from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

    encoder_grid = topology.encoder_grid
    llm_grid = topology.llm_grid
    encoder_needs_data = is_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )

    if encoder_needs_data:
        return _build_iterator_for_grid(args, encoder_grid)
    if llm_needs_data:
        return _build_iterator_for_grid(args, llm_grid)
    return None


def validate_energon_data_alignment(data_iterator, topology) -> None:
    """Check the first actual-data batch aligns across non-colocated module grids."""
    if not dist.is_initialized():
        return

    signature = data_iterator.peek_signature() if data_iterator is not None else None
    local = (get_current_dp_lane(topology), signature)
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)

    signatures_by_lane = {}
    for lane, candidate in gathered:
        if lane < 0 or candidate is None:
            continue
        signatures_by_lane.setdefault(lane, set()).add(candidate)
    mismatched = {lane: values for lane, values in signatures_by_lane.items() if len(values) > 1}
    if mismatched:
        raise RuntimeError(f"hetero Energon data loaders diverged across grids: {mismatched}")


def get_current_dp_lane(topology) -> int:
    """Return the active module DP lane for this rank, or -1 for inactive ranks."""
    if is_rank_in_grid(topology.encoder_grid):
        return get_grid_coordinate(topology.encoder_grid, "dp")
    if is_rank_in_grid(topology.llm_grid):
        return get_grid_coordinate(topology.llm_grid, "dp")
    return -1


def _build_iterator_for_grid(args, grid):
    """Build a deterministic per-DP-lane loader for one module grid."""
    tp_group = grid.get_pg("tp")
    if get_grid_coordinate(grid, "tp") != 0:
        return EnergonIterator(None, tp_group=tp_group, source_rank=False)

    from megatron.energon import NoCachePool, WorkerConfig, get_loader, get_train_dataset

    tokenizer = _build_tokenizer(args)
    encoder = MimoMultiModalPackingEncoder.from_args(args, tokenizer)
    dp_rank = get_grid_coordinate(grid, "dp")
    worker_config = WorkerConfig(
        rank=dp_rank, world_size=args.llm_dp, num_workers=args.num_workers, data_parallel_group=None
    )
    debug_rank(
        "building energon dataloader "
        f"dp_rank={dp_rank} dp_world={args.llm_dp} batch_size={args.micro_batch_size}"
    )
    dataset = get_train_dataset(
        args.data_path,
        batch_size=args.micro_batch_size,
        task_encoder=encoder,
        worker_config=worker_config,
        packing_buffer_size=args.packing_buffer_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples_per_sequence=args.max_samples_per_sequence,
    )
    return EnergonIterator(
        get_loader(dataset, cache_pool=NoCachePool()), tp_group=tp_group, source_rank=True
    )


def _build_tokenizer(args):
    from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
        MegatronMultimodalTokenizer,
    )

    return MegatronMultimodalTokenizer(
        path=args.tokenizer_model,
        prompt_format=args.tokenizer_prompt_format,
        special_tokens=[args.image_token],
        image_tag_type=args.image_tag_type,
        force_system_message=args.force_system_message,
    )


class TokenizerAdapter:
    """Wrap Megatron's multimodal tokenizer for Energon's tokenizer protocol."""

    def __init__(self, megatron_tokenizer) -> None:
        self._tok = megatron_tokenizer
        self._hf = megatron_tokenizer.tokenizer

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
        """Convert special tokens to ids."""
        return self._tok.convert_tokens_to_ids(tokens)


def _get_multimodal_encoder_base():
    from megatron.energon.task_encoder.multimodal import MultiModalPackingEncoder

    return MultiModalPackingEncoder


def _get_pretraining_conversation_cooker():
    from megatron.energon.task_encoder.cooking import Cooker

    return Cooker(
        cook_pretraining_conversation, has_subflavors={"cook": "pretraining_conversation"}
    )


_MULTIMODAL_PACKING_ENCODER_BASE = _get_multimodal_encoder_base()


def _get_energon_cooker_decorators():
    from megatron.energon.task_encoder.base import stateless
    from megatron.energon.task_encoder.cooking import cooker

    return stateless, cooker


_stateless, _cooker = _get_energon_cooker_decorators()


@_stateless
@_cooker(need_cache=True)
def cook_pretraining_conversation(sample: dict, cache, media_source=None):
    """Cook fragment-style pretraining conversation samples into multimodal text."""
    from megatron.energon.task_encoder.cooking import basic_sample_keys
    from megatron.energon.task_encoder.multimodal.cookers._image_util import to_pil_image
    from megatron.energon.task_encoder.multimodal.sample_types import ImageRef, MultiModalSample

    text_parts = []
    image_refs = []
    for turn in sample["json"]["conversation"]:
        sender = turn.get("sender", "user")
        content_parts = []
        for fragment in turn.get("fragments", []):
            fragment_type = fragment.get("t")
            value = fragment.get("value", "")
            if fragment_type == "image":
                content_parts.append("<image>")
                ext = value.rsplit(".", 1)[-1] if "." in value else ""
                if ext in sample:
                    image = to_pil_image(sample[ext])
                else:
                    image = to_pil_image(cache.get(media_source, value, sample=sample))
                width, height = image.size
                image_refs.append(ImageRef(data=image, width=width, height=height))
            elif fragment_type == "text":
                content_parts.append(value)
        content = "".join(content_parts).strip()
        if content:
            text_parts.append(f"{sender}: {content}")

    return MultiModalSample(
        **basic_sample_keys(sample), text="\n".join(text_parts), images=image_refs
    )


class MimoMultiModalPackingEncoder(_MULTIMODAL_PACKING_ENCODER_BASE):
    """Adapt Energon multimodal packing batches to MIMO's forward signature."""

    cookers = list(_MULTIMODAL_PACKING_ENCODER_BASE.cookers) + [
        _get_pretraining_conversation_cooker()
    ]

    @classmethod
    def from_args(cls, args, tokenizer):
        """Construct the encoder from hetero training args."""
        from megatron.energon.task_encoder.multimodal import PackingConfig, VisionConfig

        vision_config = VisionConfig(
            img_h=args.img_h,
            img_w=args.img_w,
            patch_dim=args.patch_dim,
            vision_model_type=getattr(args, "vision_model_type", "radio"),
            disable_vision_class_token=getattr(args, "disable_vision_class_token", True),
            pixel_shuffle=getattr(args, "pixel_shuffle", True),
            max_num_tiles=args.num_image_tiles,
            use_tiling=getattr(args, "use_tiling", True),
            use_thumbnail=getattr(args, "use_thumbnail", True),
            class_token_len=args.class_token_len,
            conv_merging=getattr(args, "conv_merging", False),
            use_tile_tags=getattr(args, "use_tile_tags", False),
            use_image_break_token=getattr(args, "image_break_token", None) is not None,
            use_area_weighted_aspect_ratio=getattr(args, "use_area_weighted_aspect_ratio", False),
            dynamic_resolution=getattr(args, "dynamic_resolution", False),
        )
        packing_config = PackingConfig(
            seq_length=args.seq_length, pad_id=args.pad_token_id, image_token_id=args.image_token_id
        )
        return cls(
            vision_config=vision_config,
            packing_config=packing_config,
            tokenizer=TokenizerAdapter(tokenizer),
            encoder_name=NEMOTRON_VISION_ENCODER_KEY,
            encoder_input_key="x",
            target_seq_length=args.seq_length,
        )

    def __init__(
        self,
        vision_config,
        packing_config,
        tokenizer,
        encoder_name: str,
        encoder_input_key: str,
        target_seq_length: Optional[int],
    ) -> None:
        super().__init__(vision_config, packing_config, tokenizer)
        from megatron.energon.task_encoder.multimodal.vision_tokens import get_num_image_embeddings

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

    def batch(self, samples: list) -> dict:
        """Expand image placeholders and return a MIMO-compatible batch."""
        image_token_id = self.packing_config.image_token_id
        ignore_index = self.packing_config.ignore_index
        pad_id = self.packing_config.pad_id

        expanded_samples = []
        all_images = []
        for sample in samples:
            expanded = self._expand_sample(sample, image_token_id, ignore_index)
            expanded_samples.append(expanded)
            all_images.extend(sample.images[: expanded.kept_tiles])
        token_rows = [sample.tokens for sample in expanded_samples]
        max_len = self._target_seq_length or max(len(row) for row in token_rows)
        input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(samples), max_len), ignore_index, dtype=torch.long)
        for row_idx, expanded in enumerate(expanded_samples):
            tokens = expanded.tokens
            input_ids[row_idx, : len(tokens)] = tokens
            labels[row_idx, : expanded.labels.numel()] = expanded.labels

        loss_mask = (labels != ignore_index).float()
        loss_mask[labels == image_token_id] = 0.0
        position_ids = torch.arange(max_len).unsqueeze(0).expand(len(samples), -1).contiguous()
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        image_tensor = None
        if all_images:
            image_tensor = self.tiling_strategy.stack(all_images)[0]
            result["modality_inputs"] = {
                "images": {self.encoder_name: {self.encoder_input_key: image_tensor}}
            }
        packing_kwargs = self._build_packing_kwargs(
            [sample.cu_lengths for sample in expanded_samples], max_len
        )
        if packing_kwargs is not None:
            result["packing_kwargs"] = packing_kwargs
        return result

    def _expand_sample(self, sample, image_token_id: int, ignore_index: int) -> ExpandedSample:
        """Expand one image token into one placeholder per image embedding."""
        if not hasattr(sample, "labels"):
            raise RuntimeError("Energon multimodal samples must provide labels")

        tokens = []
        labels = []
        image_idx = 0
        kept_tiles = 0
        budget = self._target_seq_length
        sample_labels = sample.labels.reshape(-1).tolist()
        if len(sample_labels) < sample.tokens.numel():
            raise RuntimeError(
                "Energon multimodal sample labels must be at least as long as tokens"
            )

        for idx, token in enumerate(sample.tokens.tolist()):
            if token != image_token_id:
                if budget is not None and len(tokens) + 1 > budget:
                    break
                tokens.append(token)
                labels.append(int(sample_labels[idx]))
                continue

            num_tiles = sample.num_tiles[image_idx] if image_idx < len(sample.num_tiles) else 1
            num_placeholders = num_tiles * self._embeddings_per_tile
            if budget is not None and len(tokens) + num_placeholders > budget:
                warnings.warn(
                    "Energon sample truncated at an image boundary to fit "
                    f"target sequence length {budget}.",
                    stacklevel=2,
                )
                break
            tokens.extend([image_token_id] * num_placeholders)
            labels.extend([ignore_index] * num_placeholders)
            kept_tiles += num_tiles
            image_idx += 1

        return ExpandedSample(
            tokens=torch.tensor(tokens, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            cu_lengths=torch.tensor(
                self._expanded_boundaries(sample.cu_lengths, len(tokens)), dtype=torch.long
            ),
            kept_tiles=kept_tiles,
        )

    @staticmethod
    def _expanded_boundaries(cu_lengths: torch.Tensor, expanded_token_count: int) -> list[int]:
        """Normalize Energon cumulative lengths in expanded sequence coordinates."""
        boundaries = cu_lengths.to(dtype=torch.long).flatten().tolist()
        boundaries = [min(max(int(value), 0), expanded_token_count) for value in boundaries]
        if not boundaries or boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries[-1] != expanded_token_count:
            boundaries.append(expanded_token_count)

        deduped = []
        for boundary in boundaries:
            if not deduped or boundary != deduped[-1]:
                deduped.append(boundary)
        return deduped

    @staticmethod
    def _build_packing_kwargs(
        cu_lengths_by_sample: list[torch.Tensor], max_len: int
    ) -> Optional[dict]:
        """Build packed-sequence metadata when Energon selected packed samples."""
        is_packed = any(cu_lengths.numel() > 2 for cu_lengths in cu_lengths_by_sample)
        if not is_packed:
            return None
        if len(cu_lengths_by_sample) != 1:
            raise RuntimeError("Energon packing requires micro_batch_size=1")

        cu_seqlens = cu_lengths_by_sample[0].to(dtype=torch.int32).clamp(max=max_len)
        if cu_seqlens[0] != 0:
            cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), cu_seqlens])
        if cu_seqlens[-1] != max_len:
            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([max_len], dtype=torch.int32)])
        segment_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = segment_lens.max()
        return {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens,
            "cu_seqlens_q_padded": cu_seqlens,
            "cu_seqlens_kv_padded": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_kv": max_seqlen,
            "total_tokens": max_len,
        }


class EnergonIterator:
    """Endless wrapper around an Energon dataloader with TP-rank-0 ownership."""

    def __init__(self, dataloader, tp_group=None, source_rank: bool = True) -> None:
        self._dataloader = dataloader
        self._iterator = iter(dataloader) if dataloader is not None else None
        self._tp_group = tp_group
        self._source_rank = source_rank
        self._prefetched = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._prefetched is not None:
            batch = self._prefetched
            self._prefetched = None
            return batch

        batch = self._next_local_batch() if self._source_rank else None
        if is_process_group_member(self._tp_group) and self._tp_group.size() > 1:
            obj = [batch]
            dist.broadcast_object_list(obj, src=self._tp_source_rank(), group=self._tp_group)
            batch = obj[0]
        return batch

    def peek_signature(self):
        """Read and retain the next batch, returning a compact deterministic signature."""
        if self._prefetched is None:
            self._prefetched = next(self)
        return self._batch_signature(self._prefetched)

    def _next_local_batch(self):
        """Read the next local Energon batch on the TP source rank."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def _tp_source_rank(self) -> int:
        """Return the global source rank for the local TP batch broadcast."""
        if hasattr(dist, "get_global_rank"):
            return dist.get_global_rank(self._tp_group, 0)
        return dist.get_process_group_ranks(self._tp_group)[0]

    @classmethod
    def _batch_signature(cls, batch: dict) -> tuple[int, ...]:
        """Return a compact signature for cross-grid data-alignment checks."""
        image_tensor = cls._nested_get(batch, ("modality_inputs", "images"))
        if isinstance(image_tensor, dict):
            image_tensor = cls._first_tensor(image_tensor)
        packing_kwargs = batch.get("packing_kwargs")
        return (
            cls._checksum_tensor(batch.get("input_ids")),
            cls._checksum_tensor(batch.get("labels")),
            int(batch.get("loss_mask", torch.zeros(1)).sum().item()),
            0 if image_tensor is None else int(image_tensor.shape[0]),
            cls._checksum_tensor(image_tensor),
            cls._checksum_packing_kwargs(packing_kwargs),
        )

    @staticmethod
    def _nested_get(value: dict, keys: tuple[str, ...]):
        """Return a nested dict value if every key exists."""
        current = value
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @classmethod
    def _first_tensor(cls, value):
        """Return the first tensor inside a nested mapping."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = cls._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @classmethod
    def _checksum_packing_kwargs(cls, packing_kwargs: Optional[dict]) -> int:
        """Checksum packed-sequence metadata used by the language model."""
        if packing_kwargs is None:
            return 0
        checksum = 0
        for key in sorted(packing_kwargs):
            value = packing_kwargs[key]
            if isinstance(value, torch.Tensor):
                value_checksum = cls._checksum_tensor(value)
            elif value is None:
                value_checksum = 0
            else:
                value_checksum = int(value)
            checksum = (checksum * 131 + value_checksum) % 2_147_483_647
        return checksum

    @staticmethod
    def _checksum_tensor(tensor: Optional[torch.Tensor]) -> int:
        """Return a stable, bounded checksum for a CPU tensor-like batch field."""
        if tensor is None or tensor.numel() == 0:
            return 0
        values = tensor.detach().reshape(-1)
        stride = max(values.numel() // 4096, 1)
        values = values[::stride]
        if values.is_floating_point():
            values = (values.float() * 1024).to(dtype=torch.long)
        else:
            values = values.to(dtype=torch.long)
        positions = torch.arange(1, values.numel() + 1, dtype=torch.long, device=values.device)
        return int(((values * positions).sum() % 2_147_483_647).item())
