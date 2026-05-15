# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous-rank wrapper for the MIMO Energon multimodal provider."""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Optional

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.topology import get_grid_coordinate, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, is_process_group_member


def build_energon_iterator(args, topology):
    """Build an Energon iterator for the current rank, or return None if unused."""
    from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

    encoder_grid = topology.encoder_grid
    llm_grid = topology.llm_grid
    encoder_needs_data = (
        encoder_grid is not None
        and is_rank_in_grid(encoder_grid)
        and is_pp_first_stage(encoder_grid.get_pg("pp"))
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )

    if encoder_needs_data:
        return _build_encoder_iterator(args, encoder_grid)
    if llm_needs_data:
        return _build_llm_iterator(args, llm_grid)
    return None


def validate_energon_data_alignment(data_iterator, _topology) -> None:
    """Check the first actual-data batch aligns across non-colocated module grids."""
    if not dist.is_initialized():
        return

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(
        gathered, data_iterator.peek_alignment() if data_iterator is not None else None
    )

    encoder_signatures_by_lane = {}
    llm_signatures_by_lane = {}
    for candidate in gathered:
        if candidate is None:
            continue
        target = (
            encoder_signatures_by_lane if candidate["role"] == "encoder" else llm_signatures_by_lane
        )
        for lane, signature in zip(candidate["llm_lanes"], candidate["signatures"]):
            target.setdefault(lane, set()).add(signature)

    mismatched = {}
    for lane in sorted(set(encoder_signatures_by_lane) | set(llm_signatures_by_lane)):
        encoder_values = encoder_signatures_by_lane.get(lane, set())
        llm_values = llm_signatures_by_lane.get(lane, set())
        if len(encoder_values) != 1 or len(llm_values) != 1 or encoder_values != llm_values:
            mismatched[lane] = {"encoder": sorted(encoder_values), "llm": sorted(llm_values)}
    if mismatched:
        raise RuntimeError(f"hetero Energon data loaders diverged across grids: {mismatched}")


def _build_llm_iterator(args, grid):
    """Build the single-lane LLM iterator for this grid coordinate."""
    tp_group = grid.get_pg("tp")
    if get_grid_coordinate(grid, "tp") != 0:
        lane = get_grid_coordinate(grid, "dp")
        return EnergonIterator(
            None, tp_group=tp_group, source_rank=False, alignment_role="llm", llm_lanes=[lane]
        )

    lane = get_grid_coordinate(grid, "dp")
    return _build_single_lane_iterator(
        args, tp_group=tp_group, lane=lane, role="llm", random_seed=args.seed + lane
    )


def _build_encoder_iterator(args, grid):
    """Build the encoder iterator, composing LLM-lane samples for DP fan-out."""
    tp_group = grid.get_pg("tp")
    encoder_dp_rank = get_grid_coordinate(grid, "dp")
    llm_lanes = _llm_lanes_for_encoder_rank(args, encoder_dp_rank)
    if get_grid_coordinate(grid, "tp") != 0:
        return EnergonIterator(
            None,
            tp_group=tp_group,
            source_rank=False,
            alignment_role="encoder",
            llm_lanes=llm_lanes,
        )

    if len(llm_lanes) == 1:
        return _build_single_lane_iterator(
            args,
            tp_group=tp_group,
            lane=llm_lanes[0],
            role="encoder",
            random_seed=args.seed + llm_lanes[0],
        )

    lane_iterators = [
        _build_single_lane_iterator(
            args, tp_group=None, lane=lane, role="encoder-component", random_seed=args.seed + lane
        )
        for lane in llm_lanes
    ]

    def next_encoder_batch():
        batches = [next(iterator) for iterator in lane_iterators]
        signatures = [EnergonIterator._batch_signature(batch) for batch in batches]
        return _combine_encoder_batches(batches), signatures

    return EnergonIterator(
        None,
        tp_group=tp_group,
        source_rank=True,
        local_batch_fn=next_encoder_batch,
        alignment_role="encoder",
        llm_lanes=llm_lanes,
    )


def _llm_lanes_for_encoder_rank(args, encoder_dp_rank: int) -> list[int]:
    """Return the contiguous LLM DP lanes owned by one encoder DP lane."""
    scale = args.llm_dp // args.encoder_dp
    start = encoder_dp_rank * scale
    return list(range(start, start + scale))


def _build_single_lane_iterator(args, tp_group, lane: int, role: str, random_seed: int):
    """Build a deterministic loader for one LLM data lane."""
    from examples.mimo.data.energon_multimodal_provider import build_multimodal_encoder
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    tokenizer = _build_tokenizer(args)
    encoder = build_multimodal_encoder(
        args,
        tokenizer,
        encoder_name=getattr(args, "vision_encoder_key", "radio_encoder"),
        encoder_input_key="x",
    )
    worker_config = WorkerConfig(
        rank=lane, world_size=args.llm_dp, num_workers=args.num_workers, data_parallel_group=None
    )
    debug_rank(
        "building energon dataloader "
        f"role={role} lane={lane} dp_world={args.llm_dp} batch_size={args.micro_batch_size}"
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
        get_loader(dataset),
        tp_group=tp_group,
        source_rank=True,
        random_seed=random_seed,
        alignment_role="encoder" if role.startswith("encoder") else "llm",
        llm_lanes=[lane],
    )


def _combine_encoder_batches(batches: list[dict]) -> dict:
    """Combine LLM-lane batches into one encoder batch and drop LLM-only metadata."""
    if not batches:
        raise RuntimeError("cannot combine an empty encoder batch list")

    combined = {}
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        values = [batch.get(key) for batch in batches if batch.get(key) is not None]
        if values:
            combined[key] = torch.cat(values, dim=0)

    modality_values = [
        batch.get("modality_inputs")
        for batch in batches
        if batch.get("modality_inputs") is not None
    ]
    if modality_values:
        combined["modality_inputs"] = _concat_nested_tensors(modality_values)

    return combined


def _concat_nested_tensors(values):
    """Concatenate a list of matching nested tensor structures along leading dim."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(present, dim=0)
    if isinstance(first, dict):
        keys = set().union(*(value.keys() for value in present if isinstance(value, dict)))
        merged = {}
        for key in sorted(keys):
            value = _concat_nested_tensors([item.get(key) for item in present])
            if value is not None:
                merged[key] = value
        return merged
    raise TypeError(f"cannot concatenate encoder batch value of type {type(first).__name__}")


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


class EnergonIterator:
    """Endless wrapper around an Energon dataloader with TP-rank-0 ownership."""

    def __init__(
        self,
        dataloader,
        tp_group=None,
        source_rank: bool = True,
        random_seed: Optional[int] = None,
        local_batch_fn: Optional[Callable[[], dict]] = None,
        alignment_role: Optional[str] = None,
        llm_lanes: Optional[list[int]] = None,
    ) -> None:
        self._dataloader = dataloader
        self._iterator = None
        self._tp_group = tp_group
        self._source_rank = source_rank
        self._local_batch_fn = local_batch_fn
        self._alignment_role = alignment_role
        self._llm_lanes = llm_lanes or []
        self._prefetched = None
        self._prefetched_component_signatures = None
        self._local_component_signatures = None
        self._python_random_state = None
        if random_seed is not None:
            rng = random.Random(random_seed)
            self._python_random_state = rng.getstate()

    def __iter__(self):
        return self

    def __next__(self):
        if self._prefetched is not None:
            batch = self._prefetched
            self._prefetched = None
            return batch

        batch = self._next_local_batch() if self._source_rank else None
        component_signatures = self._current_component_signatures(batch)
        if is_process_group_member(self._tp_group) and self._tp_group.size() > 1:
            obj = [(batch, component_signatures)]
            dist.broadcast_object_list(obj, src=self._tp_source_rank(), group=self._tp_group)
            batch, component_signatures = obj[0]
        self._prefetched_component_signatures = component_signatures
        return batch

    def peek_alignment(self):
        """Read and retain the next batch, returning lane signatures from TP source ranks."""
        if self._prefetched is None:
            self._prefetched = next(self)
        if not self._source_rank or self._alignment_role is None:
            return None
        signatures = self._prefetched_component_signatures
        if signatures is None:
            signatures = [self._batch_signature(self._prefetched)]
        return {
            "role": self._alignment_role,
            "llm_lanes": self._llm_lanes,
            "signatures": signatures,
        }

    def _next_local_batch(self):
        """Read the next local Energon batch on the TP source rank."""
        if self._python_random_state is None:
            result = self._read_next_local_batch()
            return self._extract_batch_and_signatures(result)

        global_random_state = random.getstate()
        try:
            random.setstate(self._python_random_state)
            result = self._read_next_local_batch()
            batch = self._extract_batch_and_signatures(result)
            self._python_random_state = random.getstate()
            return batch
        finally:
            random.setstate(global_random_state)

    def _extract_batch_and_signatures(self, result):
        """Handle local batch providers that also return component signatures."""
        self._local_component_signatures = None
        if isinstance(result, tuple) and len(result) == 2:
            batch, signatures = result
            self._local_component_signatures = signatures
            return batch
        return result

    def _read_next_local_batch(self):
        """Read from the underlying dataloader, cycling at epoch boundaries."""
        if self._local_batch_fn is not None:
            return self._local_batch_fn()
        if self._iterator is None:
            self._iterator = iter(self._dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def _current_component_signatures(self, batch):
        """Return per-lane signatures for the current batch if they can be inferred."""
        if batch is None:
            return None
        if self._local_component_signatures is not None:
            return self._local_component_signatures
        return [self._batch_signature(batch)]

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
            elif isinstance(value, str):
                value_checksum = sum(value.encode("utf-8"))
            else:
                value_checksum = int(value)
            checksum = (checksum * 131 + value_checksum) % 2_147_483_647
        return checksum

    @staticmethod
    def _checksum_tensor(tensor: Optional[torch.Tensor]) -> int:
        """Return a stable full-tensor checksum for a CPU tensor-like batch field."""
        if tensor is None or tensor.numel() == 0:
            return 0
        tensor = tensor.detach().cpu().contiguous()
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
        return int.from_bytes(digest.digest(), byteorder="big", signed=False)
