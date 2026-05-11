# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous-rank wrapper for the MIMO Energon multimodal provider."""

from __future__ import annotations

import hashlib
import random
from typing import Optional

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.topology import get_grid_coordinate, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, is_process_group_member


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

    from examples.mimo.data.energon_multimodal_provider import build_multimodal_encoder
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    tokenizer = _build_tokenizer(args)
    encoder = build_multimodal_encoder(
        args,
        tokenizer,
        encoder_name=getattr(args, "vision_encoder_key", "radio_encoder"),
        encoder_input_key="x",
    )
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
        get_loader(dataset), tp_group=tp_group, source_rank=True, random_seed=args.seed + dp_rank
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


class EnergonIterator:
    """Endless wrapper around an Energon dataloader with TP-rank-0 ownership."""

    def __init__(
        self, dataloader, tp_group=None, source_rank: bool = True, random_seed: Optional[int] = None
    ) -> None:
        self._dataloader = dataloader
        self._iterator = iter(dataloader) if dataloader is not None else None
        self._tp_group = tp_group
        self._source_rank = source_rank
        self._prefetched = None
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
        if self._python_random_state is None:
            return self._read_next_local_batch()

        global_random_state = random.getstate()
        try:
            random.setstate(self._python_random_state)
            batch = self._read_next_local_batch()
            self._python_random_state = random.getstate()
            return batch
        finally:
            random.setstate(global_random_state)

    def _read_next_local_batch(self):
        """Read from the underlying dataloader, cycling at epoch boundaries."""
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
        """Return a stable full-tensor checksum for a CPU tensor-like batch field."""
        if tensor is None or tensor.numel() == 0:
            return 0
        tensor = tensor.detach().cpu().contiguous()
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(memoryview(tensor.numpy()).cast("B"))
        return int.from_bytes(digest.digest(), byteorder="big", signed=False)
