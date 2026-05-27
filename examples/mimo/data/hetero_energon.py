# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous-rank wrapper for the MIMO Energon multimodal provider."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import deque
from typing import Callable, Optional

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.topology import get_grid_coordinate, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, is_process_group_member
from megatron.core.packed_seq_params import PackedSeqParams


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
            # energon's WorkerConfig(rank=lane, world_size=llm_dp) already
            # salts per-rank, so the seed here must be unsalted.
            random_seed=args.seed,
        )

    return _build_routed_encoder_iterator(
        args, tp_group=tp_group, encoder_dp_rank=encoder_dp_rank, llm_lanes=llm_lanes
    )


def _route_samples_to_lanes(
    loader_iter,
    *,
    lanes_per_encoder: int,
    lane_offset: int,
    num_workers_per_lane: int,
    encoder_dp_rank: int,
    pending_by_lane: list,
    max_pulls_per_step: int,
    provenance_key: str,
) -> tuple[list, int]:
    """Pull samples from a single multiplexed loader and route each one to its LLM lane.

    Samples are routed by reading the producing worker's
    ``WorkerConfig.global_worker_id()``, which the encoder batcher stamps under
    ``provenance_key``. The mapping from worker id back to local lane is:

        global_worker_id = encoder_dp_rank * num_workers_enc + local_worker_id
        global_llm_lane  = global_worker_id // num_workers_per_lane
        local_lane       = global_llm_lane - lane_offset

    Surplus samples (a worker yields a second sample for a lane that's already
    filled this step) are stashed in ``pending_by_lane`` and consumed on the
    next encoder step. ``max_pulls_per_step`` bounds the loop so a stuck or
    skewed worker pool fails loudly instead of silently stalling.

    Returns ``(lane_batches, pulls)`` where ``lane_batches[lane]`` is the sample
    routed to local lane ``lane``.
    """
    lane_batches: list = [None] * lanes_per_encoder
    filled = 0
    for lane in range(lanes_per_encoder):
        if pending_by_lane[lane]:
            lane_batches[lane] = pending_by_lane[lane].popleft()
            filled += 1
    pulls = 0
    while filled < lanes_per_encoder:
        if pulls >= max_pulls_per_step:
            missing = [i for i, b in enumerate(lane_batches) if b is None]
            raise RuntimeError(
                f"encoder dataloader did not yield samples for local_lanes={missing} "
                f"in {max_pulls_per_step} pulls (encoder_dp_rank={encoder_dp_rank}); "
                "check Energon worker rotation contract"
            )
        sample = next(loader_iter)
        pulls += 1
        wid = sample.pop(provenance_key, None)
        if wid is None:
            raise RuntimeError(
                f"encoder sample missing {provenance_key!r}; "
                "ensure build_multimodal_encoder was called with attach_provenance=True"
            )
        global_llm_lane = wid // num_workers_per_lane
        local_lane = global_llm_lane - lane_offset
        if not (0 <= local_lane < lanes_per_encoder):
            raise RuntimeError(
                f"worker_id={wid} maps to global_llm_lane={global_llm_lane}, "
                f"outside encoder rank {encoder_dp_rank} range "
                f"[{lane_offset}, {lane_offset + lanes_per_encoder})"
            )
        if lane_batches[local_lane] is None:
            lane_batches[local_lane] = sample
            filled += 1
        else:
            pending_by_lane[local_lane].append(sample)
    return lane_batches, pulls


def _build_routed_encoder_iterator(args, tp_group, encoder_dp_rank, llm_lanes):
    """Build one Energon iterator per encoder rank and route samples back to LLM lanes.

    The previous implementation built ``lanes_per_encoder`` independent Energon
    iterators per encoder rank — one per LLM data lane — which produces
    ``lanes_per_encoder × num_workers`` shard-open events at construction.
    This collapses that to a single Energon iterator with
    ``num_workers = args.num_workers * lanes_per_encoder``; each emitted batch
    is routed to its owning lane using the producing worker's
    ``WorkerConfig.global_worker_id()`` that the encoder batcher stamps onto
    every batch.

    Bit-wise sample parity with the per-lane iterator path is preserved by
    Energon's design: ``global_workers = world_size * num_workers`` is invariant
    under this reshape and per-worker seeds depend only on ``global_worker_id``
    and ``seed_offset`` (see ``megatron/energon/worker.py``), so each worker
    here produces the same shards in the same order as the per-lane worker it
    replaces.
    """
    from examples.mimo.data.energon_multimodal_provider import (
        MimoMultiModalPackingEncoder,
        build_multimodal_encoder,
    )
    from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
    from megatron.energon.cache.no_cache import NoCachePool

    if args.num_workers < 1:
        raise ValueError(
            "routed encoder iterator requires args.num_workers >= 1 "
            "(global_worker_id -> lane mapping divides by num_workers_per_lane); "
            f"got {args.num_workers}"
        )
    lanes_per_encoder = len(llm_lanes)
    num_workers_per_lane = args.num_workers
    num_workers_enc = num_workers_per_lane * lanes_per_encoder
    lane_offset = llm_lanes[0]

    tokenizer = _build_tokenizer(args)
    encoder = build_multimodal_encoder(
        args,
        tokenizer,
        encoder_name=getattr(args, "vision_encoder_key", "radio_encoder"),
        encoder_input_key="x",
        attach_provenance=True,
    )
    worker_config = WorkerConfig(
        rank=encoder_dp_rank,
        world_size=args.encoder_dp,
        num_workers=num_workers_enc,
        data_parallel_group=None,
    )
    debug_rank(
        "building routed encoder dataloader "
        f"encoder_dp_rank={encoder_dp_rank} encoder_dp={args.encoder_dp} "
        f"num_workers_enc={num_workers_enc} lanes_per_encoder={lanes_per_encoder} "
        f"lane_offset={lane_offset}"
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
    loader = get_savable_loader(
        dataset,
        cache_pool=NoCachePool(),
        watchdog_timeout_seconds=5 * 60,
        watchdog_initial_timeout_seconds=5 * 60,
    )

    loader_iter_holder: list = [None]
    # Dense integer keys (0..lanes_per_encoder-1) → use a list so the hot-path
    # routing in ``_route_samples_to_lanes`` does O(1) array indexing rather
    # than dict probing.
    pending_by_lane: list[deque] = [deque() for _ in range(lanes_per_encoder)]
    # Energon's SavableDataLoader rotates through every worker in one round,
    # so a step worst case needs ``num_workers_enc`` pulls to fill every lane
    # (one batch per worker, including the surplus to lanes that filled
    # early). The 4× factor adds slack for transient rotation skew; we cap
    # below by 2*num_workers_enc so configurations with high
    # ``num_workers_per_lane`` aren't bounded too tightly. A genuine stall
    # surfaces as a loud failure in ``_route_samples_to_lanes``.
    max_pulls_per_step = max(4 * lanes_per_encoder, 2 * num_workers_enc)
    provenance_key = MimoMultiModalPackingEncoder.PROVENANCE_KEY

    def get_loader_iter():
        if loader_iter_holder[0] is None:
            loader_iter_holder[0] = iter(loader)
        return loader_iter_holder[0]

    def save_route_state():
        return {"pending_by_lane": [list(queue) for queue in pending_by_lane]}

    def restore_route_state(state):
        saved_pending = state.get("pending_by_lane", [])
        if len(saved_pending) != lanes_per_encoder:
            raise RuntimeError(
                f"routed encoder dataloader state has {len(saved_pending)} pending lanes, "
                f"expected {lanes_per_encoder}"
            )
        for queue, values in zip(pending_by_lane, saved_pending):
            queue.clear()
            queue.extend(values)
        loader_iter_holder[0] = None

    def next_encoder_batch():
        try:
            lane_batches, _pulls = _route_samples_to_lanes(
                get_loader_iter(),
                lanes_per_encoder=lanes_per_encoder,
                lane_offset=lane_offset,
                num_workers_per_lane=num_workers_per_lane,
                encoder_dp_rank=encoder_dp_rank,
                pending_by_lane=pending_by_lane,
                max_pulls_per_step=max_pulls_per_step,
                provenance_key=provenance_key,
            )
        except StopIteration:
            # One-shot per epoch on savable-loader exhaustion. Any partial
            # ``lane_batches`` accumulated before the exception is dropped —
            # those samples count against the worker's seed sequence and are
            # never delivered. Acceptable because webdataset is streamed as
            # a pseudo-infinite source; this branch is rarely hit in practice.
            loader_iter_holder[0] = iter(loader)
            lane_batches, _pulls = _route_samples_to_lanes(
                get_loader_iter(),
                lanes_per_encoder=lanes_per_encoder,
                lane_offset=lane_offset,
                num_workers_per_lane=num_workers_per_lane,
                encoder_dp_rank=encoder_dp_rank,
                pending_by_lane=pending_by_lane,
                max_pulls_per_step=max_pulls_per_step,
                provenance_key=provenance_key,
            )
        signatures = [EnergonIterator._batch_signature(batch) for batch in lane_batches]
        return _combine_encoder_batches(lane_batches), signatures

    return EnergonIterator(
        loader,
        tp_group=tp_group,
        source_rank=True,
        random_seed=args.seed,
        local_batch_fn=next_encoder_batch,
        alignment_role="encoder",
        llm_lanes=llm_lanes,
        state_name=(
            f"train_dataloader_encoder_dprank{encoder_dp_rank:03d}"
            f"_lanes{llm_lanes[0]:03d}-{llm_lanes[-1]:03d}.pt"
        ),
        extra_state_fn=save_route_state,
        restore_extra_state_fn=restore_route_state,
        trace_dir=getattr(args, "energon_sample_trace_dir", None),
    )


def _llm_lanes_for_encoder_rank(args, encoder_dp_rank: int) -> list[int]:
    """Return the contiguous LLM DP lanes owned by one encoder DP lane."""
    scale = args.llm_dp // args.encoder_dp
    start = encoder_dp_rank * scale
    return list(range(start, start + scale))


def _build_single_lane_iterator(args, tp_group, lane: int, role: str, random_seed: int):
    """Build a deterministic loader for one LLM data lane."""
    from examples.mimo.data.energon_multimodal_provider import build_multimodal_encoder
    from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset

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
    from megatron.energon.cache.no_cache import NoCachePool

    loader = get_savable_loader(
        dataset,
        cache_pool=NoCachePool(),
        watchdog_timeout_seconds=5 * 60,
        watchdog_initial_timeout_seconds=5 * 60,
    )
    state_role = "encoder" if role.startswith("encoder") else "llm"
    return EnergonIterator(
        loader,
        tp_group=tp_group,
        source_rank=True,
        random_seed=random_seed,
        alignment_role=state_role,
        llm_lanes=[lane],
        state_name=f"train_dataloader_{state_role}_lane{lane:03d}.pt",
        trace_dir=getattr(args, "energon_sample_trace_dir", None),
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
        combined["modality_inputs"] = _merge_modality_inputs(modality_values)

    return combined


# ---------------------------------------------------------------------------
# Schema-aware merge of ``modality_inputs`` across LLM lanes served by one
# encoder rank. The structure produced by the dataset is fixed:
#
#   modality_inputs = {
#       "<modality_type>": {                # e.g. "images"
#           "<encoder_name>": {              # e.g. "radio_encoder"
#               <packed_buffer_key>: Tensor of shape (1, T_lane, C),
#               "imgs_sizes":        Tensor of shape (N_images_lane, 2),
#               "packed_seq_params": PackedSeqParams describing the T axis,
#           }
#       }
#   }
#
# Each per-lane tensor has a known concat semantics; we encode them
# explicitly rather than inferring from runtime shape variation:
#
#   * packed image buffer: leading dim is always 1 (lane batch == MBS=1);
#     dim 1 is the variable token axis -> concat along dim 1.
#   * ``imgs_sizes``: dim 0 = per-lane image count -> concat along dim 0.
#   * ``packed_seq_params``: cu_seqlens need offset-shifting -> custom merge.
# ---------------------------------------------------------------------------


def _merge_modality_inputs(per_lane_modality_inputs):
    """Merge the ``modality_inputs`` field of N per-lane batches."""
    merged = {}
    modality_types = set().union(
        *(p.keys() for p in per_lane_modality_inputs if isinstance(p, dict))
    )
    for mod_type in sorted(modality_types):
        per_lane_mod = [p[mod_type] for p in per_lane_modality_inputs if mod_type in p]
        merged_per_encoder = {}
        encoder_names = set().union(
            *(p.keys() for p in per_lane_mod if isinstance(p, dict))
        )
        for enc_name in sorted(encoder_names):
            per_lane_enc = [p[enc_name] for p in per_lane_mod if enc_name in p]
            merged_per_encoder[enc_name] = _merge_encoder_inputs(per_lane_enc)
        merged[mod_type] = merged_per_encoder
    return merged


def _merge_encoder_inputs(per_lane_enc_inputs):
    """Merge per-lane encoder-input dicts using a key-explicit schema.

    Keys are categorized by name / value type:
      * ``packed_seq_params`` -> ``_concat_packed_seq_params``
      * ``imgs_sizes``        -> ``torch.cat(..., dim=0)``
      * any other ``Tensor``  -> packed image buffer ``(1, T, C)``,
                                 concat along dim 1
    Anything else triggers a loud error so a future schema change has to be
    handled here rather than guessed at by a heuristic.
    """
    merged = {}
    keys = set().union(*(p.keys() for p in per_lane_enc_inputs if isinstance(p, dict)))
    for key in sorted(keys):
        vals = [p[key] for p in per_lane_enc_inputs if key in p]
        if not vals:
            continue
        first = vals[0]
        if isinstance(first, PackedSeqParams):
            merged[key] = _concat_packed_seq_params(vals)
        elif key == "imgs_sizes":
            assert all(isinstance(v, torch.Tensor) for v in vals), (
                f"imgs_sizes must be tensors, got {[type(v).__name__ for v in vals]}"
            )
            merged[key] = torch.cat(vals, dim=0)
        elif isinstance(first, torch.Tensor):
            # Packed image buffer: leading dim is the lane batch (==1); the
            # variable token axis is dim 1.
            assert first.dim() >= 2 and first.shape[0] == 1, (
                f"unexpected packed-buffer shape for encoder key {key!r}: "
                f"{tuple(first.shape)} (expected leading dim 1)"
            )
            merged[key] = torch.cat(vals, dim=1)
        else:
            raise TypeError(
                f"unsupported encoder-input value for key {key!r}: "
                f"{type(first).__name__}; extend _merge_encoder_inputs"
            )
    return merged


def _concat_packed_seq_params(values: list) -> PackedSeqParams:
    """Merge per-lane PackedSeqParams into one set covering the merged flat buffer.

    The dim-0 image buffers from each lane are concatenated by the surrounding
    tensor merge; here we re-number cu_seqlens so they index into that merged
    buffer. Mirrors the offset-shift rule in
    ``megatron.energon.task_encoder.multimodal.encoder``.
    """
    first = values[0]
    for v in values[1:]:
        if v.qkv_format != first.qkv_format:
            raise ValueError(
                f"qkv_format mismatch across encoder lanes: "
                f"{first.qkv_format!r} vs {v.qkv_format!r}"
            )
        if v.local_cp_size != first.local_cp_size or v.cp_group is not first.cp_group:
            raise ValueError("CP fields mismatch across encoder lanes; refusing to merge")

    def _concat_cu(attr: str):
        per_lane = [getattr(v, attr) for v in values]
        if per_lane[0] is None:
            if not all(x is None for x in per_lane):
                raise ValueError(f"{attr} present on some lanes but not others")
            return None
        merged = [per_lane[0]]
        offset = int(per_lane[0][-1].item())
        for cu in per_lane[1:]:
            merged.append(cu[1:] + offset)
            offset += int(cu[-1].item())
        return torch.cat(merged)

    def _max_scalar(attr: str):
        per_lane = [getattr(v, attr) for v in values]
        if per_lane[0] is None:
            if not all(x is None for x in per_lane):
                raise ValueError(f"{attr} present on some lanes but not others")
            return None
        if torch.is_tensor(per_lane[0]):
            return torch.stack([x.reshape(()) for x in per_lane]).max()
        return max(per_lane)

    def _sum_or_none(attr: str):
        per_lane = [getattr(v, attr) for v in values]
        if all(x is None for x in per_lane):
            return None
        if any(x is None for x in per_lane):
            raise ValueError(f"{attr} present on some lanes but not others")
        return sum(per_lane)

    return PackedSeqParams(
        qkv_format=first.qkv_format,
        cu_seqlens_q=_concat_cu("cu_seqlens_q"),
        cu_seqlens_kv=_concat_cu("cu_seqlens_kv"),
        cu_seqlens_q_padded=_concat_cu("cu_seqlens_q_padded"),
        cu_seqlens_kv_padded=_concat_cu("cu_seqlens_kv_padded"),
        max_seqlen_q=_max_scalar("max_seqlen_q"),
        max_seqlen_kv=_max_scalar("max_seqlen_kv"),
        total_tokens=_sum_or_none("total_tokens"),
        local_cp_size=first.local_cp_size,
        cp_group=first.cp_group,
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
        self,
        dataloader,
        tp_group=None,
        source_rank: bool = True,
        random_seed: Optional[int] = None,
        local_batch_fn: Optional[Callable[[], dict]] = None,
        alignment_role: Optional[str] = None,
        llm_lanes: Optional[list[int]] = None,
        state_name: Optional[str] = None,
        extra_state_fn: Optional[Callable[[], dict]] = None,
        restore_extra_state_fn: Optional[Callable[[dict], None]] = None,
        trace_dir: Optional[str] = None,
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
        self._state_name = state_name
        self._extra_state_fn = extra_state_fn
        self._restore_extra_state_fn = restore_extra_state_fn
        self._trace_dir = trace_dir
        self._trace_step = 0
        self._trace_file = None
        self._python_random_state = None
        if random_seed is not None:
            rng = random.Random(random_seed)
            self._python_random_state = rng.getstate()

    def dataloader_state_name(self) -> Optional[str]:
        """Return the per-rank checkpoint filename for this iterator's state."""
        if not self._source_rank:
            return None
        return self._state_name

    def save_state(self):
        """Return a picklable snapshot of the Energon loader and wrapper state."""
        if not self._source_rank:
            return None
        if self._dataloader is None or not hasattr(self._dataloader, "save_state_rank"):
            raise RuntimeError(f"cannot save dataloader state for {type(self._dataloader).__name__}")

        state = {
            "version": 1,
            "role": self._alignment_role,
            "llm_lanes": list(self._llm_lanes),
            "dataloader_state_dict": self._dataloader.save_state_rank(),
            "python_random_state": self._python_random_state,
            "prefetched": self._prefetched,
            "prefetched_component_signatures": self._prefetched_component_signatures,
            "trace_step": self._trace_step,
        }
        if self._extra_state_fn is not None:
            state["extra_state"] = self._extra_state_fn()
        return state

    def restore_state(self, state) -> None:
        """Restore a snapshot previously returned by :meth:`save_state`."""
        if not self._source_rank:
            return
        if state.get("version") != 1:
            raise RuntimeError(f"unsupported dataloader state version: {state.get('version')}")
        if list(state.get("llm_lanes", [])) != list(self._llm_lanes):
            raise RuntimeError(
                f"dataloader state lanes {state.get('llm_lanes')} do not match "
                f"current lanes {self._llm_lanes}"
            )
        if self._dataloader is None or not hasattr(self._dataloader, "restore_state_rank"):
            raise RuntimeError(f"cannot restore dataloader state for {type(self._dataloader).__name__}")

        self._dataloader.restore_state_rank(state["dataloader_state_dict"])
        self._iterator = None
        self._python_random_state = state.get("python_random_state")
        self._prefetched = state.get("prefetched")
        self._prefetched_component_signatures = state.get("prefetched_component_signatures")
        self._local_component_signatures = None
        self._trace_step = state.get("trace_step", self._trace_step)
        if self._restore_extra_state_fn is not None:
            self._restore_extra_state_fn(state.get("extra_state", {}))

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
        self._trace_sample(component_signatures)
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

    def _trace_sample(self, component_signatures) -> None:
        """Write optional per-source-rank sample signatures for replay validation."""
        if not self._trace_dir or not self._source_rank or component_signatures is None:
            return
        if self._trace_file is None:
            os.makedirs(self._trace_dir, exist_ok=True)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            name = self._state_name or f"{self._alignment_role}_rank{rank:05d}.pt"
            path = os.path.join(self._trace_dir, f"{name}.rank{rank:05d}.jsonl")
            self._trace_file = open(path, "a", encoding="utf-8", buffering=1)
        record = {
            "step": self._trace_step,
            "role": self._alignment_role,
            "llm_lanes": self._llm_lanes,
            "signatures": [list(signature) for signature in component_signatures],
        }
        self._trace_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._trace_step += 1

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
