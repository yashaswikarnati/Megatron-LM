# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Timeline tracing configuration for standalone heterogeneous MIMO training."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch.distributed as dist

from examples.mimo.training.hetero.topology import HeteroTopology
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.timeline import configure_pipeline_timeline


def configure_hetero_timeline(args: argparse.Namespace, topology: HeteroTopology) -> Optional[str]:
    """Configure rank-local pipeline timeline tracing and return a rank-0 summary."""
    enabled = args.timeline_profile or env_flag_enabled("MIMO_TIMELINE")
    if not enabled:
        configure_pipeline_timeline(
            enabled=False,
            output_dir=args.timeline_dir or "",
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            role="",
        )
        return None

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    scope = os.environ.get("MIMO_TIMELINE_RANKS", args.timeline_ranks)
    dp_replica = int(os.environ.get("MIMO_TIMELINE_DP_REPLICA", args.timeline_dp_replica))
    output_dir = args.timeline_dir or os.environ.get("MIMO_TIMELINE_DIR", "mimo_timeline")
    iteration_start = _optional_env_int("MIMO_TIMELINE_ITER_START", args.timeline_iter_start)
    iteration_end = _optional_env_int("MIMO_TIMELINE_ITER_END", args.timeline_iter_end)
    if iteration_start is not None and iteration_start < 1:
        raise ValueError("timeline iteration start must be >= 1")
    if iteration_end is not None and iteration_end < 1:
        raise ValueError("timeline iteration end must be >= 1")
    if (
        iteration_start is not None
        and iteration_end is not None
        and iteration_end < iteration_start
    ):
        raise ValueError("timeline iteration end must be >= timeline iteration start")
    selected_ranks = select_timeline_ranks(scope, dp_replica, topology, world_size)
    role, coords = rank_role_and_coords(rank, topology)

    configure_pipeline_timeline(
        enabled=rank in selected_ranks,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        role=role,
        metadata={
            "rank_scope": scope,
            "timeline_dp_replica": dp_replica,
            **coords,
        },
        cuda_events=args.timeline_cuda_events or env_flag_enabled("MIMO_TIMELINE_CUDA_EVENTS"),
        nvtx=args.timeline_nvtx or env_flag_enabled("MIMO_TIMELINE_NVTX"),
        iteration_start=iteration_start,
        iteration_end=iteration_end,
    )

    if rank != 0:
        return None
    return (
        "Pipeline timeline enabled: "
        f"dir={output_dir}, scope={scope}, selected_ranks={len(selected_ranks)}, "
        f"iter_start={iteration_start}, iter_end={iteration_end}"
    )


def select_timeline_ranks(
    scope: str, dp_replica: int, topology: HeteroTopology, world_size: int
) -> set[int]:
    """Select ranks to trace."""
    scope = scope.strip().lower()
    if scope == "all":
        return set(range(world_size))
    if scope == "dp-replica":
        ranks = ranks_for_dp_replica(topology.encoder_grid, dp_replica)
        ranks.update(ranks_for_dp_replica(topology.llm_grid, dp_replica))
        return ranks
    return {int(item) for item in scope.split(",") if item.strip()}


def ranks_for_dp_replica(grid: HyperCommGrid, dp_replica: int) -> set[int]:
    """Return all ranks that belong to one dense DP replica of a grid."""
    ranks = set()
    for rank in range(grid.rank_offset, grid.rank_offset + grid.size):
        coords = grid_coords(grid, rank)
        if coords.get("dp") == dp_replica:
            ranks.add(rank)
    return ranks


def rank_role_and_coords(
    rank: int, topology: HeteroTopology
) -> tuple[str, dict[str, int | str]]:
    """Return role and dense-grid coordinates for timeline metadata."""
    for role, grid in (("encoder", topology.encoder_grid), ("llm", topology.llm_grid)):
        if grid.rank_offset <= rank < grid.rank_offset + grid.size:
            coords = grid_coords(grid, rank)
            role_coords = {f"{role}_{key}": value for key, value in coords.items()}
            return role, {"module": role, **role_coords}
    return "unknown", {"module": "unknown"}


def grid_coords(grid: HyperCommGrid, rank: int) -> dict[str, int]:
    """Decode a global rank into dense HyperCommGrid coordinates."""
    local_rank = rank - grid.rank_offset
    coords = {}
    for dim_name, dim_size in zip(grid.dim_names, grid.shape):
        coords[dim_name] = local_rank % dim_size
        local_rank //= dim_size
    return coords


def env_flag_enabled(name: str) -> bool:
    """Return whether an environment flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
