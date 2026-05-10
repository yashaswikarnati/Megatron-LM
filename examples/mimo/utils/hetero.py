# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for heterogeneous MIMO examples."""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid

MOCK_MODEL_PRESET = "mock"
NEMOTRON_20L_MODEL_PRESET = "nemotron-moe-vlm-20l"
NEMOTRON_20L_HYBRID_PATTERN = "MEMEM*EMEMEM*EMEMEM*"
NEMOTRON_20L_IMAGE_SEQ_PER_TILE = 256
NEMOTRON_20L_MAX_NUM_TILES = 12
NEMOTRON_20L_DEFAULT_STAGE = "stage2"
MOCK_VISION_ENCODER_KEY = "clip_encoder"
NEMOTRON_VISION_ENCODER_KEY = "radio_encoder"


def is_nemotron_20l(args) -> bool:
    """Return whether the run should use the Nemotron6-MoE VLM 20L architecture."""
    return args.model_preset == NEMOTRON_20L_MODEL_PRESET


def debug_rank(message: str) -> None:
    """Emit per-rank startup checkpoints when MIMO_HETERO_DEBUG is set."""
    if os.environ.get("MIMO_HETERO_DEBUG"):
        rank = dist.get_rank() if dist.is_initialized() else 0
        sys.stderr.write(f"[rank {rank}] {message}\n")
        sys.stderr.flush()


def is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Return whether pg is a real process group for this rank."""
    group_member = getattr(dist, "GroupMember", None)
    non_member = getattr(group_member, "NON_GROUP_MEMBER", None)
    return pg is not None and pg != non_member


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return a base-layout dimension size."""
    return grid.shape[grid.dim_names.index(dim)]


def get_group_size_or(pg: Optional[dist.ProcessGroup], fallback: int) -> int:
    """Return pg size on member ranks, otherwise fallback."""
    return pg.size() if is_process_group_member(pg) else fallback


def get_group_rank_or(pg: Optional[dist.ProcessGroup], fallback: int = 0) -> int:
    """Return rank inside pg on member ranks, otherwise fallback."""
    return dist.get_rank(pg) if is_process_group_member(pg) else fallback
