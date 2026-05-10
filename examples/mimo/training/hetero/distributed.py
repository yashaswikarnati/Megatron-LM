# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed setup helpers for heterogeneous MIMO examples."""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from megatron.core import parallel_state


def initialize_distributed() -> None:
    """Initialize torch.distributed for torchrun."""
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    try:
        parallel_state.get_global_memory_buffer()
    except AssertionError:
        parallel_state._set_global_memory_buffer()
    dist.barrier()


def print_rank_0(message: str) -> None:
    """Print only on global rank zero."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()


def shutdown_distributed() -> None:
    """Tear down process-global Megatron and torch.distributed state."""
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    parallel_state.destroy_global_memory_buffer()
    if dist.is_initialized():
        dist.destroy_process_group()
