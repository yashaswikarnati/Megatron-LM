# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed setup helpers for heterogeneous MIMO examples."""

from __future__ import annotations

import datetime
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
        # 1-hour collective timeout: lustre Bridge-DCP reads on encoder ranks can
        # leave LLM ranks idle for several minutes; default 600 s is too short.
        # device_id is explicit: pytorch's auto-guess from global rank can cause
        # hangs in heterogeneous topologies (encoder/LLM offset != 0).
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=1),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    assert_megatron_parallel_state_uninitialized()
    try:
        parallel_state.get_global_memory_buffer()
    except AssertionError:
        parallel_state._set_global_memory_buffer()
    dist.barrier()


def assert_megatron_parallel_state_uninitialized() -> None:
    """Ensure this standalone hetero path owns Megatron process-group setup."""
    initialized_groups = []
    if parallel_state.is_initialized():
        initialized_groups.append("data_parallel")
    if parallel_state.get_model_parallel_group(check_initialized=False) is not None:
        initialized_groups.append("model_parallel")
    if parallel_state.get_tensor_model_parallel_group(check_initialized=False) is not None:
        initialized_groups.append("tensor_model_parallel")
    if parallel_state.get_pipeline_model_parallel_group(check_initialized=False) is not None:
        initialized_groups.append("pipeline_model_parallel")
    if parallel_state.get_context_parallel_group(check_initialized=False) is not None:
        initialized_groups.append("context_parallel")
    if parallel_state.get_embedding_group(check_initialized=False) is not None:
        initialized_groups.append("embedding")
    if parallel_state.get_position_embedding_group(check_initialized=False) is not None:
        initialized_groups.append("position_embedding")

    if initialized_groups:
        raise RuntimeError(
            "train_hetero.py expects Megatron parallel_state process groups to be "
            f"uninitialized, but found: {', '.join(initialized_groups)}"
        )


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
