# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unified debug profiler for MIMO training.

Provides two entry points for train.py:

    register_profilers(model)                # in model_provider()
    step_profilers(model, output_tensor)     # in forward_step()

Gated on a single environment variable:

    MIMO_DEBUG=1              Enable profiling (default: disabled)
    MIMO_DEBUG_STEPS=N        Steps to capture (default: 3)
    MIMO_DEBUG_OUTPUT=<path>  JSON output path (default: logs/debug/grad_stats_rank{rank}.json)

Both calls are no-ops when MIMO_DEBUG is unset.
"""

import atexit
import os

import torch


# ---------------------------------------------------------------------------
# Public API — called from train.py
# ---------------------------------------------------------------------------

def register_profilers(model):
    """Register debug profiler hooks on *model*. No-op when MIMO_DEBUG is unset."""
    if not os.environ.get("MIMO_DEBUG", ""):
        return

    from scripts.debug.grad_norm_profiler import GradNormProfiler

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    max_steps = int(os.environ.get("MIMO_DEBUG_STEPS", "3"))
    output = os.environ.get(
        "MIMO_DEBUG_OUTPUT",
        f"logs/debug/grad_stats_rank{rank}.json",
    )

    profiler = GradNormProfiler(model, output_path=output, max_steps=max_steps, rank=rank)
    profiler.register_hooks()
    model._debug_profiler = profiler
    atexit.register(profiler.dump)


def step_profilers(model, output_tensor):
    """Advance the debug profiler by one step. No-op when MIMO_DEBUG is unset."""
    unwrapped = _unwrap_model(model)
    if not hasattr(unwrapped, '_debug_profiler'):
        return

    profiler = unwrapped._debug_profiler
    if profiler.step > 0:
        profiler.step_end()
    profiler.step_start()


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _unwrap_model(model):
    """Peel DDP / FSDP wrappers to get the underlying user model."""
    m = model
    while hasattr(m, 'module'):
        m = m.module
    return m
