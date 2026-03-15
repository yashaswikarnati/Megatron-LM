# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Activation and gradient norm profiler for MIMO training.

Provides two entry points for train.py:

    register_profilers(model)                # in model_provider()
    step_profilers(model, output_tensor)     # in forward_step()

Gated on a single environment variable:

    MIMO_DEBUG=1              Enable profiling (default: disabled)
    MIMO_DEBUG_STEPS=N        Number of forward passes to capture (default: 3)
    MIMO_DEBUG_OUTPUT=<path>  JSON output path (default: logs/debug/stats_rank{rank}.json)

Both calls are no-ops when MIMO_DEBUG is unset.
"""

import atexit
import json
import os
from datetime import datetime

import torch


# ---------------------------------------------------------------------------
# Public API — called from train.py
# ---------------------------------------------------------------------------

def register_profilers(model):
    """Register debug profiler hooks on *model*. No-op when MIMO_DEBUG is unset."""
    if not os.environ.get("MIMO_DEBUG", ""):
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    max_steps = int(os.environ.get("MIMO_DEBUG_STEPS", "3"))
    output = os.environ.get(
        "MIMO_DEBUG_OUTPUT",
        f"logs/debug/stats_rank{rank}.json",
    )

    profiler = ActGradProfiler(model, output_path=output, max_steps=max_steps, rank=rank)
    profiler.register_hooks()
    model._debug_profiler = profiler
    atexit.register(profiler.dump)


def step_profilers(model, output_tensor):
    """Snapshot hook stats from this forward pass. No-op when MIMO_DEBUG is unset."""
    m = model
    while hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_debug_profiler'):
        m._debug_profiler.snapshot()


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class ActGradProfiler:
    """Per-layer activation and gradient norm profiler.

    Hooks capture:
        fwd/<module_name>  — forward activation output
        bwd/<module_name>  — backward grad_output (autograd gradient flow)

    Call ``snapshot()`` once per forward pass to save and reset.
    """

    def __init__(self, model, output_path=None, max_steps=5, rank=0):
        self.model = model
        self.output_path = output_path or "/tmp/act_grad_stats.json"
        self.max_steps = max_steps
        self.rank = rank
        self.step = 0
        self._hooks = []
        self._stats = {}
        self._records = []

    def register_hooks(self):
        """Register forward and backward hooks on parameter-owning modules."""
        count = 0
        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            if not list(module.parameters(recurse=False)):
                continue

            def _fwd(mod, inp, out, name=name):
                if self.step < self.max_steps:
                    self._record(f"fwd/{name}", out)

            def _bwd(mod, grad_input, grad_output, name=name):
                if self.step < self.max_steps:
                    if grad_output and grad_output[0] is not None:
                        self._record(f"bwd/{name}", grad_output[0])

            self._hooks.append(module.register_forward_hook(_fwd))
            self._hooks.append(module.register_full_backward_hook(_bwd))
            count += 1

        if self.rank == 0:
            print(f"[ActGradProfiler] Hooks on {count} modules, max {self.max_steps} steps", flush=True)

    def snapshot(self):
        """Save current hook stats and reset. Call once per forward pass."""
        if self.step >= self.max_steps or not self._stats:
            self._stats = {}
            return

        self._records.append({"step": self.step, "rank": self.rank, "stats": self._stats})

        if self.rank == 0:
            n_fwd = sum(1 for k in self._stats if k.startswith("fwd/"))
            n_bwd = sum(1 for k in self._stats if k.startswith("bwd/"))
            print(f"[ActGradProfiler] Step {self.step}: fwd={n_fwd} bwd={n_bwd}", flush=True)

        self._stats = {}
        self.step += 1

        if self.step >= self.max_steps:
            self.dump()

    def dump(self):
        """Write collected stats to JSON (rank 0 only)."""
        if self.rank != 0 or not self._records:
            return
        data = {
            "timestamp": datetime.now().isoformat(),
            "max_steps": self.max_steps,
            "total_steps_captured": len(self._records),
            "steps": self._records,
        }
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[ActGradProfiler] Dumped to {self.output_path}", flush=True)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _record(self, key, tensor):
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        with torch.no_grad():
            t = tensor.float()
            self._stats[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "norm": t.norm().item(),
                "mean": t.mean().item(),
                "absmax": t.abs().max().item(),
                "has_nan": bool(t.isnan().any().item()),
                "has_inf": bool(t.isinf().any().item()),
            }
