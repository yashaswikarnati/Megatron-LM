# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Async encoder DDP param + optimizer state offloader for colocated MIMO training.

Split-lifecycle design: params and optimizer states have independent offload/reload
cycles to maximize overlap with compute.

**Param lifecycle** (tied to encoder fwd/bwd):
  - ``offload_params()`` after encoder forward (D2H overlaps with LLM pipeline)
  - ``reload_params()`` before encoder backward (H2D overlaps with cooldown)
  - ``reload_params_sync()`` before encoder backward uses params

**Optimizer state lifecycle** (tied to optimizer.step):
  - ``offload_opt_states()`` after optimizer.step (D2H overlaps with next encoder fwd)
  - ``reload_opt_states()`` before encoder backward (H2D overlaps with cooldown)
  - Opt state sync happens inside ``reload_params_sync()``

Timeline::

    optimizer.step()
    offload_opt_states()         # async D2H opt states → overlaps with encoder fwd
    encoder_forward()            # compute overlaps with opt state D2H
    offload_params()             # async D2H params only (~1.5 GB, not 2.1 GB)
    LLM pipeline                 # param D2H finishes quickly, less allocator stall
    pre_cooldown → reload()      # async H2D params + opt states
    reload_sync()                # wait for all H2D
    encoder_backward()
    optimizer.step()             # opt states already on GPU
    offload_opt_states()         # ... next iteration
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class EncoderDDPOffloader:
    """Offloads encoder DDP params + optimizer states to CPU between phases.

    Split-lifecycle: params and optimizer states offload/reload independently
    to maximize overlap with compute.

    Args:
        encoder_ddp: The encoder's DistributedDataParallel wrapper.
    """

    def __init__(self, encoder_ddp) -> None:
        self._ddp = encoder_ddp
        self._optimizer = None
        self._params_offloaded = False
        self._opt_offloaded = False

        self._copy_stream = torch.cuda.Stream()

        # Grad buffer sizes saved before freeing.
        self._grad_data_sizes: Dict[int, int] = {}

        # Pinned CPU mirrors for optimizer GPU tensors (allocated lazily).
        self._opt_pinned_buffers: Optional[List[torch.Tensor]] = None
        # (gpu_tensor, cpu_tensor, original_storage_size) for reload.
        self._opt_offload_info: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        # Iteration counter for logging (first 3 iterations + every 10th).
        self._iter = 0

    def set_optimizer(self, encoder_optimizer) -> None:
        """Bind the encoder optimizer for state offloading."""
        self._optimizer = encoder_optimizer

    def _should_log(self) -> bool:
        return self._iter < 3 or self._iter % 10 == 0

    # ------------------------------------------------------------------
    # Param offload / reload (after encoder fwd / before encoder bwd)
    # ------------------------------------------------------------------

    def offload_params(self) -> None:
        """Async D2H offload of DDP params + free grads.  Returns immediately.

        Only offloads params (not optimizer states).  The D2H is ~1.5 GB for a
        3B encoder at TP4, completing in ~50 ms on PCIe — much less allocator
        stall than offloading params + opt states together.
        """
        if self._params_offloaded:
            return

        if self._should_log():
            logger.info(
                "[OFFLOAD_PARAMS iter=%d] mem=%.2f GB — starting async D2H",
                self._iter, torch.cuda.memory_allocated() / 1e9,
            )

        self._copy_stream.wait_stream(torch.cuda.current_stream())

        all_buffers = self._all_buffers()

        # Param D2H on copy_stream.
        with torch.cuda.stream(self._copy_stream):
            for buffer in all_buffers:
                buffer.offload_to_cpu(move_params=True, move_grads=False)

        # Free grad buffer instantly (stale, no copy needed).
        for i, buffer in enumerate(all_buffers):
            if buffer.grad_data is not None and buffer.grad_data.storage().size() > 0:
                self._grad_data_sizes[i] = buffer.grad_data.storage().size()
                buffer.grad_data.storage().resize_(0)

        self._params_offloaded = True

    # ------------------------------------------------------------------
    # Optimizer state offload / reload (after opt.step / before opt.step)
    # ------------------------------------------------------------------

    def offload_opt_states(self) -> None:
        """Async D2H offload of optimizer states.  Returns immediately.

        Call after ``optimizer.step()`` — the D2H overlaps with the next
        iteration's encoder forward, which doesn't need optimizer states.
        """
        if self._optimizer is None or self._opt_offloaded:
            return

        gpu_tensors = self._collect_optimizer_gpu_tensors()
        if not gpu_tensors:
            return

        if self._should_log():
            logger.info(
                "[OFFLOAD_OPT iter=%d] mem=%.2f GB — starting async D2H (%d tensors)",
                self._iter, torch.cuda.memory_allocated() / 1e9, len(gpu_tensors),
            )

        pinned = self._ensure_pinned_buffers(gpu_tensors)

        self._copy_stream.wait_stream(torch.cuda.current_stream())

        # Async D2H on copy_stream.
        with torch.cuda.stream(self._copy_stream):
            for gpu_t, cpu_t in zip(gpu_tensors, pinned):
                cpu_t.copy_(gpu_t, non_blocking=True)

        # Free GPU storage (allocator tracks copy_stream dependency).
        self._opt_offload_info.clear()
        for gpu_t, cpu_t in zip(gpu_tensors, pinned):
            orig_size = gpu_t.storage().size()
            self._opt_offload_info.append((gpu_t, cpu_t, orig_size))
            gpu_t.storage().resize_(0)

        self._opt_offloaded = True

    # ------------------------------------------------------------------
    # Reload (all at once, before encoder backward)
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Async H2D reload of params + opt states + restore grads.  Returns immediately.

        Call from ``pre_cooldown_func`` — the H2D overlaps with cooldown BWDs.
        """
        if self._should_log():
            logger.info(
                "[RELOAD iter=%d] mem=%.2f GB — starting async H2D",
                self._iter, torch.cuda.memory_allocated() / 1e9,
            )

        all_buffers = self._all_buffers()

        # Restore grad buffer (default stream, synchronous).
        if self._params_offloaded:
            for i, buffer in enumerate(all_buffers):
                if i in self._grad_data_sizes and buffer.grad_data is not None:
                    buffer.grad_data.storage().resize_(self._grad_data_sizes[i])
                    buffer.grad_data.zero_()
            self._grad_data_sizes.clear()

        self._copy_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._copy_stream):
            # Param H2D.
            if self._params_offloaded:
                for buffer in all_buffers:
                    buffer.reload_from_cpu(move_params=True, move_grads=False)

            # Opt state H2D.
            if self._opt_offloaded:
                for gpu_t, cpu_t, orig_size in self._opt_offload_info:
                    gpu_t.storage().resize_(orig_size)
                    gpu_t.copy_(cpu_t, non_blocking=True)

        if self._opt_offloaded:
            self._opt_offload_info.clear()
            self._opt_offloaded = False
        self._params_offloaded = False

    def reload_sync(self) -> None:
        """Wait for async H2D to complete.  Call before encoder backward."""
        log = self._should_log()
        if log:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        torch.cuda.current_stream().wait_stream(self._copy_stream)

        if log:
            t1.record()
            t1.synchronize()
            logger.info(
                "[RELOAD_SYNC iter=%d] wait=%.2f ms (0 = fully overlapped)",
                self._iter, t0.elapsed_time(t1),
            )

        self._iter += 1

    # ------------------------------------------------------------------
    # Legacy convenience (offload everything in one call)
    # ------------------------------------------------------------------

    def offload(self) -> None:
        """Offload params + opt states in one call (legacy API)."""
        self.offload_params()
        self.offload_opt_states()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_buffers(self):
        """Return all DDP buffers (regular + expert parallel)."""
        buffers = list(self._ddp.buffers)
        if hasattr(self._ddp, 'expert_parallel_buffers'):
            buffers.extend(self._ddp.expert_parallel_buffers)
        return buffers

    @staticmethod
    def _get_dist_optimizers(optimizer):
        """Extract ``DistributedOptimizer`` instances from *optimizer*."""
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.optimizer.optimizer import ChainedOptimizer

        if isinstance(optimizer, ChainedOptimizer):
            return [
                opt
                for opt in optimizer.chained_optimizers
                if isinstance(opt, DistributedOptimizer)
            ]
        elif isinstance(optimizer, DistributedOptimizer):
            return [optimizer]
        else:
            logger.warning(
                "EncoderDDPOffloader: optimizer type %s is not a DistributedOptimizer "
                "or ChainedOptimizer — optimizer state offload will be skipped.",
                type(optimizer).__name__,
            )
            return []

    def _collect_optimizer_gpu_tensors(self) -> List[torch.Tensor]:
        """Collect GPU-resident optimizer tensors to offload."""
        gpu_tensors: List[torch.Tensor] = []
        for dist_opt in self._get_dist_optimizers(self._optimizer):
            for group in dist_opt.shard_fp32_from_float16_groups:
                for param in group:
                    if param.is_cuda and param.storage().size() > 0:
                        gpu_tensors.append(param)
            for state_dict in dist_opt.optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state_dict:
                        t = state_dict[key]
                        if t.is_cuda and t.storage().size() > 0:
                            gpu_tensors.append(t)
        return gpu_tensors

    def _ensure_pinned_buffers(self, gpu_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Allocate (or reuse) pinned CPU buffers matching *gpu_tensors*."""
        if self._opt_pinned_buffers is not None and len(self._opt_pinned_buffers) == len(
            gpu_tensors
        ):
            return self._opt_pinned_buffers
        self._opt_pinned_buffers = [
            torch.empty(t.shape, dtype=t.dtype, device="cpu").pin_memory() for t in gpu_tensors
        ]
        return self._opt_pinned_buffers
