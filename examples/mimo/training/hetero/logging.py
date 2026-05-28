# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-shaped interval logging for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.optimizer import get_global_batch_size
from examples.mimo.training.hetero.step import TrainStepResult
from examples.mimo.training.hetero.topology import HeteroTopology
from examples.mimo.utils.hetero import is_process_group_member
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.transformer.moe.moe_utils import track_moe_metrics

_MB = 1024.0 * 1024.0


def _vm_rss_mb() -> float:
    """Read process RSS from /proc/self/status; cheap and dependency-free."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _vm_peak_mb() -> float:
    """Read process VmPeak (peak virtual size) from /proc/self/status."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


@dataclass
class HeteroTrainingLogger:
    """Accumulate and print interval training metrics."""

    args: argparse.Namespace
    topology: HeteroTopology
    consumed_train_samples: int = 0
    advanced_iterations: int = 0
    skipped_iterations: int = 0
    nan_iterations: int = 0
    loss_total: float = 0.0
    loss_count: int = 0
    interval_start: float = field(default_factory=time.time)
    _tb_writer: Optional[object] = field(default=None, init=False, repr=False)
    _moe_total_loss_dict: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Only the language logging rank owns the writer; other ranks no-op.
        tb_dir = getattr(self.args, "tensorboard_dir", None)
        if tb_dir and is_language_log_rank(self.topology):
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=tb_dir)

    def record_step(self, result: TrainStepResult) -> Optional[float]:
        """Update interval state from one train step and return this iteration's loss."""
        self.consumed_train_samples += get_global_batch_size(self.args)
        loss_value = reduce_language_loss(result.losses, self.topology)

        if result.skipped_iter:
            self.skipped_iterations += result.skipped_iter
            if loss_value is not None and not math.isfinite(loss_value):
                self.nan_iterations += 1
            return loss_value

        self.advanced_iterations += 1
        if loss_value is not None:
            if math.isfinite(loss_value):
                self.loss_total += loss_value
                self.loss_count += 1
            else:
                self.nan_iterations += 1
        return loss_value

    def maybe_log(self, iteration: int, optimizer, result: TrainStepResult) -> None:
        """Print Megatron-like interval metrics on the language logging rank."""
        if iteration % self.args.log_interval != 0:
            return

        elapsed = time.time() - self.interval_start
        interval_iters = max(1, self.advanced_iterations + self.skipped_iterations)
        elapsed_ms = (elapsed / interval_iters) * 1000.0
        loss_value = self.loss_total / self.loss_count if self.loss_count else None
        learning_rate = get_canonical_lr_for_logging(optimizer.param_groups)
        loss_scale = optimizer.get_loss_scale().item()

        cuda_alloc_mb = torch.cuda.memory_allocated() / _MB
        cuda_peak_mb = torch.cuda.max_memory_allocated() / _MB
        rss_mb = _vm_rss_mb()
        host_peak_mb = _vm_peak_mb()
        torch.cuda.reset_peak_memory_stats()

        if is_language_log_rank(self.topology):
            log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}]"
            log_string += " iteration {:8d}/{:8d} |".format(iteration, self.args.train_iters)
            log_string += " consumed samples: {:12d} |".format(self.consumed_train_samples)
            log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_ms)
            if learning_rate is not None:
                log_string += f" learning rate: {learning_rate:.6E} |"
            log_string += f" global batch size: {get_global_batch_size(self.args):5d} |"
            if loss_value is not None:
                log_string += f" lm loss: {loss_value:.6E} |"
            log_string += f" loss scale: {loss_scale:.1f} |"
            if result.grad_norm is not None:
                log_string += f" grad norm: {result.grad_norm:.3f} |"
            if result.num_zeros_in_grad is not None:
                log_string += f" num zeros: {int(result.num_zeros_in_grad)} |"
            log_string += " number of skipped iterations: {:3d} |".format(self.skipped_iterations)
            log_string += " number of nan iterations: {:3d} |".format(self.nan_iterations)
            log_string += (
                f" cuda alloc: {cuda_alloc_mb:.0f}MB peak: {cuda_peak_mb:.0f}MB |"
                f" host rss: {rss_mb:.0f}MB peak: {host_peak_mb:.0f}MB |"
            )
            sys.stdout.write(f"{log_string}\n")
            sys.stdout.flush()
        num_moe_experts = getattr(self.args, "num_moe_experts", None)
        if num_moe_experts and is_process_group_member(
            getattr(self.topology.language_pg, "dp_cp", None)
        ):
            hybrid_pat = getattr(self.args, "hybrid_layer_pattern", None)
            if hybrid_pat:
                num_moe_layers = hybrid_pat.count("E")
            else:
                num_moe_layers = getattr(self.args, "num_layers", 0)
            track_moe_metrics(
                loss_scale=1.0 / max(1, getattr(self.args, "num_microbatches", 1)),
                iteration=iteration,
                writer=self._tb_writer,
                wandb_writer=None,
                total_loss_dict=self._moe_total_loss_dict,
                per_layer_logging=False,
                force_initialize=True,
                track_names=["seq_load_balancing_loss"],
                num_layers=num_moe_layers,
                moe_layer_freq=None,
                mtp_num_layers=getattr(self.args, "mtp_num_layers", None),
                pg_collection=self.topology.language_pg,
            )

        if self._tb_writer is not None:
            batch_size = get_global_batch_size(self.args)
            samples = self.consumed_train_samples
            if loss_value is not None:
                self._tb_writer.add_scalar("lm loss", loss_value, iteration)
                self._tb_writer.add_scalar("lm loss vs samples", loss_value, samples)
            if learning_rate is not None:
                self._tb_writer.add_scalar("learning-rate", learning_rate, iteration)
                self._tb_writer.add_scalar(
                    "learning-rate vs samples", learning_rate, samples
                )
            self._tb_writer.add_scalar("batch-size", batch_size, iteration)
            self._tb_writer.add_scalar("batch-size vs samples", batch_size, samples)
            self._tb_writer.add_scalar("loss-scale", loss_scale, iteration)
            if result.grad_norm is not None:
                self._tb_writer.add_scalar("grad-norm", result.grad_norm, iteration)
                self._tb_writer.add_scalar(
                    "grad-norm vs samples", result.grad_norm, samples
                )
            if result.num_zeros_in_grad is not None:
                self._tb_writer.add_scalar(
                    "num-zeros", result.num_zeros_in_grad, iteration
                )
            self._tb_writer.add_scalar(
                "iteration-time-ms", elapsed_ms, iteration
            )
            self._tb_writer.add_scalar("mem/cuda-allocated-mb", cuda_alloc_mb, iteration)
            self._tb_writer.add_scalar("mem/cuda-peak-mb", cuda_peak_mb, iteration)
            self._tb_writer.add_scalar("mem/host-rss-mb", rss_mb, iteration)
            self._tb_writer.add_scalar("mem/host-peak-mb", host_peak_mb, iteration)
            self._tb_writer.flush()
        self.reset_interval()

    def reset_interval(self) -> None:
        """Reset interval accumulators after a log event."""
        self.advanced_iterations = 0
        self.skipped_iterations = 0
        self.nan_iterations = 0
        self.loss_total = 0.0
        self.loss_count = 0
        self.interval_start = time.time()


@torch.no_grad()
def reduce_language_loss(losses: list[dict], topology: HeteroTopology) -> Optional[float]:
    """Reduce raw loss/token vectors over the language DP/CP logging group."""
    language_pg = topology.language_pg
    loss_acc = torch.zeros(2, dtype=torch.float32, device="cuda")
    is_log_stage = (
        is_process_group_member(getattr(language_pg, "dp_cp", None))
        and (is_pp_last_stage(language_pg.pp))
        and language_pg.tp.rank() == 0
    )
    if not is_log_stage:
        return None

    if losses:
        for loss_dict in losses:
            loss = loss_dict.get("lm loss")
            if isinstance(loss, torch.Tensor):
                loss_acc += loss.detach().to(device="cuda", dtype=torch.float32).view(2)
            elif loss is not None:
                loss_acc += torch.tensor(loss, dtype=torch.float32, device="cuda").view(2)

    dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM, group=language_pg.dp_cp)
    return loss_acc[0].item() / loss_acc[1].item() if loss_acc[1].item() else None


def is_language_log_rank(topology: HeteroTopology) -> bool:
    """Return whether this rank should print language-side training metrics."""
    language_pg = topology.language_pg
    if not (
        is_process_group_member(getattr(language_pg, "dp_cp", None))
        and is_pp_last_stage(language_pg.pp)
        and language_pg.tp.rank() == 0
    ):
        return False
    return dist.get_rank() == dist.get_global_rank(language_pg.dp_cp, 0)
