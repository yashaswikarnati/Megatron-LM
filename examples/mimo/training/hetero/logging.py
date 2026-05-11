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
            sys.stdout.write(f"{log_string}\n")
            sys.stdout.flush()
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
