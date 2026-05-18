# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Forward/backward step behavior for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from examples.mimo.training.hetero.grad_sync import (
    mark_modality_participation,
    reset_modality_participation,
    zero_active_grad_buffers,
)
from examples.mimo.training.hetero.optimizer import get_global_batch_size
from examples.mimo.training.hetero.topology import HeteroTopology
from examples.mimo.utils.hetero import debug_rank
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.timeline import timeline_event


@dataclass
class TrainStepResult:
    """Megatron-style result returned by one hetero training step."""

    losses: list[dict[str, Any]]
    skipped_iter: int
    update_successful: bool
    grad_norm: Optional[float]
    num_zeros_in_grad: Optional[int]


def loss_func(output_tensor: torch.Tensor, *, loss_mask: torch.Tensor):
    """Return terminal language-model loss sum, local token count, and logging tensors."""
    if output_tensor is None:
        raise RuntimeError("terminal language stage returned no loss tensor")
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(
            "loss_func expects the terminal language stage to return a tensor, "
            f"got {type(output_tensor).__name__}"
        )

    output = output_tensor.float()
    if loss_mask is None:
        raise RuntimeError("train_hetero.py requires a loss_mask for per-token loss")
    if output.shape != loss_mask.shape:
        raise RuntimeError(
            f"loss output shape {tuple(output.shape)} does not match loss_mask shape "
            f"{tuple(loss_mask.shape)}; per-token loss cannot be scaled correctly"
        )

    masked = output * loss_mask.float()
    num_tokens = loss_mask.float().sum().to(torch.int)
    loss_sum = masked.sum()
    return (
        loss_sum,
        num_tokens,
        {"lm loss": torch.stack((loss_sum.detach(), num_tokens.detach().float()))},
    )


def forward_step(data_iterator, model):
    """Forward step consumed by the MCore pipeline schedule."""
    with timeline_event("data.next"):
        batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    with timeline_event("data.to_cuda", cuda=True):
        batch = move_batch_to_cuda(batch)
    mark_modality_participation(model, batch)
    debug_rank("forward_step batch prepared")
    debug_rank("forward_step model call start")
    output_tensor, loss_mask = model(**batch)
    debug_rank("forward_step model call done")
    return output_tensor, partial(loss_func, loss_mask=loss_mask)


def move_batch_to_cuda(value):
    """Move tensors in nested batch structures to the current CUDA device."""
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch_to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_cuda(item) for item in value)
    return value


def train_step(
    args: argparse.Namespace,
    model: MimoModel,
    topology: HeteroTopology,
    optimizer,
    opt_param_scheduler,
    communicator: MultiModulePipelineCommunicator,
    data_iterator,
) -> TrainStepResult:
    """Run one Megatron-shaped hetero training step."""
    zero_active_grad_buffers(model)
    reset_modality_participation(model)
    optimizer.zero_grad()

    debug_rank("starting forward/backward schedule")
    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=[model],
        num_microbatches=args.num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
        p2p_communicator=communicator,
        pg_collection=topology.schedule_pg_collection,
    )
    debug_rank("schedule complete")

    debug_rank("optimizer step starting")
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    update_successful = reduce_update_success(update_successful)
    debug_rank("optimizer step complete")

    if update_successful:
        opt_param_scheduler.step(increment=get_global_batch_size(args))
        skipped_iter = 0
    else:
        # Match Megatron train_step semantics: failed updates skip LR advancement but
        # do not abort the run.
        skipped_iter = 1

    return TrainStepResult(
        losses=losses,
        skipped_iter=skipped_iter,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def reduce_update_success(update_successful: bool) -> bool:
    """Match Megatron's cross-rank success agreement for hetero process groups."""
    value = torch.tensor([1 if update_successful else 0], dtype=torch.int, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())
