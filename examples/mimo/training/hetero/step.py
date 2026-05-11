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
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_last_stage

from examples.mimo.training.hetero.runtime import (
    HeteroRuntime,
    build_no_sync_func,
    zero_active_grad_buffers,
)
from examples.mimo.training.hetero.scheduler import get_global_batch_size
from examples.mimo.training.hetero.topology import HeteroTopology
from examples.mimo.utils.hetero import debug_rank, is_process_group_member


@dataclass
class TrainStepResult:
    """Megatron-style result returned by one hetero training step."""

    losses: list[dict[str, Any]]
    skipped_iter: int
    update_successful: bool
    grad_norm: Optional[float]
    num_zeros_in_grad: Optional[int]


def wire_training_hooks(runtime: HeteroRuntime, topology: HeteroTopology) -> None:
    """Attach MIMO-specific grad sync hooks expected by the pipeline schedule."""
    mimo_model = runtime.model
    language_pg = topology.language_pg
    vision_pg = topology.vision_pg

    def is_token_source_rank() -> bool:
        return (
            is_process_group_member(getattr(language_pg, "pp", None))
            and is_process_group_member(getattr(language_pg, "tp", None))
            and is_pp_last_stage(language_pg.pp)
            and language_pg.tp.rank() == 0
        )

    def finalize_grads_func(_model_list, num_tokens, force_all_reduce=False, **_kwargs):
        if num_tokens is None:
            raise RuntimeError("train_hetero.py expects calculate_per_token_loss=True")

        token_count = torch.zeros(1, dtype=torch.float32, device="cuda")
        if is_token_source_rank():
            token_count[0] = num_tokens.to(device="cuda", dtype=torch.float32).sum()
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        global_num_tokens = token_count.item()

        if mimo_model.language_model is not None:
            debug_rank("finalizing language grads")
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
            debug_rank("language grads finalized")
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                debug_rank("finalizing vision grads")
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=vision_pg,
                    force_all_reduce=force_all_reduce,
                )
                debug_rank("vision grads finalized")

        if global_num_tokens > 0:
            scale = 1.0 / global_num_tokens
            if mimo_model.language_model is not None:
                debug_rank("scaling language grads")
                mimo_model.language_model.scale_gradients(scale)
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    debug_rank("scaling vision grads")
                    submodule.scale_gradients(scale)

    mimo_model.config.no_sync_func = build_no_sync_func(mimo_model)
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device="cuda", requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )


def loss_func(loss_mask: Optional[torch.Tensor], output_tensor):
    """Return raw loss sum, local token count, and logging tensors."""
    if output_tensor is None:
        zero = torch.tensor(0.0, device="cuda", requires_grad=True)
        zero_count = torch.tensor(0, device="cuda", dtype=torch.int)
        return zero, zero_count, {"lm loss sum": zero.detach(), "lm tokens": zero_count}

    if isinstance(output_tensor, dict):
        output = output_tensor.get(
            MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
        )
    else:
        output = output_tensor

    if output is None:
        zero = torch.tensor(0.0, device="cuda", requires_grad=True)
        zero_count = torch.tensor(0, device="cuda", dtype=torch.int)
        return zero, zero_count, {"lm loss sum": zero.detach(), "lm tokens": zero_count}

    output = output.float()
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
        {"lm loss sum": loss_sum.detach(), "lm tokens": num_tokens.detach()},
    )


def forward_step(data_iterator, model):
    """Forward step consumed by the MCore pipeline schedule."""
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    batch = move_batch_to_cuda(batch)
    debug_rank("forward_step batch prepared")
    debug_rank("forward_step model call start")
    output_tensor, loss_mask = model(**batch)
    debug_rank("forward_step model call done")
    return output_tensor, partial(loss_func, loss_mask)


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
    runtime: HeteroRuntime,
    topology: HeteroTopology,
    optimizer,
    opt_param_scheduler,
    communicator: MultiModulePipelineCommunicator,
    data_iterator,
) -> TrainStepResult:
    """Run one Megatron-shaped hetero training step."""
    zero_active_grad_buffers(runtime.model)
    optimizer.zero_grad()

    debug_rank("starting forward/backward schedule")
    losses = schedule.forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=[runtime.model],
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
