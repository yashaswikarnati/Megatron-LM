# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient finalization and DDP sync helpers for heterogeneous MIMO training."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.runtime import active_ddp_modules
from examples.mimo.training.hetero.topology import HeteroTopology
from examples.mimo.utils.hetero import debug_rank, is_process_group_member
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.utils import is_pp_last_stage


def configure_grad_sync(mimo_model: MimoModel, topology: HeteroTopology) -> None:
    """Configure grad-finalization callbacks consumed by the pipeline schedule."""
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

        global_num_tokens = torch.zeros(1, dtype=torch.float32, device="cuda")
        if is_token_source_rank():
            # MCore has already summed loss-mask token counts across microbatches
            # for this gradient-accumulation step. Match Megatron's normalization
            # domain by reducing the language last-stage count over DP and CP.
            token_count = num_tokens.to(device="cuda", dtype=torch.float32).sum().view(1)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=language_pg.dp_cp)
            if dist.get_rank(language_pg.dp_cp) == 0:
                global_num_tokens.copy_(token_count)
        # Publish the already DP/CP-reduced language token count to encoder ranks too.
        dist.all_reduce(global_num_tokens, op=dist.ReduceOp.MAX)
        global_num_tokens_value = global_num_tokens.item()

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

        if global_num_tokens_value > 0:
            scale = 1.0 / global_num_tokens_value
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


def zero_active_grad_buffers(mimo_model: MimoModel) -> None:
    """Clear MCore DDP grad buffers before each training iteration."""
    for module in active_ddp_modules(mimo_model):
        module.zero_grad_buffer()


def build_no_sync_func(mimo_model: MimoModel):
    """Build a no_sync context spanning all active MIMO submodules."""

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    return no_sync_func
