# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient finalization and DDP sync helpers for heterogeneous MIMO training."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist

from examples.mimo.training.hetero.runtime import iter_active_ddp_modules
from examples.mimo.training.hetero.topology import HeteroTopology
from examples.mimo.utils.hetero import debug_rank, is_process_group_member
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.utils import is_pp_last_stage

# Sentinel attribute set on a modality submodule by forward_step when its rank
# processed image input this step. Used instead of scanning grad buffers.
_PARTICIPATED_ATTR = "_mimo_rank_processed_input"


def mark_modality_participation(model, batch) -> None:
    """Tag each modality submodule with whether this rank has image input
    this step. Called from forward_step before the model forward.
    """
    if not hasattr(model, "modality_submodules"):
        return
    images = batch.get("images") if isinstance(batch, dict) else None
    if isinstance(images, torch.Tensor):
        had_input = images.numel() > 0
    elif isinstance(images, (list, tuple)):
        had_input = len(images) > 0
    else:
        had_input = False
    for submodule in model.modality_submodules.values():
        if submodule is not None:
            setattr(submodule, _PARTICIPATED_ATTR, had_input)


def reset_modality_participation(mimo_model: MimoModel) -> None:
    """Clear per-step participation flags at the top of each train_step."""
    for submodule in mimo_model.modality_submodules.values():
        if submodule is not None:
            setattr(submodule, _PARTICIPATED_ATTR, False)


def _vision_participation_count(submodule, vision_dp_group) -> float:
    """All-reduce a 1-element bool across the vision DP group to get the
    number of DP ranks that processed image input this step.
    """
    val = 1.0 if getattr(submodule, _PARTICIPATED_ATTR, False) else 0.0
    indicator = torch.tensor([val], dtype=torch.float32, device="cuda")
    dist.all_reduce(indicator, op=dist.ReduceOp.SUM, group=vision_dp_group)
    return float(indicator.item())


def configure_grad_sync(args, mimo_model: MimoModel, topology: HeteroTopology) -> None:
    """Configure grad-finalization callbacks consumed by the pipeline schedule."""
    language_pg = topology.language_pg
    vision_pg = topology.vision_pg
    correct_encoder_grad = bool(
        getattr(args, "correct_encoder_grad_for_partial_participation", False)
    )

    def is_token_source_rank() -> bool:
        return (
            is_process_group_member(getattr(language_pg, "pp", None))
            and is_process_group_member(getattr(language_pg, "tp", None))
            and is_pp_last_stage(language_pg.pp)
            and language_pg.tp.rank() == 0
        )

    def finalize_grads_func(_model_list, num_tokens, force_all_reduce=False, **_kwargs):
        if num_tokens is None:
            raise RuntimeError("hetero train loop expects calculate_per_token_loss=True")

        global_num_tokens = torch.zeros(1, dtype=torch.float32, device="cuda")
        if is_token_source_rank():
            # MCore has already summed loss-mask token counts across microbatches
            # for this gradient-accumulation step. Reduce over DP/CP to match
            # Megatron's normalization domain.
            token_count = num_tokens.to(device="cuda", dtype=torch.float32).sum().view(1)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=language_pg.dp_cp)
            if dist.get_rank(language_pg.dp_cp) == 0:
                global_num_tokens.copy_(token_count)
        # Publish the language-side count to encoder ranks too.
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

        # Combine the per-token normalization with any partial-participation
        # correction into a single scale_gradients call per submodule.
        lang_scale = 1.0 / global_num_tokens_value if global_num_tokens_value > 0 else 0.0
        if lang_scale != 0.0 and mimo_model.language_model is not None:
            debug_rank("scaling language grads")
            mimo_model.language_model.scale_gradients(lang_scale)

        for submodule in mimo_model.modality_submodules.values():
            if submodule is None:
                continue
            vision_scale = lang_scale
            if correct_encoder_grad and vision_pg is not None:
                vision_dp_group = getattr(vision_pg, "dp", None)
                if is_process_group_member(vision_dp_group):
                    vision_dp_size = dist.get_world_size(vision_dp_group)
                    if vision_dp_size > 1:
                        participation = _vision_participation_count(
                            submodule, vision_dp_group
                        )
                        debug_rank(
                            f"vision participation: {participation}/{vision_dp_size}"
                        )
                        if 0.0 < participation < vision_dp_size:
                            vision_scale *= vision_dp_size / participation

            debug_rank("finalizing vision grads")
            finalize_model_grads(
                [submodule],
                num_tokens=None,
                pg_collection=vision_pg,
                force_all_reduce=force_all_reduce,
            )
            debug_rank("vision grads finalized")
            if vision_scale != 0.0:
                debug_rank("scaling vision grads")
                submodule.scale_gradients(vision_scale)

    mimo_model.config.no_sync_func = build_no_sync_func(mimo_model)
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device="cuda", requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )


def zero_active_grad_buffers(mimo_model: MimoModel) -> None:
    """Clear MCore DDP grad buffers before each training iteration."""
    for module in iter_active_ddp_modules(mimo_model):
        module.zero_grad_buffer()


def build_no_sync_func(mimo_model: MimoModel):
    """Build a no_sync context spanning all active MIMO submodules."""

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            for module in iter_active_ddp_modules(mimo_model):
                stack.enter_context(module.no_sync())
            yield

    return no_sync_func
