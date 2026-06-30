# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime setup (RNG seeding, freezing, DDP wrapping) for hetero MIMO training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Callable

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.training.initialize import _set_random_seed
from megatron.training.models.dist_utils import (
    prepare_existing_model_chunks_for_distributed_training,
)
from megatron.training.utils import print_rank_0


class _EncoderFloat16Module(Float16Module):
    """Float16Module that keeps encoder outputs in model precision for the bridge."""

    def forward(self, *inputs, fp32_output=False, **kwargs):  # noqa: D102
        return super().forward(*inputs, fp32_output=fp32_output, **kwargs)


def _seed_module_rng(
    args: argparse.Namespace,
    pg_collection: ProcessGroupCollection,
    role_seed_offset: int,
    data_parallel_random_init: bool,
) -> None:
    """Seed one active role through the stock explicit-process-group path."""
    _set_random_seed(
        args.seed + role_seed_offset,
        data_parallel_random_init,
        getattr(args, "te_rng_tracker", False),
        getattr(args, "inference_rng_tracker", False),
        use_cudagraphable_rng=getattr(args, "cuda_graph_impl", "none") != "none",
        pp_group=pg_collection.pp,
        dp_group=pg_collection.dp,
        tp_group=pg_collection.tp,
        ep_group=pg_collection.ep,
        etp_group=pg_collection.expt_tp,
    )


def _freeze_modality_submodule(submodule: torch.nn.Module, args: argparse.Namespace) -> None:
    """Freeze the encoder backbone (--freeze-vit) and/or projector (--freeze-projection)."""
    if getattr(args, "freeze_vit", False):
        submodule.encoders.requires_grad_(False)
    if getattr(args, "freeze_projection", False):
        submodule.input_projections.requires_grad_(False)
        submodule.output_projections.requires_grad_(False)


def _module_config(module: torch.nn.Module):
    """Return the module's own config, else the first descendant config (e.g. an encoder)."""
    config = getattr(module, "config", None)
    if config is not None:
        return config
    for child in module.modules():
        config = getattr(child, "config", None)
        if config is not None:
            return config
    raise ValueError("Cannot resolve a config for DDP wrapping from module")


def prepare_active_modules_for_distributed_training(
    args: argparse.Namespace,
    mimo_model: MimoModel,
    topology: HeteroTopology,
    ddp_config: DistributedDataParallelConfig | None,
    built_with_meta_device: bool,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
) -> None:
    """Apply the shared stock lifecycle separately to each active MIMO child."""
    if wrap_with_ddp and ddp_config is None:
        raise ValueError("ddp_config is required when wrap_with_ddp is True")

    if mimo_model.language_model is not None:
        if getattr(args, "freeze_lm", False):
            mimo_model.language_model.requires_grad_(False)
        language_config = _module_config(mimo_model.language_model)
        language_ddp_config = replace(ddp_config) if ddp_config is not None else None
        print_rank_0("preparing language model for distributed training")
        mimo_model.language_model = prepare_existing_model_chunks_for_distributed_training(
            [mimo_model.language_model],
            language_config,
            topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            built_with_meta_device=built_with_meta_device,
            ddp_config=language_ddp_config,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            mixed_precision_wrapper=mixed_precision_wrapper,
        )[0]

    for name, submodule in mimo_model.modality_submodules.items():
        if submodule is None or name not in topology.module_pgs:
            continue
        _freeze_modality_submodule(submodule, args)
        encoder_config = _module_config(submodule)
        if getattr(submodule, "config", None) is None:
            # The modality container lacks config; publish it for the fp32 DDP path.
            submodule.config = encoder_config
        encoder_ddp_config = (
            replace(
                ddp_config,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            )
            if ddp_config is not None
            else None
        )
        encoder_wrapper = (
            _EncoderFloat16Module
            if mixed_precision_wrapper is Float16Module
            else mixed_precision_wrapper
        )
        print_rank_0(f"preparing modality submodule {name!r} for distributed training")
        mimo_model.modality_submodules[name] = (
            prepare_existing_model_chunks_for_distributed_training(
                [submodule],
                encoder_config,
                topology.module_pgs[name],
                built_with_meta_device=built_with_meta_device,
                ddp_config=encoder_ddp_config,
                overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
                use_megatron_fsdp=use_megatron_fsdp,
                use_torch_fsdp2=use_torch_fsdp2,
                wrap_with_ddp=wrap_with_ddp,
                data_parallel_random_init=data_parallel_random_init,
                mixed_precision_wrapper=encoder_wrapper,
            )[0]
        )
