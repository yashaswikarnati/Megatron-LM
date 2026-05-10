# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model runtime construction for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from examples.mimo.model_providers.hetero_vlm import (
    get_vision_encoder_module,
    iter_vision_projection_modules,
    language_model_spec,
    vision_submodules_spec,
)
from examples.mimo.training.hetero.topology import HeteroTopology, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, get_group_rank_or


@dataclass
class HeteroRuntime:
    """Runtime-owned model state for a hetero MIMO training run."""

    model: MimoModel

    def destroy(self) -> None:
        """Destroy runtime-owned model communication state."""
        self.model.destroy()


def build_mimo_runtime(args: argparse.Namespace, topology: HeteroTopology) -> HeteroRuntime:
    """Build the MIMO model and wrap active modules in MCore DDP."""
    language_pg = topology.language_pg
    vision_pg = topology.vision_pg
    rank_in_language_grid = is_rank_in_grid(topology.llm_grid)
    rank_in_encoder_grid = is_rank_in_grid(topology.encoder_grid)
    debug_rank(
        "building model specs "
        f"rank_in_encoder={rank_in_encoder_grid} rank_in_language={rank_in_language_grid}"
    )
    if rank_in_language_grid:
        set_model_init_seed(args, language_pg, role_offset=20_000)
        initialize_model_parallel_rng(args, language_pg)
    elif rank_in_encoder_grid:
        set_model_init_seed(args, vision_pg, role_offset=10_000)
        initialize_model_parallel_rng(args, vision_pg)

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec(
            args, language_pg if rank_in_language_grid else None, topology.llm_grid
        ),
        modality_submodules_spec={
            topology.encoder_name: vision_submodules_spec(
                args, vision_pg if rank_in_encoder_grid else None, topology.encoder_grid
            )
        },
        special_token_ids={topology.encoder_name: args.image_token_id},
        module_to_grid_map=topology.module_to_grid_map,
    )

    debug_rank("constructing MimoModel")
    mimo_model = MimoModel(
        mimo_config,
        cp_group=language_pg.cp if rank_in_language_grid else None,
        tp_group=language_pg.tp if rank_in_language_grid else None,
    )
    debug_rank("moving MimoModel to cuda")
    mimo_model.to(torch.device("cuda"))
    if not args.fp32:
        mimo_model.to(torch.bfloat16)
    debug_rank("MimoModel moved to target dtype/device")

    wrap_active_modules(args, mimo_model, topology)
    broadcast_active_params(mimo_model)
    return HeteroRuntime(model=mimo_model)


def wrap_active_modules(
    args: argparse.Namespace, mimo_model: MimoModel, topology: HeteroTopology
) -> None:
    """Freeze and DDP-wrap active local MIMO modules."""
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=args.overlap_grad_reduce,
        bucket_size=args.ddp_bucket_size if args.ddp_bucket_size > 0 else None,
        use_distributed_optimizer=True,
    )
    if mimo_model.language_model is not None:
        if args.freeze_lm:
            set_module_requires_grad(mimo_model.language_model, False)
        debug_rank("wrapping language model in DDP")
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=topology.language_pg,
        )
        debug_rank("language model DDP ready")

    if topology.encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[topology.encoder_name]
        if submodule is None:
            return

        encoder_module = get_vision_encoder_module(args, submodule)
        if args.freeze_vit:
            set_module_requires_grad(encoder_module, False)
        if args.freeze_projection:
            for projection in iter_vision_projection_modules(submodule):
                set_module_requires_grad(projection, False)
        debug_rank("wrapping vision submodule in DDP")
        mimo_model.modality_submodules[topology.encoder_name] = DistributedDataParallel(
            config=encoder_module.config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=topology.vision_pg,
        )
        debug_rank("vision submodule DDP ready")


def set_module_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool) -> None:
    """Set requires_grad for every parameter in a module when the module exists."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_model_init_seed(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_offset: int
) -> None:
    """Seed CPU model init consistently across TP/DP peers for one module role."""
    pp_rank = get_group_rank_or(getattr(pg_collection, "pp", None))
    torch.manual_seed(args.seed + role_offset + (100 * pp_rank))


def initialize_model_parallel_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection
) -> None:
    """Initialize CUDA RNG tracker using the active module's hetero process groups."""
    pp_rank = get_group_rank_or(getattr(pg_collection, "pp", None))
    tp_rank = get_group_rank_or(getattr(pg_collection, "tp", None))
    ep_rank = get_group_rank_or(getattr(pg_collection, "ep", None))
    expt_tp_rank = get_group_rank_or(getattr(pg_collection, "expt_tp", None))
    model_parallel_cuda_manual_seed(
        args.seed + (100 * pp_rank),
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        etp_rank=expt_tp_rank,
        force_reset_rng=True,
    )


def active_ddp_modules(mimo_model: MimoModel) -> list[DistributedDataParallel]:
    """Return active DDP-wrapped submodules owned by this rank."""
    modules = []
    if isinstance(mimo_model.language_model, DistributedDataParallel):
        modules.append(mimo_model.language_model)
    modules.extend(
        submodule
        for submodule in mimo_model.modality_submodules.values()
        if isinstance(submodule, DistributedDataParallel)
    )
    return modules


def broadcast_active_params(mimo_model: MimoModel) -> None:
    """Synchronize initial parameters across each module's DP groups."""
    for module in active_ddp_modules(mimo_model):
        module.broadcast_params()


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
