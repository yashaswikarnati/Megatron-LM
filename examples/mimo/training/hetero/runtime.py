# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model runtime construction for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from typing import Iterator, Optional

import torch

from examples.mimo.model_providers.nemotron_moe_vlm import (
    get_vision_encoder_module,
    iter_vision_projection_modules,
    language_model_spec,
    vision_submodules_spec,
)
from examples.mimo.training.hetero.topology import HeteroTopology, is_rank_in_grid
from examples.mimo.utils.hetero import debug_rank, get_group_rank_or
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def build_mimo_runtime(args: argparse.Namespace, topology: HeteroTopology) -> MimoModel:
    """Build the MIMO model and wrap active modules in MCore DDP."""
    language_pg = topology.language_pg
    vision_pg = topology.vision_pg
    rank_in_language_grid = is_rank_in_grid(topology.llm_grid)
    rank_in_encoder_grid = is_rank_in_grid(topology.encoder_grid)
    debug_rank(
        "building model specs "
        f"rank_in_encoder={rank_in_encoder_grid} rank_in_language={rank_in_language_grid}"
    )
    # The CUDA RNG tracker is process-global; this runtime assumes non-colocated module grids,
    # so each rank configures RNG state for exactly one module role.
    if rank_in_language_grid:
        configure_module_rng(args, language_pg, role_seed_offset=20_000)
    elif rank_in_encoder_grid:
        configure_module_rng(args, vision_pg, role_seed_offset=10_000)

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

    wrap_active_modules_with_ddp(args, mimo_model, topology)
    _propagate_tp_groups_for_checkpoint(mimo_model, topology)
    return mimo_model


def _propagate_tp_groups_for_checkpoint(mimo_model: MimoModel, topology: HeteroTopology) -> None:
    """Set self.tp_group on every active submodule so dist-ckpt save/load works.

    `MegatronModule.sharded_state_dict` falls back to
    `parallel_state.get_tensor_model_parallel_group()` when `self.tp_group` is missing,
    which is unset in the hetero loop. A few descendants don't set tp_group themselves
    (e.g. `ExtendedRMSNorm` in `megatron/core/ssm/mamba_mixer.py:93`, RADIO encoder
    internal submodules), so we walk each active branch once after construction and
    stamp the correct group.

    This MUST run after every constructor that sets tp_group has executed — i.e.
    after `wrap_active_modules_with_ddp` returns. A future refactor that lazily sets
    tp_group inside forward will silently shadow this value (since
    `_stamp_tp_group` uses `hasattr` to skip already-set attributes); if you see
    spurious "unexpected tp_group" behavior, that's the likely culprit.
    """
    rank_in_language_grid = is_rank_in_grid(topology.llm_grid)
    rank_in_encoder_grid = is_rank_in_grid(topology.encoder_grid)

    if rank_in_language_grid and mimo_model.language_model is not None:
        _stamp_tp_group(mimo_model.language_model, topology.language_pg.tp)

    if rank_in_encoder_grid and topology.encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[topology.encoder_name]
        if submodule is not None:
            _stamp_tp_group(submodule, topology.vision_pg.tp)


def _stamp_tp_group(module: torch.nn.Module, tp_group) -> None:
    """Stamp `tp_group` on every descendant that doesn't already carry one.

    Mirrors the `if not hasattr(self, 'tp_group')` fallback that
    `MegatronModule.sharded_state_dict` and similar paths perform. Modules that
    explicitly set `tp_group` (to a real group or None) are left alone.
    """
    inner = module.module if isinstance(module, DistributedDataParallel) else module
    for sub in inner.modules():
        if not hasattr(sub, "tp_group"):
            sub.tp_group = tp_group


def wrap_active_modules_with_ddp(
    args: argparse.Namespace, mimo_model: MimoModel, topology: HeteroTopology
) -> None:
    """Freeze and DDP-wrap active local MIMO modules."""
    language_ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=args.overlap_grad_reduce,
        bucket_size=args.ddp_bucket_size if args.ddp_bucket_size > 0 else None,
        use_distributed_optimizer=True,
    )
    vision_ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False,
        bucket_size=args.ddp_bucket_size if args.ddp_bucket_size > 0 else None,
        use_distributed_optimizer=True,
    )
    if mimo_model.language_model is not None:
        if args.freeze_lm:
            set_module_requires_grad(mimo_model.language_model, False)
        debug_rank("wrapping language model in DDP")
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=language_ddp_config,
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
            ddp_config=vision_ddp_config,
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


def configure_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed module init and CUDA RNG tracker for the active module role.

    The seed is identical across DP/CP replicas for a module PP stage, and differs across
    module roles and PP stages.
    """
    pp_rank = get_group_rank_or(getattr(pg_collection, "pp", None))
    tp_rank = get_group_rank_or(getattr(pg_collection, "tp", None))
    ep_rank = get_group_rank_or(getattr(pg_collection, "ep", None))
    expt_tp_rank = get_group_rank_or(getattr(pg_collection, "expt_tp", None))
    seed = args.seed + role_seed_offset + (100 * pp_rank)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(
        seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=expt_tp_rank, force_reset_rng=True
    )


def iter_active_ddp_modules(mimo_model: MimoModel) -> Iterator[DistributedDataParallel]:
    """Yield active DDP-wrapped submodules owned by this rank."""
    if isinstance(mimo_model.language_model, DistributedDataParallel):
        yield mimo_model.language_model
    for submodule in mimo_model.modality_submodules.values():
        if isinstance(submodule, DistributedDataParallel):
            yield submodule
