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
    rank_in_encoder_grid = topology.encoder_grid is not None and is_rank_in_grid(
        topology.encoder_grid
    )
    debug_rank(
        "building model specs "
        f"rank_in_encoder={rank_in_encoder_grid} rank_in_language={rank_in_language_grid}"
    )
    # The CUDA RNG tracker is process-global; this runtime assumes non-colocated module grids,
    # so each rank configures RNG state for exactly one module role.
    if rank_in_language_grid:
        configure_module_rng(args, language_pg, role_seed_offset=20_000)
    elif rank_in_encoder_grid:
        assert vision_pg is not None
        configure_module_rng(args, vision_pg, role_seed_offset=10_000)

    modality_submodules_spec = {}
    special_token_ids = {}
    if topology.encoder_grid is not None:
        modality_submodules_spec[topology.encoder_name] = vision_submodules_spec(
            args, vision_pg if rank_in_encoder_grid else None, topology.encoder_grid
        )
        special_token_ids[topology.encoder_name] = args.image_token_id

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec(
            args, language_pg if rank_in_language_grid else None, topology.llm_grid
        ),
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
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
    return mimo_model


def _resolve_bucket_size(
    args: argparse.Namespace, module: Optional[torch.nn.Module]
) -> Optional[int]:
    """Resolve DDP bucket_size for a module.

    Precedence:
    1. --ddp-num-buckets (set): bucket_size = num_params // num_buckets.
    2. --ddp-bucket-size > 0: use that value.
    3. Else None (mcore auto-default = max(40M, 1M * dp_size)).
    """
    num_buckets = getattr(args, "ddp_num_buckets", None)
    if num_buckets is not None:
        if num_buckets <= 0:
            raise ValueError("--ddp-num-buckets must be > 0 when set")
        if args.ddp_bucket_size and args.ddp_bucket_size > 0:
            raise ValueError(
                "--ddp-num-buckets and --ddp-bucket-size are mutually exclusive"
            )
        if module is None:
            return None
        num_params = sum(p.numel() for p in module.parameters())
        if num_params <= 0:
            return None
        return max(1, num_params // num_buckets)
    if args.ddp_bucket_size and args.ddp_bucket_size > 0:
        return args.ddp_bucket_size
    return None


def wrap_active_modules_with_ddp(
    args: argparse.Namespace, mimo_model: MimoModel, topology: HeteroTopology
) -> None:
    """Freeze and DDP-wrap active local MIMO modules."""
    pad_buckets = getattr(args, "ddp_pad_buckets_for_high_nccl_busbw", False)
    if mimo_model.language_model is not None:
        if args.freeze_lm:
            set_module_requires_grad(mimo_model.language_model, False)
        language_ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=args.overlap_grad_reduce,
            overlap_param_gather=getattr(args, "overlap_param_gather", False),
            bucket_size=_resolve_bucket_size(args, mimo_model.language_model),
            pad_buckets_for_high_nccl_busbw=pad_buckets,
            use_distributed_optimizer=True,
            # Keep main_grad in fp32. Default False → bf16 main_grad → step-2
            # weight drift after Adam.
            grad_reduce_in_fp32=getattr(args, "accumulate_allreduce_grads_in_fp32", True),
        )
        debug_rank("wrapping language model in DDP")
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=language_ddp_config,
            module=mimo_model.language_model,
            pg_collection=topology.language_pg,
        )
        debug_rank("language model DDP ready")

    if (
        topology.encoder_grid is not None
        and topology.encoder_name in mimo_model.modality_submodules
    ):
        assert topology.vision_pg is not None
        submodule = mimo_model.modality_submodules[topology.encoder_name]
        if submodule is None:
            return

        encoder_module = get_vision_encoder_module(args, submodule)
        if args.freeze_vit:
            set_module_requires_grad(encoder_module, False)
        if args.freeze_projection:
            for projection in iter_vision_projection_modules(submodule):
                set_module_requires_grad(projection, False)
        # Vision DDP keeps all overlap off: actual-data batches may be text-only,
        # so some encoder DP ranks see zero grads/params per step; overlap'd
        # collectives are not safe under that partial participation.
        vision_ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            bucket_size=_resolve_bucket_size(args, submodule),
            pad_buckets_for_high_nccl_busbw=pad_buckets,
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=getattr(args, "accumulate_allreduce_grads_in_fp32", True),
        )
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
