# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime setup (RNG seeding, freezing, DDP wrapping) for hetero MIMO training."""

from __future__ import annotations

import argparse

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_pg_rank, get_pg_size
from megatron.training.initialize import _set_random_seed
from megatron.training.models.dist_utils import (
    prepare_existing_model_chunks_for_distributed_training,
)


class _EncoderFloat16Module(Float16Module):
    """Float16Module that keeps encoder outputs in model precision for the bridge."""

    def forward(self, *inputs, fp32_output=False, **kwargs):  # noqa: D102
        return super().forward(*inputs, fp32_output=fp32_output, **kwargs)


def configure_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed the CUDA RNG tracker for one module role from its tp/pp coordinates plus the offset.

    The seed is shared across a module's DP/CP replicas but distinct across PP stages and roles,
    so disjoint modules (and stages) get independent RNG state. Caller invokes once per active
    module on this rank.
    """
    for _required in ("pp", "tp", "ep", "expt_tp"):
        assert (
            getattr(pg_collection, _required, None) is not None
        ), f"pg_collection passed to configure_module_rng must define {_required}"
    pp_rank = get_pg_rank(pg_collection.pp)
    tp_rank = get_pg_rank(pg_collection.tp)
    ep_rank = get_pg_rank(pg_collection.ep)
    expt_tp_rank = get_pg_rank(pg_collection.expt_tp)
    seed = args.seed + role_seed_offset + (100 * pp_rank)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(
        seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=expt_tp_rank, force_reset_rng=True
    )


def _seed_module_rng(
    args: argparse.Namespace, pg_collection: ProcessGroupCollection, role_seed_offset: int
) -> None:
    """Seed host + CUDA RNG for one module role from its parallel groups (no mpu here)."""
    _set_random_seed(
        args.seed + role_seed_offset,
        args.data_parallel_random_init,
        args.te_rng_tracker,
        args.inference_rng_tracker,
        use_cudagraphable_rng=args.cuda_graph_impl != "none",
        pp_group=pg_collection.pp,
        dp_group=pg_collection.dp_cp,
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


def _ddp_config_for(args: argparse.Namespace, *, overlap_grad_reduce: bool) -> DistributedDataParallelConfig:
    """Per-module DDP config; bucket size is resolved downstream by the stock wrap path."""
    return DistributedDataParallelConfig(
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_grad_reduce and getattr(args, "overlap_param_gather", False),
        num_buckets=getattr(args, "ddp_num_buckets", None),
        bucket_size=getattr(args, "ddp_bucket_size", None),
        pad_buckets_for_high_nccl_busbw=getattr(args, "ddp_pad_buckets_for_high_nccl_busbw", False),
        use_distributed_optimizer=True,
        grad_reduce_in_fp32=getattr(args, "accumulate_allreduce_grads_in_fp32", True),
    )


def wrap_active_modules_with_ddp(
    args: argparse.Namespace,
    mimo_model: MimoModel,
    topology: HeteroTopology,
    *,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    data_parallel_random_init: bool = False,
    overlap_param_gather_with_optimizer_step: bool = False,
) -> None:
    """Freeze (per --freeze-* flags) and distributed-wrap each active module via the stock lifecycle."""

    def _prepare(module, pg_collection, *, is_encoder, overlap_grad_reduce):
        config = _module_config(module)
        return prepare_existing_model_chunks_for_distributed_training(
            [module],
            config,
            pg_collection,
            built_with_meta_device=config.init_model_with_meta_device,
            ddp_config=_ddp_config_for(args, overlap_grad_reduce=overlap_grad_reduce),
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            data_parallel_random_init=data_parallel_random_init,
            mixed_precision_wrapper=_EncoderFloat16Module if is_encoder else Float16Module,
        )[0]

    if mimo_model.language_model is not None:
        if getattr(args, "freeze_lm", False):
            mimo_model.language_model.requires_grad_(False)
        mimo_model.language_model = _prepare(
            mimo_model.language_model,
            topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            is_encoder=False,
            overlap_grad_reduce=getattr(args, "overlap_grad_reduce", False),
        )

    for name, submodule in mimo_model.modality_submodules.items():
        if submodule is None or name not in topology.module_pgs:
            continue
        _freeze_modality_submodule(submodule, args)
        mimo_model.modality_submodules[name] = _prepare(
            submodule, topology.module_pgs[name], is_encoder=True, overlap_grad_reduce=False
        )
