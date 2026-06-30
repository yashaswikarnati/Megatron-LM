# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank runtime setup (RNG seeding, freezing, DDP wrapping) for hetero MIMO training."""

from __future__ import annotations

import argparse
from dataclasses import replace

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
from megatron.training.models.dist_utils import _ddp_wrap
from megatron.training.utils import print_rank_0


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


def _maybe_float16_wrap(module: torch.nn.Module, config, is_encoder: bool) -> torch.nn.Module:
    """Wrap a submodule in Float16Module when fp16/bf16 is enabled; encoders keep bf16 outputs."""
    if not (getattr(config, "fp16", False) or getattr(config, "bf16", False)):
        # ModalitySubmodules is a container around independently configured encoders and
        # projections, so it does not own a config itself. The shared distributed-wrapper
        # lifecycle resolves the model config from the module it wraps; publish the config
        # selected by _module_config for the unwrapped fp32 path.
        if getattr(module, "config", None) is None:
            module.config = config
        return module
    cls = _EncoderFloat16Module if is_encoder else Float16Module
    return cls(config, module)


def wrap_active_modules_with_ddp(
    args: argparse.Namespace,
    mimo_model: MimoModel,
    topology: HeteroTopology,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    data_parallel_random_init: bool = False,
) -> None:
    """Freeze (per --freeze-* flags), Float16Module-wrap, and DDP-wrap each active module."""
    if mimo_model.language_model is not None:
        if getattr(args, "freeze_lm", False):
            mimo_model.language_model.requires_grad_(False)
        module_ddp_config = replace(ddp_config)
        lm_config = _module_config(mimo_model.language_model)
        lm_module = _maybe_float16_wrap(mimo_model.language_model, lm_config, is_encoder=False)
        print_rank_0("wrapping language model in DDP")
        mimo_model.language_model = _ddp_wrap(
            [lm_module],
            data_parallel_random_init,
            module_ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp,
            use_torch_fsdp2,
            pg_collection=topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
        )[0]

    for name, submodule in mimo_model.modality_submodules.items():
        if submodule is None or name not in topology.module_pgs:
            continue
        _freeze_modality_submodule(submodule, args)
        module_ddp_config = replace(
            ddp_config,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        )
        enc_config = _module_config(submodule)
        enc_module = _maybe_float16_wrap(submodule, enc_config, is_encoder=True)
        print_rank_0(f"wrapping modality submodule {name!r} in DDP")
        mimo_model.modality_submodules[name] = _ddp_wrap(
            [enc_module],
            data_parallel_random_init,
            module_ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp,
            use_torch_fsdp2,
            pg_collection=topology.module_pgs[name],
        )[0]
