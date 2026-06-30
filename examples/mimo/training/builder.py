# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""ModelBuilder for heterogeneous MIMO (Nemotron6-MoE VLM) training on disjoint grids."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import torch

from examples.mimo.model_providers.nemotron_moe_vlm import (
    language_model_spec,
    vision_submodules_spec,
)
from examples.mimo.training.grad_sync import configure_grad_sync
from examples.mimo.training.runtime import _seed_module_rng, wrap_active_modules_with_ddp
from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.training.models.base import ModelBuilder, ModelConfig

# Per-role seed offsets keep the disjoint modules' RNG independent.
_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


@dataclass(kw_only=True)
class MimoBuildConfig(ModelConfig):
    """Model config carrier; only ``builder`` serializes (``_topology``/``_args`` are runtime-only)."""

    builder: ClassVar[str] = "examples.mimo.training.builder.MimoModelBuilder"
    _topology: Optional[HeteroTopology] = field(default=None)
    _args: Optional[argparse.Namespace] = field(default=None)


def _encoder_module_name(topology: HeteroTopology) -> Optional[str]:
    """Return the single modality (encoder) grid name, or ``None`` for an LLM-only run."""
    names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    return names[0] if names else None


def _resolve_role(topology: HeteroTopology):
    """Resolve this rank's module role from the grids (non-colocated: one grid per rank)."""
    encoder_name = _encoder_module_name(topology)
    rank_in_language = topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid()
    rank_in_encoder = (
        encoder_name is not None and topology.grids[encoder_name].is_current_rank_in_grid()
    )
    language_pg = topology.module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
    encoder_pg = topology.module_pgs.get(encoder_name) if encoder_name is not None else None
    return encoder_name, rank_in_language, rank_in_encoder, language_pg, encoder_pg


def _mimo_branch_name(topology: HeteroTopology) -> str:
    """Return this rank's MIMO branch ("language" or the encoder grid name)."""
    if topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid():
        return "language"
    encoder_name = _encoder_module_name(topology)
    return encoder_name if encoder_name is not None else "language"


class MimoModelBuilder(ModelBuilder["MimoModel", MimoBuildConfig]):
    """Build this rank's hetero ``MimoModel`` on the disjoint vision/language grids."""

    def __init__(self, model_config: MimoBuildConfig):
        super().__init__(model_config)
        assert model_config._topology is not None, "MimoBuildConfig requires a topology"
        assert model_config._args is not None, "MimoBuildConfig requires parsed args"
        self._topology: HeteroTopology = model_config._topology
        self._args: argparse.Namespace = model_config._args

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MimoModel:
        """Build this rank's bare ``MimoModel`` from its role/topology (ignores pg_collection)."""
        topology, args = self._topology, self._args
        encoder_name, rank_in_language, rank_in_encoder, language_pg, _ = _resolve_role(topology)

        modality_submodules_spec = {}
        special_token_ids = {}
        if encoder_name is not None:
            encoder_pg = topology.module_pgs.get(encoder_name)
            modality_submodules_spec[encoder_name] = vision_submodules_spec(
                args, encoder_pg if rank_in_encoder else None, topology.grids[encoder_name]
            )
            special_token_ids[encoder_name] = args.image_token_id

        mimo_config = MimoModelConfig(
            language_model_spec=language_model_spec(
                args,
                language_pg if rank_in_language else None,
                topology.grids[MIMO_LANGUAGE_MODULE_KEY],
            ),
            modality_submodules_spec=modality_submodules_spec,
            special_token_ids=special_token_ids,
            module_to_grid_map=topology.grids,
        )
        mimo_model = MimoModel(
            mimo_config,
            cp_group=language_pg.cp if rank_in_language else None,
            tp_group=language_pg.tp if rank_in_language else None,
        )
        mimo_model.to(torch.device("cuda"))
        return mimo_model

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: (
            Callable[[Any, MegatronModule], MegatronModule] | None
        ) = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[MimoModel]:
        """Build, per-submodule-DDP-wrap, and configure this rank's hetero MimoModel."""
        topology, args = self._topology, self._args
        _, rank_in_language, rank_in_encoder, language_pg, encoder_pg = _resolve_role(topology)

        # Seed per-role RNG before weight init so disjoint modules init independently.
        if rank_in_language:
            assert language_pg is not None
            _seed_module_rng(args, language_pg, _LANGUAGE_SEED_OFFSET)
        elif rank_in_encoder:
            assert encoder_pg is not None
            _seed_module_rng(args, encoder_pg, _ENCODER_SEED_OFFSET)

        mimo_model = self.build_model(pg_collection)

        if wrap_with_ddp:
            wrap_active_modules_with_ddp(
                args,
                mimo_model,
                topology,
                use_megatron_fsdp=use_megatron_fsdp,
                use_torch_fsdp2=use_torch_fsdp2,
                data_parallel_random_init=data_parallel_random_init,
                overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            )

        configure_grad_sync(args, mimo_model, topology)
        # Per-grid rng key namespace (read by stock save/load); torch_dist only.
        mimo_model.rng_state_key_prefix = f"mimo.{_mimo_branch_name(topology)}."
        return [mimo_model]
