# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model builder for the single-encoder heterogeneous MIMO training example."""

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
from examples.mimo.training.runtime import (
    _seed_module_rng,
    prepare_active_modules_for_distributed_training,
)
from examples.mimo.training.topology import HeteroTopology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks

_LANGUAGE_SEED_OFFSET = 20_000
_ENCODER_SEED_OFFSET = 10_000


@dataclass(kw_only=True)
class MimoBuildConfig(ModelConfig):
    """Runtime-only topology and arguments used by :class:`MimoModelBuilder`."""

    builder: ClassVar[str] = "examples.mimo.training.builder.MimoModelBuilder"
    _topology: Optional[HeteroTopology] = field(default=None)
    _args: Optional[argparse.Namespace] = field(default=None)


def _encoder_module_name(topology: HeteroTopology) -> Optional[str]:
    """Return the example's optional single encoder module name."""
    encoder_names = [
        name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY
    ]
    if len(encoder_names) > 1:
        raise ValueError(
            "The heterogeneous MIMO example currently supports at most one encoder module"
        )
    return encoder_names[0] if encoder_names else None


def _resolve_role(topology: HeteroTopology):
    """Resolve the language/encoder modules active on this rank."""
    encoder_name = _encoder_module_name(topology)
    rank_in_language = topology.grids[
        MIMO_LANGUAGE_MODULE_KEY
    ].is_current_rank_in_grid()
    rank_in_encoder = (
        encoder_name is not None
        and topology.grids[encoder_name].is_current_rank_in_grid()
    )
    language_pg = topology.module_pgs.get(MIMO_LANGUAGE_MODULE_KEY)
    encoder_pg = (
        topology.module_pgs.get(encoder_name) if encoder_name is not None else None
    )
    return encoder_name, rank_in_language, rank_in_encoder, language_pg, encoder_pg


class MimoModelBuilder(ModelBuilder[MimoModel, MimoBuildConfig]):
    """Build and prepare this rank's active heterogeneous MIMO module."""

    def __init__(self, model_config: MimoBuildConfig):
        super().__init__(model_config)
        if model_config._topology is None:
            raise ValueError("MimoBuildConfig requires a topology")
        if model_config._args is None:
            raise ValueError("MimoBuildConfig requires parsed args")
        self._topology = model_config._topology
        self._args = model_config._args

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MimoModel:
        """Build the bare rank-local MIMO model; the shared lifecycle places it later."""
        del pg_collection, pre_process, post_process, vp_stage
        topology = self._topology
        args = self._args
        encoder_name, rank_in_language, rank_in_encoder, language_pg, encoder_pg = (
            _resolve_role(topology)
        )

        modality_submodules_spec = {}
        special_token_ids = {}
        if encoder_name is not None:
            modality_submodules_spec[encoder_name] = vision_submodules_spec(
                args,
                encoder_pg if rank_in_encoder else None,
                topology.grids[encoder_name],
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
        return MimoModel(
            mimo_config,
            cp_group=language_pg.cp if rank_in_language else None,
            tp_group=language_pg.tp if rank_in_language else None,
        )

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule]
        | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[MimoModel]:
        """Seed, build, prepare, and configure the active rank-local MIMO model."""
        if wrap_with_ddp and ddp_config is None:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        topology = self._topology
        args = self._args
        _, rank_in_language, rank_in_encoder, language_pg, encoder_pg = _resolve_role(
            topology
        )
        if rank_in_language == rank_in_encoder:
            raise ValueError(
                "Non-colocated MIMO requires exactly one active language or encoder role per rank"
            )
        if rank_in_language:
            assert language_pg is not None
            _seed_module_rng(
                args,
                language_pg,
                _LANGUAGE_SEED_OFFSET,
                data_parallel_random_init,
            )
        elif rank_in_encoder:
            assert encoder_pg is not None
            _seed_module_rng(
                args,
                encoder_pg,
                _ENCODER_SEED_OFFSET,
                data_parallel_random_init,
            )

        built_with_meta_device = getattr(args, "init_model_with_meta_device", False)
        if built_with_meta_device:
            with torch.device("meta"):
                mimo_model = self.build_model(pg_collection)
        else:
            mimo_model = self.build_model(pg_collection)

        mimo_model.model_type = model_type
        model_list = [mimo_model]
        _model = compose_hooks(self._model_config.pre_wrap_hooks)(model_list)
        if _model is not None:
            model_list = _model
        if len(model_list) != 1:
            raise ValueError(
                f"MIMO pre-wrap hooks must return exactly one outer model; got {len(model_list)}"
            )
        mimo_model = model_list[0]

        prepare_active_modules_for_distributed_training(
            args,
            mimo_model,
            topology,
            ddp_config=ddp_config,
            built_with_meta_device=built_with_meta_device,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            mixed_precision_wrapper=mixed_precision_wrapper,
        )
        configure_grad_sync(args, mimo_model, topology)

        model_list = [mimo_model]
        _model = compose_hooks(self._model_config.post_wrap_hooks)(model_list)
        if _model is not None:
            model_list = _model
        if len(model_list) != 1:
            raise ValueError(
                f"MIMO post-wrap hooks must return exactly one outer model; got {len(model_list)}"
            )
        return model_list
