# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous MIMO (Nemotron6-MoE VLM) training entry point.

Builds the per-module process groups for the disjoint vision/language grids, then
drives training through the stock ``pretrain()`` path. The model is constructed by
``MimoModelBuilder`` (resolved from ``cfg.model``); the optimizer is built by
``get_megatron_optimizer`` (which dispatches on ``MimoModel``). No setup seam.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    language_model_spec,
)
from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from examples.mimo.training.args import (
    add_hetero_grid_args,
    build_module_grid_specs,
    validate_hetero_grid_args,
)
from examples.mimo.training.builder import MimoBuildConfig
from examples.mimo.training.data import add_data_args, select_data_iterator
from examples.mimo.training.distributed import initialize_distributed, shutdown_distributed
from examples.mimo.training.step import mimo_forward_step
from examples.mimo.training.topology import HeteroTopology, _encoder_module_name, create_topology
from megatron.core.config import set_experimental_flag
from megatron.core.enums import ModelType
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.training import pretrain


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the model-provider, hetero-grid, and data arg groups."""
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_data_args(parser)
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Parse/validate args; architecture flags come from the CLI (run script)."""
    args = parse_args(extra_args_provider)
    validate_hetero_grid_args(args, int(os.environ.get("WORLD_SIZE", args.world_size)))
    validate_args(args, {}, data_parallel_size_override=args.llm_dp)

    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0
    if getattr(args, "padded_vocab_size", None) is None:
        # No tokenizer in the mock run; pad the vocab to the language TP shard.
        multiple = args.make_vocab_size_divisible_by * args.llm_tp
        args.padded_vocab_size = int(math.ceil(args.vocab_size / multiple) * multiple)
    args.dataloader_type = "external"  # per-rank iterator passed through
    if getattr(args, "eval_interval", None) is None:
        args.eval_interval = args.train_iters or 1
    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set global state; the microbatch calculator keys on the language data-parallel size."""
    set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)


def _module_dependency_map(topology: HeteroTopology) -> dict:
    """Build the encoder->language dependency map the communicator's topology arg needs."""
    encoder_name = _encoder_module_name(topology)
    if encoder_name is None:
        return {MIMO_LANGUAGE_MODULE_KEY: []}
    return {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}


def build_pipeline_communicator(
    args: argparse.Namespace, topology: HeteroTopology
) -> MultiModulePipelineCommunicator:
    """Build the cross-module P2P communicator (vision encoder emits 2D activations).

    The communicator reads p2p flags (pipeline_dtype, batch_p2p_comm, ...) off the
    language ``TransformerConfig`` — the same config the language model is built from.
    """
    encoder_name = _encoder_module_name(topology)
    module_output_ndim = {}
    if encoder_name is not None:
        module_output_ndim[encoder_name] = 2

    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_config = language_model_spec(args, None, language_grid).params["config"]
    return MultiModulePipelineCommunicator(
        topology.grids,
        _module_dependency_map(topology),
        language_config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim=module_output_ndim,
    )


def main() -> None:
    """Run heterogeneous MIMO training."""
    args = _parse_and_validate()

    _setup_globals(args)
    initialize_distributed()

    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    specs = build_module_grid_specs(args, world_size, RADIO_ENCODER_MODULE_NAME)
    topology = create_topology(specs)
    communicator = build_pipeline_communicator(args, topology)
    # Independent mock iterators (each owns its generator) for train/valid/test.
    train_iter, valid_iter, test_iter = (select_data_iterator(args, topology) for _ in range(3))

    # Carrier: only ``builder`` is serialized; topology/args ride as underscore fields.
    model_cfg = MimoBuildConfig(_topology=topology, _args=args)
    cfg = pretrain_cfg_container_from_args(args, model_cfg)

    def _dataset_provider(train_val_test_num_samples):
        return train_iter, valid_iter, test_iter

    _dataset_provider.is_distributed = True

    try:
        pretrain(
            cfg,
            _dataset_provider,
            ModelType.encoder_or_decoder,
            mimo_forward_step,
            model_provider=None,
            skip_model_parallel_init=True,
            p2p_communicator=communicator,
            pg_collection=topology.pg_collection,
        )
    finally:
        topology.destroy()
        shutdown_distributed()


if __name__ == "__main__":
    main()
