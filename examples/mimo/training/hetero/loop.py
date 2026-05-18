# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Top-level orchestration for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
import torch

from examples.mimo.training.hetero.args import prepare_args
from examples.mimo.training.hetero.checkpointing import load_checkpoint, save_checkpoint
from examples.mimo.training.hetero.data import select_data_iterator, validate_data_iterator
from examples.mimo.training.hetero.distributed import print_rank_0
from examples.mimo.training.hetero.grad_sync import configure_grad_sync
from examples.mimo.training.hetero.logging import HeteroTrainingLogger
from examples.mimo.training.hetero.optimizer import build_optimizer, build_optimizer_param_scheduler
from examples.mimo.training.hetero.runtime import build_mimo_runtime
from examples.mimo.training.hetero.step import train_step
from examples.mimo.training.hetero.timeline import configure_hetero_timeline
from examples.mimo.training.hetero.topology import HeteroTopology, create_topology
from examples.mimo.utils.hetero import debug_rank, is_process_group_member
from examples.mimo.utils.model_helpers import load_nemotron_vlm_ckpt_hetero
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.timeline import (
    close_pipeline_timeline,
    flush_pipeline_timeline,
    set_pipeline_timeline_iteration,
)


def run_train_loop(args: argparse.Namespace) -> None:
    """Run heterogeneous MIMO training."""
    world_size = torch.distributed.get_world_size()
    encoder_size, llm_size = prepare_args(args, world_size)

    topology: Optional[HeteroTopology] = None
    model: Optional[MimoModel] = None
    try:
        topology = create_topology(args, encoder_size, llm_size)
        timeline_summary = configure_hetero_timeline(args, topology)
        if timeline_summary is not None:
            print_rank_0(timeline_summary)

        # Match Megatron's _set_random_seed: seed python random and numpy
        # too. energon's text_packing.random.shuffle uses the global random
        # module, so dataset-construction RNG draws would diverge otherwise.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        debug_rank("building MIMO model")
        model = build_mimo_runtime(args, topology)
        debug_rank("configuring gradient sync")
        configure_grad_sync(args, model, topology)

        debug_rank("building MIMO optimizer")
        optimizer = build_optimizer(args, model)
        opt_param_scheduler = build_optimizer_param_scheduler(args, optimizer)
        debug_rank("MIMO optimizer ready")

        debug_rank("building pipeline communicator")
        communicator = build_pipeline_communicator(model, topology)
        debug_rank("selecting data iterator")
        data_iterator = select_data_iterator(args, topology)
        validate_data_iterator(args, data_iterator, topology)
        logger = HeteroTrainingLogger(args=args, topology=topology)
        debug_rank("training setup ready")

        nemotron_ckpt = getattr(args, "load_nemotron_checkpoint", None)
        if nemotron_ckpt:
            # Pre-vlm-05 Nemotron-format ckpt: route through the converter and
            # start at iteration 0 with fresh optimizer + RNG. Mutually
            # exclusive with --load.
            if args.load:
                raise ValueError(
                    "--load and --load-nemotron-checkpoint are mutually exclusive; "
                    "pick one"
                )
            from examples.mimo.model_providers.nemotron_moe_vlm import NEMOTRON_VISION_ENCODER_KEY

            rank_in_llm = topology.language_pg is not None and is_process_group_member(
                getattr(topology.language_pg, "dp_cp", None)
            )
            rank_in_enc = topology.vision_pg is not None and is_process_group_member(
                getattr(topology.vision_pg, "dp_cp", None)
            )
            has_encoder = (
                rank_in_enc
                and topology.encoder_name in getattr(model, "modality_submodules", {})
                and model.modality_submodules[topology.encoder_name] is not None
            )
            has_language = rank_in_llm and getattr(model, "language_model", None) is not None
            load_nemotron_vlm_ckpt_hetero(
                model,
                nemotron_ckpt,
                encoder_name=topology.encoder_name,
                radio_encoder_key=NEMOTRON_VISION_ENCODER_KEY,
                has_encoder=has_encoder,
                has_language=has_language,
                language_dp_cp_group=(
                    getattr(topology.language_pg, "dp_cp", None) if has_language else None
                ),
                encoder_dp_cp_group=(
                    getattr(topology.vision_pg, "dp_cp", None) if has_encoder else None
                ),
                skip_projection=False,
            )
            # DistributedOptimizer was built before this custom load, so its
            # FP32 main-param shards still hold the model-provider init
            # weights. Refresh them from the just-loaded model params.
            optimizer.reload_model_params()
            print_rank_0(
                f"loaded Nemotron-format ckpt weights from {nemotron_ckpt}; "
                "refreshed optimizer main params; starting at iteration 0"
            )
            start_iteration = 0
        else:
            start_iteration = load_checkpoint(model, optimizer, opt_param_scheduler, args, topology)
        if start_iteration >= args.train_iters:
            print_rank_0(
                f"Resume iteration ({start_iteration}) >= --train-iters ({args.train_iters}); "
                "nothing to train."
            )
            return

        print_rank_0(
            "Starting hetero MIMO training: "
            f"world_size={world_size}, encoder_size={topology.encoder_size}, "
            f"llm_size={topology.llm_size}, "
            f"iters={start_iteration + 1}..{args.train_iters}, "
            f"dataset_provider={args.dataset_provider}"
        )

        last_saved = start_iteration
        for iteration in range(start_iteration + 1, args.train_iters + 1):
            debug_rank(f"iteration {iteration}: train step start")
            set_pipeline_timeline_iteration(iteration)
            result = train_step(
                args, model, topology, optimizer, opt_param_scheduler, communicator, data_iterator
            )
            flush_pipeline_timeline()
            logger.record_step(result)
            logger.maybe_log(iteration, optimizer, result)
            debug_rank(f"iteration {iteration}: train step complete")

            if (
                args.save
                and args.save_interval
                and iteration % args.save_interval == 0
                and iteration != args.train_iters
            ):
                save_checkpoint(iteration, model, optimizer, opt_param_scheduler, args, topology)
                last_saved = iteration

        if args.save and last_saved != args.train_iters:
            save_checkpoint(args.train_iters, model, optimizer, opt_param_scheduler, args, topology)
    finally:
        close_pipeline_timeline()
        if model is not None:
            model.destroy()
        if topology is not None:
            topology.destroy()


def build_pipeline_communicator(
    model: MimoModel, topology: HeteroTopology
) -> MultiModulePipelineCommunicator:
    """Build the MIMO pipeline communicator used by the train schedule."""
    module_output_ndim = {}
    if topology.encoder_grid is not None:
        module_output_ndim[topology.encoder_name] = 2
    return MultiModulePipelineCommunicator(
        topology.module_to_grid_map,
        topology.module_dependency_map,
        model.config,
        dim_mapping={"s": 0, "h": 2, "b": 1},
        module_output_ndim=module_output_ndim,
    )
