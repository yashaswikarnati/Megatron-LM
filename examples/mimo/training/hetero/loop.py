# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Top-level orchestration for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from examples.mimo.training.hetero.args import prepare_args
from examples.mimo.training.hetero.checkpointing import (
    load_checkpoint,
    load_vision_from_checkpoint,
    save_checkpoint,
)
from examples.mimo.training.hetero.data import select_data_iterator, validate_data_iterator
from examples.mimo.training.hetero.distributed import print_rank_0
from examples.mimo.training.hetero.grad_sync import configure_grad_sync
from examples.mimo.training.hetero.logging import HeteroTrainingLogger
from examples.mimo.training.hetero.optimizer import build_optimizer, build_optimizer_param_scheduler
from examples.mimo.training.hetero.runtime import build_mimo_runtime
from examples.mimo.training.hetero.step import train_step
from examples.mimo.training.hetero.timeline import configure_hetero_timeline
from examples.mimo.training.hetero.topology import HeteroTopology, create_topology
from examples.mimo.utils.hetero import debug_rank
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

        torch.manual_seed(args.seed)
        debug_rank("building MIMO model")
        model = build_mimo_runtime(args, topology)
        debug_rank("configuring gradient sync")
        configure_grad_sync(model, topology)

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

        start_iteration = load_checkpoint(model, optimizer, opt_param_scheduler, args, topology)
        if start_iteration == 0 and args.load_vision_from is not None:
            # Vision-only warm-start when `--load` did NOT resolve a full ckpt.
            # The load mutates radio encoder params in place (post-DDP-wrap,
            # post-optimizer build). We then call optimizer.reload_model_params()
            # to refresh the distributed optimizer's fp32 main-param mirror from
            # the just-loaded bf16 model — same pattern megatron uses after
            # in-place param mutation in upcycling (training.py:1826).
            load_vision_from_checkpoint(model, args, topology)
            if optimizer is not None and not optimizer.is_stub_optimizer:
                optimizer.reload_model_params()
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
