# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Top-level orchestration for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from examples.mimo.data.hetero_mock import MockVLMIterator
from examples.mimo.training.hetero.args import prepare_args
from examples.mimo.training.hetero.distributed import print_rank_0
from examples.mimo.training.hetero.logging import HeteroTrainingLogger
from examples.mimo.training.hetero.runtime import HeteroRuntime, build_mimo_runtime
from examples.mimo.training.hetero.scheduler import build_optimizer_param_scheduler
from examples.mimo.training.hetero.step import train_step, wire_training_hooks
from examples.mimo.training.hetero.topology import (
    HeteroTopology,
    create_topology,
    get_grid_coordinate,
    is_rank_in_grid,
)
from examples.mimo.utils.hetero import debug_rank


def run_train_loop(args: argparse.Namespace) -> None:
    """Run mock-data heterogeneous MIMO training."""
    world_size = torch.distributed.get_world_size()
    encoder_size, llm_size = prepare_args(args, world_size)

    topology: Optional[HeteroTopology] = None
    runtime: Optional[HeteroRuntime] = None
    try:
        topology = create_topology(args, encoder_size, llm_size)

        torch.manual_seed(args.seed)
        debug_rank("building MIMO model")
        runtime = build_mimo_runtime(args, topology)
        debug_rank("wiring training hooks")
        wire_training_hooks(runtime, topology)

        debug_rank("building MIMO optimizer")
        optimizer = build_optimizer(args, runtime, topology)
        opt_param_scheduler = build_optimizer_param_scheduler(args, optimizer)
        debug_rank("MIMO optimizer ready")

        debug_rank("building pipeline communicator")
        communicator = MultiModulePipelineCommunicator(
            topology.module_to_grid_map,
            topology.module_dependency_map,
            runtime.model.config,
            dim_mapping={"s": 0, "h": 2, "b": 1},
            module_output_ndim={topology.encoder_name: 2},
        )
        debug_rank("selecting data iterator")
        data_iterator = select_data_iterator(args, topology)
        logger = HeteroTrainingLogger(args=args, topology=topology)
        debug_rank("training setup ready")

        print_rank_0(
            "Starting hetero MIMO mock training: "
            f"world_size={world_size}, encoder_size={topology.encoder_size}, "
            f"llm_size={topology.llm_size}, train_iters={args.train_iters}"
        )

        for iteration in range(1, args.train_iters + 1):
            debug_rank(f"iteration {iteration}: train step start")
            result = train_step(
                args, runtime, topology, optimizer, opt_param_scheduler, communicator, data_iterator
            )
            logger.record_step(result)
            logger.maybe_log(iteration, optimizer, result)
            debug_rank(f"iteration {iteration}: train step complete")
    finally:
        if runtime is not None:
            runtime.destroy()
        if topology is not None:
            topology.destroy()


def build_optimizer(args: argparse.Namespace, runtime: HeteroRuntime, topology: HeteroTopology):
    """Build the MIMO optimizer for active hetero module optimizers."""
    return get_mimo_optimizer(
        runtime.model,
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
        stats_group=topology.optimizer_stats_group,
    )


def select_data_iterator(
    args: argparse.Namespace, topology: HeteroTopology
) -> Optional[MockVLMIterator]:
    """Create the per-role mock-data iterator needed by local ranks."""
    llm_mbs = args.micro_batch_size
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")
    encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp

    encoder_grid = topology.encoder_grid
    llm_grid = topology.llm_grid
    encoder_needs_data = is_rank_in_grid(encoder_grid) and is_pp_first_stage(
        encoder_grid.get_pg("pp")
    )
    llm_needs_data = is_rank_in_grid(llm_grid) and (
        is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp"))
    )

    if encoder_needs_data and not llm_needs_data:
        return MockVLMIterator(
            args,
            encoder_mbs,
            topology.encoder_name,
            get_mock_data_seed(args, encoder_grid, module_seed_offset=0),
        )
    if llm_needs_data and not encoder_needs_data:
        return MockVLMIterator(
            args,
            llm_mbs,
            topology.encoder_name,
            get_mock_data_seed(args, llm_grid, module_seed_offset=100_000),
        )
    if encoder_needs_data and llm_needs_data:
        return MockVLMIterator(
            args,
            llm_mbs,
            topology.encoder_name,
            get_mock_data_seed(args, llm_grid, module_seed_offset=100_000),
        )
    return None


def get_mock_data_seed(args: argparse.Namespace, grid, module_seed_offset: int) -> int:
    """Seed mock data by data-parallel lane so PP/TP stages see coherent batches."""
    dp_lane = get_grid_coordinate(grid, "dp") if "dp" in grid.dim_names else 0
    return args.seed + module_seed_offset + dp_lane
