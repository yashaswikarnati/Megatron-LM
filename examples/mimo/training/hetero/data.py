# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Data iterator selection for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from typing import Optional

from examples.mimo.data.hetero_mock import MockVLMIterator
from examples.mimo.training.hetero.topology import (
    HeteroTopology,
    get_grid_coordinate,
    is_rank_in_grid,
)
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage


def select_data_iterator(args: argparse.Namespace, topology: HeteroTopology) -> Optional[object]:
    """Create the per-role data iterator needed by local ranks."""
    if args.dataset_provider == "mock":
        return select_mock_data_iterator(args, topology)
    if args.dataset_provider == "energon_multimodal":
        from examples.mimo.data.hetero_energon import build_energon_iterator

        return build_energon_iterator(args, topology)
    raise ValueError(f"unsupported dataset provider: {args.dataset_provider}")


def validate_data_iterator(
    args: argparse.Namespace, data_iterator, topology: HeteroTopology
) -> None:
    """Run data-provider checks that must happen outside the pipeline schedule."""
    if args.dataset_provider == "energon_multimodal":
        from examples.mimo.data.hetero_energon import validate_energon_data_alignment

        validate_energon_data_alignment(data_iterator, topology)


def select_mock_data_iterator(
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
