# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HyperCommGrid and process-group ownership for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist

from examples.mimo.utils.hetero import debug_rank, is_process_group_member
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)

ENCODER_MODULE_NAME = "images"
EmbeddingGroupMap = dict[tuple[int, ...], tuple[dist.ProcessGroup, dist.ProcessGroup]]


@dataclass
class HeteroTopology:
    """Process groups and rank topology for one hetero MIMO run."""

    encoder_grid: HyperCommGrid
    llm_grid: HyperCommGrid
    language_pg: ProcessGroupCollection
    vision_pg: ProcessGroupCollection
    schedule_pg_collection: MultiModuleProcessGroupCollection
    embedding_groups: EmbeddingGroupMap
    encoder_size: int
    llm_size: int
    encoder_name: str = ENCODER_MODULE_NAME

    @property
    def module_to_grid_map(self) -> dict[str, HyperCommGrid]:
        """Return the MIMO module-to-grid mapping consumed by schedules and models."""
        return {self.encoder_name: self.encoder_grid, MIMO_LANGUAGE_MODULE_KEY: self.llm_grid}

    @property
    def module_dependency_map(self) -> dict[str, list[str]]:
        """Return the static encoder-to-language MIMO dependency graph."""
        return {self.encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}

    def destroy(self) -> None:
        """Destroy all process groups owned by this topology."""
        destroy_embedding_groups(self.embedding_groups)
        self.encoder_grid.destroy()
        self.llm_grid.destroy()
        BridgeCommunicator.destroy_broadcast_pgs()


def create_topology(args: argparse.Namespace, encoder_size: int, llm_size: int) -> HeteroTopology:
    """Create all rank-global process groups in one deterministic order."""
    encoder_grid = None
    llm_grid = None
    embedding_groups: Optional[EmbeddingGroupMap] = None
    try:
        debug_rank("creating encoder grid")
        encoder_grid = create_hypercomm_grid(
            offset=args.encoder_offset,
            tp=args.encoder_tp,
            cp=args.encoder_cp,
            pp=args.encoder_pp,
            dp=args.encoder_dp,
            ep=args.encoder_ep,
            expt_tp=args.encoder_expt_tp,
            expt_dp=args.encoder_expt_dp,
        )
        debug_rank("creating language grid")
        llm_grid = create_hypercomm_grid(
            offset=args.llm_offset,
            tp=args.llm_tp,
            cp=args.llm_cp,
            pp=args.llm_pp,
            dp=args.llm_dp,
            ep=args.llm_ep,
            expt_tp=args.llm_expt_tp,
            expt_dp=args.llm_expt_dp,
        )
        debug_rank("creating embedding groups")
        embedding_groups = create_embedding_groups([encoder_grid, llm_grid])
        debug_rank("embedding groups ready")

        language_pg = populate_embedding_groups(
            get_pg_collection(llm_grid), embedding_groups, is_language_model=True
        )
        vision_pg = populate_embedding_groups(
            get_pg_collection(encoder_grid), embedding_groups, is_language_model=False
        )
        schedule_pg_collection = build_schedule_pg_collection(
            ENCODER_MODULE_NAME, encoder_grid, llm_grid, vision_pg, language_pg
        )

        return HeteroTopology(
            encoder_grid=encoder_grid,
            llm_grid=llm_grid,
            language_pg=language_pg,
            vision_pg=vision_pg,
            schedule_pg_collection=schedule_pg_collection,
            embedding_groups=embedding_groups,
            encoder_size=encoder_size,
            llm_size=llm_size,
        )
    except Exception:
        if embedding_groups is not None:
            destroy_embedding_groups(embedding_groups)
        if encoder_grid is not None:
            encoder_grid.destroy()
        if llm_grid is not None:
            llm_grid.destroy()
        BridgeCommunicator.destroy_broadcast_pgs()
        raise


def create_hypercomm_grid(
    offset: int,
    tp: int,
    cp: int,
    pp: int,
    dp: int,
    ep: int,
    expt_tp: Optional[int],
    expt_dp: Optional[int],
) -> HyperCommGrid:
    """Create a dense grid plus expert layout and required process groups."""
    expt_tp = tp if expt_tp is None else expt_tp
    module_world_size = tp * cp * pp * dp
    expert_model_size = expt_tp * ep * pp
    if module_world_size % expert_model_size != 0:
        raise ValueError(
            f"module_world_size ({module_world_size}) must be divisible by "
            f"expt_tp*ep*pp ({expert_model_size})"
        )
    if expt_dp is None:
        expt_dp = module_world_size // expert_model_size
    if expt_tp * ep * expt_dp * pp != module_world_size:
        raise ValueError(
            f"expt_tp*ep*expt_dp*pp ({expt_tp * ep * expt_dp * pp}) must equal "
            f"module_world_size ({module_world_size})"
        )

    grid = HyperCommGrid(
        shape=[tp, cp, dp, pp],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.register_layout("expert", [expt_tp, ep, expt_dp, pp], ["expt_tp", "ep", "expt_dp", "pp"])

    try:
        for dims in (
            ["tp"],
            ["cp"],
            ["pp"],
            ["dp"],
            ["dp", "cp"],
            ["tp", "cp"],
            ["ep"],
            ["expt_tp"],
            ["expt_dp"],
            ["tp", "pp"],
            ["tp", "cp", "dp"],
            ["tp", "cp", "pp", "dp"],
            ["expt_tp", "ep"],
            ["expt_tp", "ep", "pp"],
        ):
            grid.create_pg(dims)
    except Exception:
        grid.destroy()
        raise

    return grid


def get_pg_collection(grid: HyperCommGrid) -> ProcessGroupCollection:
    """Build a ProcessGroupCollection from a populated HyperCommGrid."""
    return ProcessGroupCollection.from_hyper_comm_grid(
        grid,
        required_pgs=[
            "tp",
            "cp",
            "pp",
            "dp",
            "dp_cp",
            "tp_cp",
            "mp",
            "tp_dp_cp",
            "ep",
            "expt_tp",
            "expt_dp",
            "tp_ep",
            "tp_ep_pp",
            "intra_dist_opt",
        ],
    )


def create_embedding_groups(grids: list[HyperCommGrid]) -> EmbeddingGroupMap:
    """Create PP-derived embedding groups and return handles by PP rank tuple.

    These groups must be created collectively by all ranks in one deterministic order before
    module-local ProcessGroupCollections are populated.
    """
    embedding_groups: EmbeddingGroupMap = {}
    pp_rank_sets: list[tuple[int, ...]] = []
    seen_pp_rank_sets = set()
    for grid in sorted(grids, key=lambda candidate: (candidate.rank_offset, candidate.size)):
        for pp_ranks in grid.get_rank_enum("pp"):
            pp_rank_tuple = tuple(pp_ranks)
            if pp_rank_tuple in seen_pp_rank_sets:
                continue
            pp_rank_sets.append(pp_rank_tuple)
            seen_pp_rank_sets.add(pp_rank_tuple)

    try:
        for pp_ranks in pp_rank_sets:
            pos_embd_ranks = [pp_ranks[0]]
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            pos_embd_pg = None
            embd_pg = None
            try:
                pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
                embd_pg = dist.new_group(ranks=embd_ranks)
                embedding_groups[pp_ranks] = (pos_embd_pg, embd_pg)
            except Exception:
                destroy_process_group_if_member(pos_embd_pg)
                destroy_process_group_if_member(embd_pg)
                raise
    except Exception:
        destroy_embedding_groups(embedding_groups)
        raise

    return embedding_groups


def destroy_embedding_groups(embedding_groups: EmbeddingGroupMap) -> None:
    """Destroy embedding process groups returned by create_embedding_groups."""
    destroyed_embedding_pgs = set()
    for pos_embd_pg, embd_pg in embedding_groups.values():
        for pg in (pos_embd_pg, embd_pg):
            if id(pg) in destroyed_embedding_pgs:
                continue
            destroy_process_group_if_member(pg)
            destroyed_embedding_pgs.add(id(pg))
    embedding_groups.clear()


def populate_embedding_groups(
    pg_collection: ProcessGroupCollection,
    embedding_groups: EmbeddingGroupMap,
    is_language_model: bool = False,
) -> ProcessGroupCollection:
    """Populate Megatron's required embedding fields on a ProcessGroupCollection."""
    if not is_process_group_member(getattr(pg_collection, "pp", None)):
        return pg_collection

    pp_ranks = tuple(dist.get_process_group_ranks(pg_collection.pp))
    pos_embd_pg, embd_pg = embedding_groups[pp_ranks]

    pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
    if is_language_model:
        pg_collection.embd = (
            embd_pg
            if is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp)
            else None
        )
    else:
        pg_collection.embd = None

    return pg_collection


def build_schedule_pg_collection(
    encoder_name: str,
    encoder_grid: HyperCommGrid,
    llm_grid: HyperCommGrid,
    vision_pg: ProcessGroupCollection,
    language_pg: ProcessGroupCollection,
) -> MultiModuleProcessGroupCollection:
    """Build the schedule-facing process group collection for this rank."""
    module_pgs = {}
    language_model_module_name = None
    if is_rank_in_grid(encoder_grid):
        module_pgs[encoder_name] = vision_pg
    if is_rank_in_grid(llm_grid):
        module_pgs[MIMO_LANGUAGE_MODULE_KEY] = language_pg
        language_model_module_name = MIMO_LANGUAGE_MODULE_KEY

    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs, language_model_module_name=language_model_module_name
    )


def destroy_process_group_if_member(pg: Optional[dist.ProcessGroup]) -> None:
    """Destroy pg when this rank owns a process-group handle."""
    if is_process_group_member(pg):
        dist.destroy_process_group(pg)


def is_rank_in_grid(grid: HyperCommGrid) -> bool:
    """Return whether this global rank is inside a grid's rank span."""
    rank = dist.get_rank()
    return grid.rank_offset <= rank < grid.rank_offset + grid.size


def get_grid_coordinate(grid: HyperCommGrid, dim: str) -> int:
    """Return this rank's coordinate for a base-layout dimension."""
    if not is_rank_in_grid(grid):
        return 0

    local_rank = dist.get_rank() - grid.rank_offset
    coordinates = {}
    for dim_name, dim_size in zip(grid.dim_names, grid.shape):
        coordinates[dim_name] = local_rank % dim_size
        local_rank //= dim_size
    return coordinates[dim]
