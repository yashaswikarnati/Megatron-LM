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
LanguageEmbeddingGroups = dict[tuple[int, ...], Optional[dist.ProcessGroup]]


@dataclass
class HeteroTopology:
    """Process groups and rank topology for one hetero MIMO run."""

    encoder_grid: Optional[HyperCommGrid]
    llm_grid: HyperCommGrid
    language_pg: ProcessGroupCollection
    vision_pg: Optional[ProcessGroupCollection]
    schedule_pg_collection: MultiModuleProcessGroupCollection
    language_embedding_groups: LanguageEmbeddingGroups
    encoder_size: int
    llm_size: int
    encoder_name: str = ENCODER_MODULE_NAME

    @property
    def module_to_grid_map(self) -> dict[str, HyperCommGrid]:
        """Return the MIMO module-to-grid mapping consumed by schedules and models."""
        if self.encoder_grid is None:
            return {MIMO_LANGUAGE_MODULE_KEY: self.llm_grid}
        return {self.encoder_name: self.encoder_grid, MIMO_LANGUAGE_MODULE_KEY: self.llm_grid}

    @property
    def module_dependency_map(self) -> dict[str, list[str]]:
        """Return the static encoder-to-language MIMO dependency graph."""
        if self.encoder_grid is None:
            return {MIMO_LANGUAGE_MODULE_KEY: []}
        return {self.encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}

    def destroy(self) -> None:
        """Destroy all process groups owned by this topology."""
        destroy_embedding_groups(self.language_embedding_groups)
        if self.encoder_grid is not None:
            self.encoder_grid.destroy()
        self.llm_grid.destroy()
        BridgeCommunicator.destroy_broadcast_pgs()


def create_topology(args: argparse.Namespace, encoder_size: int, llm_size: int) -> HeteroTopology:
    """Create all rank-global process groups in one deterministic order."""
    encoder_grid = None
    llm_grid = None
    language_embedding_groups: Optional[LanguageEmbeddingGroups] = None
    try:
        if not args.llm_only:
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
        debug_rank("creating language embedding groups")
        language_embedding_groups = create_language_embedding_groups(llm_grid)
        debug_rank("language embedding groups ready")

        language_pg = populate_language_embedding_groups(
            get_pg_collection(llm_grid), language_embedding_groups
        )
        vision_pg = (
            None
            if encoder_grid is None
            else clear_embedding_groups(get_pg_collection(encoder_grid))
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
            language_embedding_groups=language_embedding_groups,
            encoder_size=encoder_size,
            llm_size=llm_size,
        )
    except Exception:
        if language_embedding_groups is not None:
            destroy_embedding_groups(language_embedding_groups)
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
    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.intra_dp_cp = pg.dp_cp
    pg.tp_cp = grid.get_pg(["tp", "cp"])
    pg.mp = grid.get_pg(["tp", "pp"])
    pg.tp_dp_cp = grid.get_pg(["tp", "dp", "cp"])
    pg.ep = grid.get_pg("ep")
    pg.expt_tp = grid.get_pg("expt_tp")
    pg.expt_dp = grid.get_pg("expt_dp")
    pg.intra_expt_dp = pg.expt_dp
    pg.tp_ep = grid.get_pg(["expt_tp", "ep"])
    pg.tp_ep_pp = grid.get_pg(["expt_tp", "ep", "pp"])
    pg.intra_dist_opt = grid.get_pg(["tp", "cp", "dp", "pp"])
    return pg


def create_language_embedding_groups(grid: HyperCommGrid) -> LanguageEmbeddingGroups:
    """Create language-model embedding groups keyed by PP rank tuple.

    A language grid has one PP group per TP/CP/DP lane, so the rank tuple is the stable key used
    to attach the matching first/last-stage embedding group to each ProcessGroupCollection.
    """
    embedding_groups: LanguageEmbeddingGroups = {}

    try:
        for pp_ranks in grid.get_rank_enum("pp"):
            pp_rank_tuple = tuple(pp_ranks)
            if pp_rank_tuple[0] == pp_rank_tuple[-1]:
                embedding_groups[pp_rank_tuple] = None
                continue

            embd_pg = None
            try:
                embd_pg = dist.new_group(ranks=[pp_rank_tuple[0], pp_rank_tuple[-1]])
                embedding_groups[pp_rank_tuple] = embd_pg
            except Exception:
                destroy_process_group_if_member(embd_pg)
                raise
    except Exception:
        destroy_embedding_groups(embedding_groups)
        raise

    return embedding_groups


def destroy_embedding_groups(embedding_groups: LanguageEmbeddingGroups) -> None:
    """Destroy embedding process groups returned by create_language_embedding_groups."""
    destroyed_embedding_pgs = set()
    for embd_pg in embedding_groups.values():
        if embd_pg is None or id(embd_pg) in destroyed_embedding_pgs:
            continue
        destroy_process_group_if_member(embd_pg)
        destroyed_embedding_pgs.add(id(embd_pg))
    embedding_groups.clear()


def populate_language_embedding_groups(
    pg_collection: ProcessGroupCollection,
    embedding_groups: LanguageEmbeddingGroups,
) -> ProcessGroupCollection:
    """Populate language embedding fields required by finalize_model_grads."""
    pg_collection.pos_embd = None
    pg_collection.embd = None
    if not is_process_group_member(getattr(pg_collection, "pp", None)):
        return pg_collection

    pp_ranks = tuple(dist.get_process_group_ranks(pg_collection.pp))
    if is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp):
        pg_collection.embd = embedding_groups[pp_ranks]

    return pg_collection


def clear_embedding_groups(pg_collection: ProcessGroupCollection) -> ProcessGroupCollection:
    """Populate embedding fields with None for modules that do not share embeddings."""
    pg_collection.pos_embd = None
    pg_collection.embd = None
    return pg_collection


def build_schedule_pg_collection(
    encoder_name: str,
    encoder_grid: Optional[HyperCommGrid],
    llm_grid: HyperCommGrid,
    vision_pg: Optional[ProcessGroupCollection],
    language_pg: ProcessGroupCollection,
) -> MultiModuleProcessGroupCollection:
    """Build the schedule-facing process group collection for this rank."""
    module_pgs = {}
    language_model_module_name = None
    if encoder_grid is not None and is_rank_in_grid(encoder_grid):
        assert vision_pg is not None
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
