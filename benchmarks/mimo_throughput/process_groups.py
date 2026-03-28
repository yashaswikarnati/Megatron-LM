# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HyperCommGrid and ProcessGroupCollection lifecycle management for MIMO benchmarks.

Encapsulates the grid creation, PG creation, embedding group management, and teardown
patterns used by colocated and homogeneous MIMO training configurations.
"""

from typing import List

import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection


class ProcessGroupManager:
    """Manages HyperCommGrid and ProcessGroupCollection lifecycle.

    Tracks all created grids and embedding PG caches so they can be torn down
    cleanly between benchmark experiments.
    """

    def __init__(self):
        self._grids: List[HyperCommGrid] = []
        self._embedding_pg_cache: dict = {}

    def create_grid(self, tp: int, dp: int, pp: int = 1, offset: int = 0) -> HyperCommGrid:
        """Create a HyperCommGrid with specified parallelism.

        Grid shape: [tp, cp=1, pp, dp, ep=1, expt_dp=1]

        Creates ALL PGs needed by DDP, optimizer, and schedule:
        - Core: tp, cp, pp, dp
        - Composite: dp_cp, ep, expt_dp
        - Optimizer: tp_pp (mp), tp_ep_pp, dp_ep, tp_cp_ep_pp_dp (intra_dist_opt)

        Args:
            tp: Tensor parallel degree.
            dp: Data parallel degree.
            pp: Pipeline parallel degree (default 1).
            offset: Rank offset for the grid (default 0).

        Returns:
            Configured HyperCommGrid with all required PGs created.
        """
        grid = HyperCommGrid(
            shape=[tp, 1, pp, dp, 1, 1],  # [tp, cp, pp, dp, ep, expt_dp]
            dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
            rank_offset=offset,
            backend="nccl",
        )

        # Core single-dimension groups
        for dim in ["tp", "cp", "pp", "dp", "ep", "expt_dp"]:
            grid.create_pg([dim])

        # Composite groups for DDP and data loading
        grid.create_pg(["dp", "cp"])

        # Optimizer-required composite groups (see _get_pg_collection_for_optimizer)
        grid.create_pg(["tp", "pp"])  # mp
        grid.create_pg(["tp", "ep", "pp"])  # tp_ep_pp
        grid.create_pg(["dp", "ep"])  # expt_dp for optimizer
        grid.create_pg(["tp", "cp", "ep", "pp", "dp"])  # intra_dist_opt

        self._grids.append(grid)
        return grid

    def create_embedding_groups(self, grids: List[HyperCommGrid]):
        """Create embedding PGs for all grids upfront.

        dist.new_group is a collective -- ALL ranks must call it, even non-members.
        Must be called BEFORE get_pg_collection() so the cached groups are available.

        Args:
            grids: List of HyperCommGrids to create embedding groups for.
        """
        for grid in grids:
            pp_group = grid.get_pg("pp")
            if not pp_group:
                continue

            pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
            cache_key = tuple(pp_ranks)

            if cache_key not in self._embedding_pg_cache:
                pos_embd_ranks = [pp_ranks[0]]
                embd_ranks = [pp_ranks[0]]
                if pp_ranks[-1] != pp_ranks[0]:
                    embd_ranks.append(pp_ranks[-1])
                self._embedding_pg_cache[cache_key] = (
                    dist.new_group(ranks=pos_embd_ranks),
                    dist.new_group(ranks=embd_ranks),
                )

    def get_pg_collection(
        self, grid: HyperCommGrid, is_language_model: bool = False
    ) -> ProcessGroupCollection:
        """Build ProcessGroupCollection from grid with embedding groups.

        Sets the core PGs (tp, cp, pp, ep, dp, dp_cp, expt_dp) and conditionally
        adds pos_embd and embd groups from the cache.

        Args:
            grid: HyperCommGrid to extract PGs from.
            is_language_model: If True, set embd PG on first/last PP stages.

        Returns:
            Fully configured ProcessGroupCollection.
        """
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = grid.get_pg("tp")
        pg_collection.cp = grid.get_pg("cp")
        pg_collection.pp = grid.get_pg("pp")
        pg_collection.ep = grid.get_pg("ep")
        pg_collection.dp = grid.get_pg("dp")
        pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
        pg_collection.expt_dp = grid.get_pg("expt_dp")

        # Add embedding groups from cache
        pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
        cache_key = tuple(pp_ranks)

        if cache_key in self._embedding_pg_cache:
            pos_embd_pg, embd_pg = self._embedding_pg_cache[cache_key]
            pg_collection.pos_embd = (
                pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
            )
            pg_collection.embd = (
                embd_pg
                if is_language_model
                and (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
                else None
            )

        return pg_collection

    def destroy_all(self):
        """Destroy all tracked grids and cached PGs."""
        for grid in self._grids:
            grid.destroy()
        self._grids.clear()
        for pos_embd_pg, embd_pg in self._embedding_pg_cache.values():
            if pos_embd_pg is not None:
                dist.destroy_process_group(pos_embd_pg)
            if embd_pg is not None:
                dist.destroy_process_group(embd_pg)
        self._embedding_pg_cache.clear()
        BridgeCommunicator.destroy_broadcast_pgs()
