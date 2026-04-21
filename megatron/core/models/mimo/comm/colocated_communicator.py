# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class SliceInfo:
    """Batch dimension slice information for a rank's data partition."""

    start: int
    size: int


class BridgeDirection(str, Enum):
    """Which side of the bridge scales up, if any.

    ``FAN_IN`` — src has more DP replicas than dest; forward all-gathers
    src outputs along the batch dim, backward narrows the sibling dest
    gradient down to this src rank's slot.

    ``FAN_OUT`` — dest has more DP replicas; forward narrows, backward
    all-gathers across the sibling dest DP ranks (the adjoint of narrow
    is not zero-pad-and-scatter because every dest rank consumes a
    different slice of the same src activation).

    ``EQUAL`` — matching DP; the bridge is a pure passthrough.
    """

    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    EQUAL = "equal"


class ColocatedBridgeCommunicator:
    """Bridges tensors between colocated modules with different TP/DP layouts.

    Default ``dim_mapping`` assumes 3D ``(b, s, h)``. Callers bridging
    ``MimoModel``'s pre-flattened ``(s*b, h)`` encoder output should pass
    ``dim_mapping={'b': 0, 'h': 1}``; this relies on a uniform token count per
    sample so dim 0 divides evenly by the DP scale.

    Precondition: the input must be TP-replicated across the src TP group —
    i.e. all TP ranks inside a src DP replica hold the same tensor on the
    batch dim. The bridge never gathers along TP; violating this silently
    produces wrong results.
    """

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        src_module_name: str = "src",
        dest_module_name: str = "dest",
        dim_mapping: Optional[Dict[str, int]] = None,
    ):
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.dim_mapping = dim_mapping or {'b': 0, 's': 1, 'h': 2}
        self.current_rank = dist.get_rank()

        self._validate_grids()
        self._extract_parallelism_info()
        self._build_rank_mappings()

        # At most one direction is active; fan-in and fan-out are mutually
        # exclusive (one of ``src_dp / dest_dp`` is >1, the other is 1).
        # Equal DP uses no collective at all. Unify behind a single
        # ``gather_pg`` + ``direction`` + ``scale`` rather than a fan-in
        # and fan-out pair of attributes.
        self.gather_pg: Optional[dist.ProcessGroup] = None
        self.gather_group_ranks: List[List[int]] = []

        if self.src_dp_size > self.dest_dp_size:
            self.direction = BridgeDirection.FAN_IN
            self.scale = self.src_dp_size // self.dest_dp_size
            self.gather_group_ranks = self._build_gather_groups(
                iter_size=self.dest_dp_size,
                sibling_tp_size=self.src_tp_size,
                scale=self.scale,
                rank_to_pos=self.rank_to_src_pos,
            )
            self.gather_pg, _ = dist.new_subgroups_by_enumeration(
                self.gather_group_ranks, backend='nccl'
            )
        elif self.dest_dp_size > self.src_dp_size:
            self.direction = BridgeDirection.FAN_OUT
            self.scale = self.dest_dp_size // self.src_dp_size
            # Fan-out gather groups must be split by dest CP level: every
            # world rank must land in exactly one subgroup, so a single
            # pooled group per (src_dp, dest_tp) would orphan cp>0 ranks.
            # With dest_cp_size == 1 this collapses to the original
            # (src_dp, dest_tp) product.
            self.gather_group_ranks = self._build_fan_out_gather_groups(
                iter_size=self.src_dp_size,
                sibling_tp_size=self.dest_tp_size,
                scale=self.scale,
                cp_size=self.dest_cp_size,
                rank_to_coords=self.rank_to_dest_coords,
            )
            self.gather_pg, _ = dist.new_subgroups_by_enumeration(
                self.gather_group_ranks, backend='nccl'
            )
        else:
            self.direction = BridgeDirection.EQUAL
            self.scale = 1

        logging.info(
            f"[Rank {self.current_rank}] ColocatedBridgeCommunicator: "
            f"{src_module_name}({self.src_tp_size}TP/{self.src_dp_size}DP) -> "
            f"{dest_module_name}({self.dest_tp_size}TP/{self.dest_dp_size}DP), "
            f"direction={self.direction.value}, scale={self.scale}"
        )

    def _validate_grids(self):
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            for required in ('tp', 'dp'):
                if required not in grid.dim_names:
                    raise ValueError(
                        f"{name} grid must have '{required}' dimension, "
                        f"got dim_names={grid.dim_names}"
                    )

        if self.src_grid.size != self.dest_grid.size:
            raise ValueError(
                f"Grids must span same number of ranks: "
                f"src={self.src_grid.size}, dest={self.dest_grid.size}"
            )

        if self.src_grid.rank_offset != self.dest_grid.rank_offset:
            raise ValueError(
                f"Grids must have same rank offset: "
                f"src={self.src_grid.rank_offset}, dest={self.dest_grid.rank_offset}"
            )

        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            if 'pp' in grid.dim_names:
                pp_size = grid.shape[grid.dim_names.index('pp')]
                if pp_size != 1:
                    raise ValueError(
                        f"{name} PP must be 1 for ColocatedBridgeCommunicator, got {pp_size}"
                    )

        # Source (encoder) must have CP=1. Dest (LLM) may have CP>1 — the LLM
        # shards sequence across CP ranks via PartitionAdapter after receiving
        # the bridge's full-sequence output. The bridge's backward path
        # reduces partial-sequence gradients across dest CP siblings before
        # returning the full-sequence gradient to the encoder.
        if 'cp' in self.src_grid.dim_names:
            src_cp_size = self.src_grid.shape[self.src_grid.dim_names.index('cp')]
            if src_cp_size != 1:
                raise ValueError(
                    f"Source CP must be 1 for ColocatedBridgeCommunicator, got {src_cp_size}"
                )

        # _build_rank_mappings assumes that _gen_rank_enum(['tp']) yields cp
        # varying fastest for fixed dp — true only when cp appears BEFORE dp
        # in dim_names (reversed, cp becomes an inner loop around dp). If the
        # caller reverses them, dp_idx advances at the wrong cp level and
        # rank_to_dest_coords is silently wrong. Guard explicitly.
        if 'cp' in self.dest_grid.dim_names:
            dim_names = self.dest_grid.dim_names
            if dim_names.index('cp') > dim_names.index('dp'):
                raise ValueError(
                    f"dest_grid dim_names must have 'cp' before 'dp' "
                    f"(e.g. ['tp','cp','pp','dp']); got {dim_names}"
                )

        src_dp = self.src_grid.shape[self.src_grid.dim_names.index('dp')]
        dest_dp = self.dest_grid.shape[self.dest_grid.dim_names.index('dp')]
        if src_dp % dest_dp != 0 and dest_dp % src_dp != 0:
            raise ValueError(
                f"DP sizes must be evenly divisible: src_dp={src_dp}, dest_dp={dest_dp}"
            )

    def _extract_parallelism_info(self):
        self.src_tp_size = self.src_grid.shape[self.src_grid.dim_names.index('tp')]
        self.src_dp_size = self.src_grid.shape[self.src_grid.dim_names.index('dp')]
        self.dest_tp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('tp')]
        self.dest_dp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('dp')]
        if 'cp' in self.dest_grid.dim_names:
            self.dest_cp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('cp')]
        else:
            self.dest_cp_size = 1
        # Reuse the existing CP group from dest_grid (caller creates it via
        # grid.create_pg(['cp'])). None when dest CP=1. Used in backward to
        # reduce sequence-sharded gradients across CP siblings before returning
        # to the encoder — see the backward docstring for the math.
        self.dest_cp_pg: Optional[dist.ProcessGroup] = (
            self.dest_grid.get_pg('cp') if self.dest_cp_size > 1 else None
        )

    @staticmethod
    def _get_rank_dim_coord(rank: int, grid: HyperCommGrid, dim_name: str) -> int:
        """Extract a rank's coordinate for a specific grid dimension."""
        dim_idx = grid.dim_names.index(dim_name)
        temp = rank - grid.rank_offset
        for i in range(dim_idx):
            temp //= grid.shape[i]
        return temp % grid.shape[dim_idx]

    def _build_rank_mappings(self):
        self.rank_to_src_pos: Dict[int, Tuple[int, int]] = {}
        # rank_to_dest_pos: canonical (cp_idx=0, pp_idx=0) rank per (dp, tp)
        # slot — preserves the one-entry-per-slot contract downstream group
        # construction depends on.
        self.rank_to_dest_pos: Dict[int, Tuple[int, int]] = {}
        # rank_to_dest_coords: every dest rank's full (dp_idx, tp_idx, cp_idx)
        # for PP stage 0. Used to build per-CP-level fan-out groups and to
        # drive the intra-CP gradient reduction in backward.
        self.rank_to_dest_coords: Dict[int, Tuple[int, int, int]] = {}

        src_tp_groups = self.src_grid.get_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(src_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_src_pos[rank] = (dp_idx, tp_idx)

        # Dest iteration: get_rank_enum(['tp']) returns dp*pp*cp tp-groups.
        # We advance dp_idx only after the final cp level of each dp; every
        # (cp, pp=0) rank still records its coords so fan-out backward can
        # locate CP siblings.
        dest_has_pp = 'pp' in self.dest_grid.dim_names
        dest_has_cp = 'cp' in self.dest_grid.dim_names
        dest_tp_groups = self.dest_grid.get_rank_enum(['tp'])
        dp_idx = 0
        for tp_group in dest_tp_groups:
            if dest_has_pp:
                pp_coord = self._get_rank_dim_coord(tp_group[0], self.dest_grid, 'pp')
                if pp_coord != 0:
                    continue
            cp_coord = (
                self._get_rank_dim_coord(tp_group[0], self.dest_grid, 'cp') if dest_has_cp else 0
            )
            # get_rank_enum yields cp varying fastest for fixed dp (with
            # pp=0 filtered). All cp levels of one dp share dp_idx; we
            # advance dp_idx only after the final cp level of that dp.
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_dest_coords[rank] = (dp_idx, tp_idx, cp_coord)
                if cp_coord == 0:
                    self.rank_to_dest_pos[rank] = (dp_idx, tp_idx)
            if cp_coord == self.dest_cp_size - 1:
                dp_idx += 1

    @staticmethod
    def _build_gather_groups(
        iter_size: int,
        sibling_tp_size: int,
        scale: int,
        rank_to_pos: Dict[int, Tuple[int, int]],
    ) -> List[List[int]]:
        """Build ``iter_size * sibling_tp_size`` gather groups of ``scale`` ranks.

        For each slot on the "iterating" side and each TP shard on the
        sibling side, collect the ``scale`` sibling ranks whose DP indices
        map into that slot. Append order equals group-local-rank order,
        which ``all_gather_into_tensor`` uses to concatenate outputs — do
        not sort.
        """
        groups: List[List[int]] = []
        for iter_idx in range(iter_size):
            sibling_dp_indices = range(iter_idx * scale, (iter_idx + 1) * scale)
            for sibling_tp_idx in range(sibling_tp_size):
                group_ranks = []
                for sibling_dp_idx in sibling_dp_indices:
                    for rank, (dp, tp) in rank_to_pos.items():
                        if dp == sibling_dp_idx and tp == sibling_tp_idx:
                            group_ranks.append(rank)
                            break
                groups.append(group_ranks)
        return groups

    @staticmethod
    def _build_fan_out_gather_groups(
        iter_size: int,
        sibling_tp_size: int,
        scale: int,
        cp_size: int,
        rank_to_coords: Dict[int, Tuple[int, int, int]],
    ) -> List[List[int]]:
        """Build fan-out gather groups split per (src_dp, dest_tp, dest_cp).

        Splitting by ``cp_idx`` is required because
        ``new_subgroups_by_enumeration`` demands every world rank land in
        exactly one subgroup — a single pooled group per (src_dp, dest_tp)
        would leave cp>0 ranks orphaned. After the CP reduction in backward,
        each cp-level's all-gather produces the same full-batch gradient.
        When ``cp_size == 1`` this degenerates to the original
        (src_dp, dest_tp) product.
        """
        coords_to_rank: Dict[Tuple[int, int, int], int] = {
            coords: rank for rank, coords in rank_to_coords.items()
        }
        groups: List[List[int]] = []
        for iter_idx in range(iter_size):
            sibling_dp_indices = range(iter_idx * scale, (iter_idx + 1) * scale)
            for sibling_tp_idx in range(sibling_tp_size):
                for cp_idx in range(cp_size):
                    group_ranks = [
                        coords_to_rank[(sibling_dp_idx, sibling_tp_idx, cp_idx)]
                        for sibling_dp_idx in sibling_dp_indices
                    ]
                    groups.append(group_ranks)
        return groups

    def is_fan_in(self) -> bool:
        """True if src DP > dest DP (forward all-gathers)."""
        return self.direction is BridgeDirection.FAN_IN

    def is_fan_out(self) -> bool:
        """True if src DP < dest DP (forward narrows)."""
        return self.direction is BridgeDirection.FAN_OUT

    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Compute this rank's slice of ``batch_size`` on the narrowing side.

        For FAN_OUT this is the forward narrow; for FAN_IN it is the
        backward narrow against the post-gather batch. EQUAL returns the
        identity slice.

        Raises ``ValueError`` if ``batch_size`` is not divisible by ``scale``.
        """
        if self.direction is BridgeDirection.EQUAL:
            return SliceInfo(start=0, size=batch_size)
        self._check_divisible(batch_size)
        if self.direction is BridgeDirection.FAN_OUT:
            # rank_to_dest_pos only tracks cp=0 canonical slots; CP>0 dest
            # ranks must slice the same batch slot as their cp=0 sibling so
            # the intra-CP all_reduce in backward sees matching shapes.
            # rank_to_dest_coords has an entry per (dp, tp, cp).
            dp_idx = self.rank_to_dest_coords[self.current_rank][0]
        else:  # FAN_IN
            dp_idx = self.rank_to_src_pos[self.current_rank][0]
        slot = dp_idx % self.scale
        slice_size = batch_size // self.scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def _check_divisible(self, batch_size: int) -> None:
        if batch_size % self.scale != 0:
            raise ValueError(
                f"ColocatedBridgeCommunicator: batch dim size {batch_size} is "
                f"not divisible by {self.direction.value} scale={self.scale}."
            )

    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform ``tensor`` from src TP/DP layout to dest TP/DP layout.

        Raises ``ValueError`` when FAN_OUT and the batch dim is not
        divisible by ``scale``; FAN_IN only slices on the backward pass
        and re-checks via ``get_slice_info`` there.
        """
        if self.direction is BridgeDirection.FAN_OUT:
            self._check_divisible(tensor.shape[self.dim_mapping['b']])
        return _ColocatedCommunicate.apply(tensor, self)

    def destroy(self) -> None:
        """Release the NCCL subgroup created by this communicator.

        NCCL caps concurrent communicators; long-lived or repeated
        construction leaks PGs without this call.
        """
        if self.gather_pg is not None:
            dist.destroy_process_group(self.gather_pg)
            self.gather_pg = None


class _ColocatedCommunicate(torch.autograd.Function):
    """Autograd function for colocated communication with correct backward pass."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, comm: ColocatedBridgeCommunicator) -> torch.Tensor:
        ctx.comm = comm
        ctx.batch_dim = comm.dim_mapping['b']

        if comm.direction is BridgeDirection.FAN_OUT:
            # Narrow this rank's slice out of the full src batch.
            slice_info = comm.get_slice_info(tensor.shape[ctx.batch_dim])
            return tensor.narrow(ctx.batch_dim, slice_info.start, slice_info.size).contiguous()

        if comm.direction is BridgeDirection.FAN_IN:
            # All-gather sibling src outputs into a single full-batch tensor.
            return _all_gather_along_batch_dim(tensor, comm.gather_pg, ctx.batch_dim)

        # EQUAL: pure passthrough.
        return tensor.contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Adjoint of forward: narrow for fan-in, all-gather for fan-out.

        Fan-out's forward is ``narrow``, whose naive adjoint is zero-pad.
        That would leave each src rank with only its own dest rank's
        slice of the gradient, missing the contributions from every
        other dest rank that consumed a different slice of the same src
        activation. Instead we all-gather across the fan-out sibling
        group, reconstructing the full src-batch gradient (symmetric
        with the fan-in forward's all-gather).

        When the dest grid has CP>1, the LLM's PartitionAdapter.shard slices
        sequence via ``index_select`` whose autograd adjoint is a scatter /
        zero-pad. So the grad flowing into this backward is already zero-
        padded along the sequence dimension — each CP rank holds only its
        own sequence chunks, zeros elsewhere. We therefore run an intra-CP
        ``all_reduce(SUM)`` on the incoming gradient *before* the direction-
        specific op. After the reduction every CP sibling holds the same
        full-sequence gradient, and the downstream narrow (fan-in) or
        all-gather (fan-out) proceeds exactly as in the CP=1 case.
        """
        comm = ctx.comm
        batch_dim = ctx.batch_dim

        # CP sequence-reduction step. Runs for both fan-in and fan-out when
        # dest CP>1. See the docstring for the math. Clone first so the in-
        # place all_reduce does not mutate the tensor autograd passed in.
        if comm.dest_cp_pg is not None:
            grad_output = grad_output.contiguous().clone()
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=comm.dest_cp_pg)

        if comm.direction is BridgeDirection.FAN_OUT:
            return _all_gather_along_batch_dim(grad_output, comm.gather_pg, batch_dim), None

        if comm.direction is BridgeDirection.FAN_IN:
            slice_info = comm.get_slice_info(grad_output.shape[batch_dim])
            return (
                grad_output.narrow(batch_dim, slice_info.start, slice_info.size).contiguous(),
                None,
            )

        return grad_output.contiguous(), None


def _all_gather_along_batch_dim(
    tensor: torch.Tensor, group: dist.ProcessGroup, batch_dim: int
) -> torch.Tensor:
    """All-gather ``tensor`` along an arbitrary batch dim into a single tensor.

    ``all_gather_into_tensor`` concatenates along dim 0, so when the
    batch dim is not 0 we move it, gather, then restore.
    """
    world_size = dist.get_world_size(group)
    src = tensor.contiguous()
    if batch_dim != 0:
        src = src.movedim(batch_dim, 0).contiguous()
    out_shape = list(src.shape)
    out_shape[0] *= world_size
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(out, src, group=group)
    if batch_dim != 0:
        out = out.movedim(0, batch_dim).contiguous()
    return out
