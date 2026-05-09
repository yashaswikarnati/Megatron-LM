# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch.distributed as dist

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    from absl import logging

    HAVE_ABSL = True
except ImportError:
    import logging
    import warnings

    logging = logging.getLogger(__name__)
    warnings.warn(
        "absl.logging is not installed. Using logging.getLogger(__name__) instead. "
        "Please install absl.logging with `pip install absl-py` to use absl.logging."
    )
    HAVE_ABSL = False


def _is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Return whether pg is a real process group for this rank."""
    group_member = getattr(dist, "GroupMember", None)
    non_member = getattr(group_member, "NON_GROUP_MEMBER", None)
    return pg is not None and pg != non_member


@dataclass
class _GridLayout:
    """Rank layout owned by a HyperCommGrid.

    The base layout is the original Cartesian grid. Registered layouts are
    alternate factorizations over the same rank span.
    """

    shape: list[int]
    dim_names: list[str]
    aliases: dict[str, list[str]] = field(default_factory=dict)


class HyperCommGrid:
    r"""N-dimensional communication grid.

    Manages an arbitrary number of parallelisms as a hyperrectangle. Each dimension is given a name
    at initialization time. The order of ``dim_names`` implies the mapping order equivalent to
    the ``order`` argument of MCore's ``initialize_model_parallel``. Internally, it has to be
    reversed to match n-D array.

    For any combination of dimensions, a process group can only be created once.
    Creating process groups for the same combination with different options is not supported.

    Note:
        ``create_pg()`` over specific dims must be explicitly called to create a process group.
        We don't create a process group in the ``get_pg()`` function because there are many options
        (kwargs) that can be passed when creating a process group, which ``get_pg()`` should not
        be exposed to.

    Examples:
        >>> grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])
        >>> dp_group = grid.create_pg("dp")
        >>> # retrieve dp_group from grid after creation
        >>> # dp_group = grid.get_pg("dp")
        >>>
        >>> # It is equivalent to calling the following functions in MCore parallel_state
        >>> # with world size 120.
        >>> parallel_state.initialize_model_parallel(
        >>>     tensor_model_parallel_size=2,
        >>>     context_parallel_size=3,
        >>>     pipeline_model_parallel_size=4,
        >>>     order="tp-cp-pp-dp")
        >>> dp_group_mcore = parallel_state.get_data_parallel_group()
        >>>
        >>> # We can create group from multiple leading dims and also pass more options.
        >>> pg_options = ProcessGroupNCCL.Options()
        >>> pg_options.config.max_ctas = 8
        >>> dp_cp_group = grid.create_pg(
        >>>     ["cp", "dp"], pg_options=pg_options,
        >>>     group_desc="WEIGHT_GRADIENT_COMM_GROUP")


    Args:
        shape: Shape of the communication grid.
        dim_names: Name of each dimension corresponding to shape. Must have the same length as
            shape.
        rank_offset: Starting rank when the grid doesn't span the entire communication world.
            Default 0.
        backend: Backend for creating process group. Default None and will use default backend.
    """

    def __init__(
        self,
        shape: list[int],
        dim_names: list[str],
        rank_offset: int = 0,
        backend: Optional[str] = None,
    ) -> None:
        if len(shape) != len(dim_names):
            raise ValueError(f"len(shape) {shape} != len(dim_names) {dim_names}")

        # Querying environment instead of calling torch.distributed.get_world_size() for mock
        # testing without initializing process group.
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            raise RuntimeError(
                "Cannot determine world size: WORLD_SIZE environment variable not set and "
                "torch.distributed is not initialized. Please either set WORLD_SIZE or "
                "initialize torch.distributed before creating HyperCommGrid."
            )
        self.rank_offset = rank_offset
        self.size = np.prod(shape)
        if rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {rank_offset}")
        if self.size > world_size - rank_offset:
            raise RuntimeError(
                f"Grid shape {shape} is over sized with world size {world_size} and rank "
                f"offset {self.rank_offset}"
            )

        # [:] insures a copy
        self.shape = shape[:]
        self.dim_names = dim_names[:]
        self.backend = backend
        self._pgs: dict[str, dist.ProcessGroup] = {}
        self._layouts: dict[str, _GridLayout] = {"base": _GridLayout(self.shape, self.dim_names)}
        self._aliases: dict[str, tuple[str, list[str]]] = {}

    def register_layout(
        self,
        name: str,
        shape: list[int],
        dim_names: list[str],
        aliases: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Register an alternate rank layout over this grid's rank span.

        Registered layouts are useful when the same module rank universe has
        more than one valid factorization, such as dense
        ``tp/cp/dp/pp`` groups and expert ``expt_tp/ep/expt_dp/pp``
        groups. The original constructor remains a single Cartesian grid.

        Args:
            name: Unique name for the alternate layout.
            shape: Shape of the alternate layout. Its product must equal the
                base grid size.
            dim_names: Dimension names for the alternate layout.
            aliases: Optional names for composite groups in this layout.
                For example, ``{"tp_ep": ["expt_tp", "ep"]}``.
        """
        if name == "base":
            raise ValueError("'base' is reserved for the default HyperCommGrid layout")
        if name in self._layouts:
            raise ValueError(f"Layout {name!r} is already registered")
        if len(shape) != len(dim_names):
            raise ValueError(f"len(shape) {shape} != len(dim_names) {dim_names}")
        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"Layout {name!r} has duplicate dim_names: {dim_names}")
        if np.prod(shape) != self.size:
            raise ValueError(
                f"Layout {name!r} shape {shape} has size {np.prod(shape)}, "
                f"but base grid size is {self.size}"
            )

        layout = _GridLayout(shape[:], dim_names[:])

        for dim in set(dim_names).intersection(self.dim_names):
            base_enum = self._gen_rank_enum_for_layout([dim], "base")
            layout_enum = self._gen_rank_enum_for_layout([dim], None, layout)
            if base_enum != layout_enum:
                raise ValueError(
                    f"Layout {name!r} dimension {dim!r} collides with the base layout "
                    "but has different rank enumeration"
                )

        aliases = aliases or {}
        for alias_name, alias_dims in aliases.items():
            if alias_name in self._aliases:
                raise ValueError(f"Alias {alias_name!r} is already registered")
            if alias_name in self.dim_names or alias_name in dim_names:
                raise ValueError(f"Alias {alias_name!r} conflicts with an existing dimension name")
            if "-" in alias_name:
                raise ValueError(
                    f"Alias {alias_name!r} cannot contain '-' because process group keys use '-'"
                )
            if len(set(alias_dims)) != len(alias_dims):
                raise ValueError(f"Alias {alias_name!r} has duplicate dimensions: {alias_dims}")
            missing_dims = [dim for dim in alias_dims if dim not in dim_names]
            if missing_dims:
                raise ValueError(
                    f"Alias {alias_name!r} references dimensions not in layout {name!r}: "
                    f"{missing_dims}"
                )
            layout.aliases[alias_name] = alias_dims[:]

        self._layouts[name] = layout
        for alias_name, alias_dims in layout.aliases.items():
            self._aliases[alias_name] = (name, alias_dims[:])

    def has_layout(self, name: str) -> bool:
        """Return whether a named layout is registered."""
        return name in self._layouts

    def has_alias(self, name: str) -> bool:
        """Return whether an alias is registered."""
        return name in self._aliases

    def get_alias_dims(self, name: str) -> list[str]:
        """Return a copy of the dimensions referenced by an alias."""
        if name not in self._aliases:
            raise KeyError(f"Alias {name!r} is not registered")
        _, alias_dims = self._aliases[name]
        return alias_dims[:]

    def create_pg(self, dims: Union[str, list[str]], **kwargs: Any) -> dist.ProcessGroup | None:
        r"""Create a process group based on a list of dimension names

        Note: The unique key used to store the process group internally will follow the reversed
        order of the original dim_names. For example, if dim_names=["tp", "cp", "dp"] and you
        create a process group with dims=["dp", "tp"], the unique_group_key will be "dp-tp"
        (ordered according to the reversed dim_names order: ["dp", "cp", "tp"]).

        Args:
            dims: Name of leading dimensions to create process group

        Keyword arguments are directly passed into new_subgroups_by_enumeration(). The docstring
        is copied from new_subgroups_by_enumeration().

        Keyword args from `dist.new_subgroups_by_enumeration`:
            timeout (timedelta, optional): see `init_process_group` for details and default value.
            pg_options (ProcessGroupOptions, optional): process group options
                specifying what additional options need to be passed in during
                the construction of specific process groups.
            group_desc (str, optional): A string describing the group. Each subgroup will
                inherit its group_desc.

        Returns:
            dist.ProcessGroup | None: The created process group.

        Raises:
            KeyError: If attempting to recreate a process group with an existing key.
        """
        # ordered_dims follows the reversed order of the owning layout's dim_names.
        layout_name, ordered_dims, unique_group_key = self._resolve_dims(dims)

        if unique_group_key in self._pgs:
            raise KeyError(
                f"Process group {dims} has already been created. Because there is no way to check "
                f"whether options to create process group matches the first, we error out instead "
                f"of returning the process group that has already been created before."
            )

        rank_enum = self._gen_rank_enum_for_layout(ordered_dims, layout_name)
        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend, **kwargs)

        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(
                f"Generated process group for {unique_group_key} with enumeration {rank_enum}"
            )
        self._pgs[unique_group_key] = pg
        return pg

    def destroy(self) -> None:
        """Destroy all process groups created by this grid."""
        for pg in self._pgs.values():
            if _is_process_group_member(pg):
                dist.destroy_process_group(pg)
        self._pgs.clear()

    def get_pg(self, dims: Union[str, list[str]]) -> dist.ProcessGroup:
        r"""Get a process group based on a list of dimension names

        Args:
            dims: Name of leading dimensions to create process group
        """
        _, _, unique_group_key = self._resolve_dims(dims)

        if unique_group_key not in self._pgs:
            raise KeyError(
                f"Process group for {unique_group_key} hasn't been created. Call create_pg first."
            )

        return self._pgs[unique_group_key]

    def get_rank_enum(
        self, dims: Union[str, list[str]], layout_name: Optional[str] = None
    ) -> list[list[int]]:
        r"""Get the rank enumeration for the requested dimension(s).

        This is the exact enumeration that would be used by create_pg for the same
        dims. It is useful for creating additional groups whose membership is derived from
        the grid (e.g., embedding/position-embedding groups derived from PP groups).

        Args:
            dims: Dimension name or list of dimension names.
            layout_name: Optional registered layout name. When unset, the
                owning layout is inferred from dims or aliases.

        Returns:
            List of rank lists (one per subgroup).
        """
        if layout_name is None:
            layout_name, ordered_dims, _ = self._resolve_dims(dims)
        else:
            dims = self._expand_alias(dims, layout_name)
            ordered_dims, _ = self._order_dims_for_layout(dims, layout_name)
        return self._gen_rank_enum_for_layout(ordered_dims, layout_name)

    def _gen_rank_enum(self, dims: list[str]) -> list[list[int]]:
        return self._gen_rank_enum_for_layout(dims, "base")

    def _gen_rank_enum_for_layout(
        self, dims: list[str], layout_name: Optional[str], layout: Optional[_GridLayout] = None
    ) -> list[list[int]]:
        r"""Generate rank enumeration before calling new_subgroups_by_enumeration

        This function returns ranks grouped by the specified dimensions, but in REVERSE order
        of the input dimensions. For example, if you request dimensions ["a", "b"],
        the ranks will be grouped by "b-a" order.

        Example:
            For a grid with shape [2, 2, 2] and dim_names ["a", "b", "c"]:
            _gen_rank_enum(["a", "b"]) returns [[0, 2, 1, 3], [4, 6, 5, 7]]

            This groups ranks first by dimension "b", then by dimension "a":
            - Group 0: ranks where c=0, grouped by b-a: [0, 2, 1, 3]
            - Group 1: ranks where c=1, grouped by b-a: [4, 6, 5, 7]

        Args:
            dims: Name of leading dimensions to create process group

        Although the function is lightweight enough to be inlined, a standalone one makes it
        easier to test against MCore's RankGenerator
        """

        if not HAVE_EINOPS:
            raise RuntimeError(
                "einops is not installed. Please install it with `pip install einops`."
            )
        if layout is None:
            assert layout_name is not None
            layout = self._layouts[layout_name]

        # Need to reverse order of dim_names to match MCore convention
        dim_names_reverse = layout.dim_names[::-1]

        remaining_dims = []
        for v in dim_names_reverse:
            if v not in dims:
                remaining_dims.append(v)

        rearrange_str = (
            f"({' '.join(dim_names_reverse)}) -> ({' '.join(remaining_dims)}) ({' '.join(dims)})"
        )
        logging.debug(rearrange_str)

        shape_dict = {d: s for d, s in zip(layout.dim_names, layout.shape)}
        return einops.rearrange(
            np.arange(self.rank_offset, self.rank_offset + self.size), rearrange_str, **shape_dict
        ).tolist()

    def _order_dims(self, dims: Union[str, list[str]]) -> Tuple[list[str], str]:
        return self._order_dims_for_layout(dims, "base")

    def _order_dims_for_layout(
        self, dims: Union[str, list[str]], layout_name: str
    ) -> Tuple[list[str], str]:
        r"""Reorder dims based on the order of self.dim_names"""
        layout = self._layouts[layout_name]
        if not isinstance(dims, list):
            if dims not in layout.dim_names:
                raise ValueError(
                    f"Dimension {dims!r} is not in layout {layout_name!r}: {layout.dim_names}"
                )
            ordered_dims = [dims]
        else:
            dim_names_reverse = layout.dim_names[::-1]
            missing_dims = [d for d in dims if d not in dim_names_reverse]
            if missing_dims:
                raise ValueError(
                    f"Dimensions {missing_dims} are not in layout {layout_name!r}: "
                    f"{layout.dim_names}"
                )
            indices = sorted([dim_names_reverse.index(d) for d in dims])
            if len(indices) == 1:
                ordered_dims = [dim_names_reverse[indices[0]]]
            else:
                ordered_dims = list(itemgetter(*indices)(dim_names_reverse))

        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def _resolve_dims(self, dims: Union[str, list[str]]) -> Tuple[str, list[str], str]:
        if isinstance(dims, str) and dims in self._aliases:
            layout_name, alias_dims = self._aliases[dims]
            ordered_dims, _ = self._order_dims_for_layout(alias_dims, layout_name)
            return layout_name, ordered_dims, dims

        raw_dims = [dims] if isinstance(dims, str) else dims

        if all(dim in self.dim_names for dim in raw_dims):
            ordered_dims, unique_group_key = self._order_dims_for_layout(raw_dims, "base")
            return "base", ordered_dims, unique_group_key

        candidate_layouts = [
            name
            for name, layout in self._layouts.items()
            if name != "base" and all(dim in layout.dim_names for dim in raw_dims)
        ]
        if not candidate_layouts:
            raise ValueError(
                f"Dimensions {raw_dims} are not all present in a single registered layout"
            )
        if len(candidate_layouts) > 1:
            raise ValueError(
                f"Dimensions {raw_dims} match multiple registered layouts: {candidate_layouts}"
            )

        layout_name = candidate_layouts[0]
        if len(raw_dims) > 1:
            aliases = sorted(self._layouts[layout_name].aliases)
            raise ValueError(
                f"Composite dimensions {raw_dims} from registered layout {layout_name!r} "
                f"must use an explicit alias. Available aliases: {aliases}"
            )

        ordered_dims, unique_group_key = self._order_dims_for_layout(raw_dims, layout_name)
        return layout_name, ordered_dims, unique_group_key

    def _expand_alias(self, dims: Union[str, list[str]], layout_name: str) -> Union[str, list[str]]:
        if not isinstance(dims, str) or dims not in self._aliases:
            return dims

        alias_layout_name, alias_dims = self._aliases[dims]
        if alias_layout_name != layout_name:
            raise ValueError(
                f"Alias {dims!r} belongs to layout {alias_layout_name!r}, not {layout_name!r}"
            )
        return alias_dims[:]

    def is_current_rank_in_grid(self) -> bool:
        """Check if the current rank belongs to this grid.

        Returns:
            True if the current rank is within [rank_offset, rank_offset + size).
        """
        rank = dist.get_rank()
        return bool(self.rank_offset <= rank < self.rank_offset + self.size)
