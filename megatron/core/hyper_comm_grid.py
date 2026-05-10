# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
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
        alt_factorizations: Optional alternate factorizations of a contiguous block of the primary
            dim_names. Each entry re-expresses the same rank slab under different axis names with
            a different shape. Used to overlap expert parallelism (EP / ETP / EDP) onto the same
            ranks that carry TP / CP / DP without inflating the world size. The mapping has the
            shape ``{name: {"shape": [...], "dim_names": [...], "replaces": [...]}}``.

            Constraints (enforced at construction):

            * ``replaces`` must be a contiguous slice of the primary ``dim_names``.
            * The product of the alt ``shape`` must equal the product of the primary shape values
              at the covered positions.
            * Alt ``dim_names`` must not collide with primary ``dim_names`` or with names from
              any other alt factorization.

            Example for NMFW-464 expert overlap (8 ranks, ``tp=cp=dp=2``, ``ep=etp=edp=2``)::

                HyperCommGrid(
                    [2, 2, 2, 1], ["tp", "cp", "dp", "pp"],
                    alt_factorizations={
                        "expert": {
                            "shape": [2, 2, 2],
                            "dim_names": ["etp", "ep", "edp"],
                            "replaces": ["tp", "cp", "dp"],
                        },
                    },
                )

            ``create_pg("ep")`` then enumerates the same rank slab as ``create_pg("cp")`` would
            for the primary, but under the expert factorization. Mixing covered primary dims
            (e.g. ``tp``) with alt dims (e.g. ``ep``) in a single ``create_pg`` call is rejected
            because the two views share ranks and the combined group is ambiguous.
    """

    def __init__(
        self,
        shape: list[int],
        dim_names: list[str],
        rank_offset: int = 0,
        backend: Optional[str] = None,
        alt_factorizations: Optional[dict[str, dict[str, Any]]] = None,
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

        # Alt factorizations: each builds a "shadow" (dim_names, shape) that expresses the same
        # flat rank range under a different naming, by replacing the contiguous slice of primary
        # dim_names listed in ``replaces`` with the alt's dim_names and shape. The shadow drives
        # einops enumeration when the caller asks for groups along alt axes.
        self._alt_shadows: dict[str, Tuple[list[str], list[int]]] = {}
        # Map from primary dim name → alt name that replaces it. Used to detect ambiguous
        # mixed-factorization requests in ``_resolve_dims``.
        self._replaced_to_alt: dict[str, str] = {}
        # Map from alt-axis dim name → alt name that owns it.
        self._dim_to_alt: dict[str, str] = {}
        if alt_factorizations:
            for alt_name, alt_spec in alt_factorizations.items():
                shadow = self._validate_and_build_alt(alt_name, alt_spec)
                self._alt_shadows[alt_name] = shadow
                for d in alt_spec["dim_names"]:
                    self._dim_to_alt[d] = alt_name
                for d in alt_spec["replaces"]:
                    if d in self._replaced_to_alt:
                        other = self._replaced_to_alt[d]
                        raise ValueError(
                            f"alt_factorization {alt_name!r}: primary dim {d!r} is already "
                            f"replaced by alt factorization {other!r}; alt factorizations must "
                            f"replace disjoint slices of the primary"
                        )
                    self._replaced_to_alt[d] = alt_name

    def _validate_and_build_alt(
        self, alt_name: str, alt_spec: dict[str, Any]
    ) -> Tuple[list[str], list[int]]:
        r"""Validate one alt factorization and return its ``(shadow_dim_names, shadow_shape)``."""
        for required in ("shape", "dim_names", "replaces"):
            if required not in alt_spec:
                raise ValueError(
                    f"alt_factorization {alt_name!r} is missing required key {required!r}"
                )
        alt_shape = list(alt_spec["shape"])
        alt_dim_names = list(alt_spec["dim_names"])
        replaces = list(alt_spec["replaces"])
        if len(alt_shape) != len(alt_dim_names):
            raise ValueError(
                f"alt_factorization {alt_name!r}: len(shape) {alt_shape} != "
                f"len(dim_names) {alt_dim_names}"
            )
        if not replaces:
            raise ValueError(f"alt_factorization {alt_name!r}: replaces must be non-empty")

        # replaces must be a contiguous slice of primary dim_names
        for c in replaces:
            if c not in self.dim_names:
                raise ValueError(
                    f"alt_factorization {alt_name!r}: replaces entry {c!r} is not a primary dim"
                )
        first_idx = self.dim_names.index(replaces[0])
        expected = self.dim_names[first_idx : first_idx + len(replaces)]
        if expected != replaces:
            raise ValueError(
                f"alt_factorization {alt_name!r}: replaces {replaces} must be a contiguous slice "
                f"of primary dim_names {self.dim_names}"
            )

        # product(alt.shape) == product(primary.shape over replaced positions)
        primary_replaced_prod = int(np.prod(self.shape[first_idx : first_idx + len(replaces)]))
        alt_prod = int(np.prod(alt_shape))
        if alt_prod != primary_replaced_prod:
            raise ValueError(
                f"alt_factorization {alt_name!r}: product(shape) {alt_prod} != product of "
                f"primary replaced dims {primary_replaced_prod}"
            )

        # alt dim_names must not collide with primary or other alt names
        for d in alt_dim_names:
            if d in self.dim_names:
                raise ValueError(
                    f"alt_factorization {alt_name!r}: dim {d!r} collides with primary dim_names"
                )
            if d in self._dim_to_alt:
                other = self._dim_to_alt[d]
                raise ValueError(
                    f"alt_factorization {alt_name!r}: dim {d!r} collides with alt "
                    f"factorization {other!r}"
                )

        # Build shadow by replacing the contiguous slice with alt
        shadow_dim_names = (
            self.dim_names[:first_idx] + alt_dim_names + self.dim_names[first_idx + len(replaces) :]
        )
        shadow_shape = self.shape[:first_idx] + alt_shape + self.shape[first_idx + len(replaces) :]
        return shadow_dim_names, shadow_shape

    def _resolve_dims(self, dims_list: list[str]) -> Tuple[list[str], list[int]]:
        r"""Pick the layout (primary or alt-shadow) that should handle ``dims_list``.

        Returns:
            ``(dim_names, shape)`` — the layout to use for rank-enumeration.

        Raises:
            KeyError: if a requested dim is unknown.
            ValueError: if the request mixes replaced primary dims with alt dims, or alt dims
                from different factorizations.
        """
        alts_used: set[str] = set()
        has_replaced_primary = False
        for d in dims_list:
            if d in self._dim_to_alt:
                alts_used.add(self._dim_to_alt[d])
            elif d in self.dim_names:
                if d in self._replaced_to_alt:
                    has_replaced_primary = True
            else:
                raise KeyError(f"Dimension {d!r} is not a primary or alt dim of this grid")

        if len(alts_used) > 1:
            raise ValueError(
                f"create_pg/get_pg cannot mix dims from multiple alt factorizations: "
                f"{sorted(alts_used)}"
            )

        if alts_used:
            alt_name = next(iter(alts_used))
            if has_replaced_primary:
                raise ValueError(
                    f"Cannot combine replaced primary dims with dims from alt factorization "
                    f"{alt_name!r}; the views share ranks and the combined group is ambiguous"
                )
            return self._alt_shadows[alt_name]

        return self.dim_names, self.shape

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
        dims_list = [dims] if isinstance(dims, str) else list(dims)
        layout_names, layout_shape = self._resolve_dims(dims_list)
        # ordered_dims and unique_group_key follow the reversed order of layout_names
        ordered_dims, unique_group_key = self._order_dims(dims, dim_names=layout_names)

        if unique_group_key in self._pgs:
            raise KeyError(
                f"Process group {dims} has already been created. Because there is no way to check "
                f"whether options to create process group matches the first, we error out instead "
                f"of returning the process group that has already been created before."
            )

        rank_enum = self._gen_rank_enum(ordered_dims, dim_names=layout_names, shape=layout_shape)
        pg, _ = dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend, **kwargs)

        if dist.is_initialized() and dist.get_rank() == 0:
            logging.info(
                f"Generated process group for {unique_group_key} with enumeration {rank_enum}"
            )
        self._pgs[unique_group_key] = pg
        return pg

    def destroy(self) -> None:
        """Destroy all process groups created by this grid."""
        for pg in self._pgs.values():
            if pg is not None:
                dist.destroy_process_group(pg)
        self._pgs.clear()

    def get_pg(self, dims: Union[str, list[str]]) -> dist.ProcessGroup:
        r"""Get a process group based on a list of dimension names

        Args:
            dims: Name of leading dimensions to create process group
        """
        dims_list = [dims] if isinstance(dims, str) else list(dims)
        layout_names, _ = self._resolve_dims(dims_list)
        _, unique_group_key = self._order_dims(dims, dim_names=layout_names)

        if unique_group_key not in self._pgs:
            raise KeyError(
                f"Process group for {unique_group_key} hasn't been created. Call create_pg first."
            )

        return self._pgs[unique_group_key]

    def get_rank_enum(self, dims: Union[str, list[str]]) -> list[list[int]]:
        r"""Get the rank enumeration for the requested dimension(s).

        This is the exact enumeration that would be used by create_pg for the same
        dims. It is useful for creating additional groups whose membership is derived from
        the grid (e.g., embedding/position-embedding groups derived from PP groups).

        Args:
            dims: Dimension name or list of dimension names.

        Returns:
            List of rank lists (one per subgroup).
        """
        dims_list = [dims] if isinstance(dims, str) else list(dims)
        layout_names, layout_shape = self._resolve_dims(dims_list)
        ordered_dims, _ = self._order_dims(dims, dim_names=layout_names)
        return self._gen_rank_enum(ordered_dims, dim_names=layout_names, shape=layout_shape)

    def _gen_rank_enum(
        self,
        dims: list[str],
        dim_names: Optional[list[str]] = None,
        shape: Optional[list[int]] = None,
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
            dim_names: Layout dim_names to use; defaults to ``self.dim_names``. When the caller
                requests groups along an alt factorization, this is overridden by the alt's
                shadow dim_names.
            shape: Layout shape to use; defaults to ``self.shape``. Like ``dim_names``, this is
                overridden when generating groups along an alt factorization.

        Although the function is lightweight enough to be inlined, a standalone one makes it
        easier to test against MCore's RankGenerator
        """

        if not HAVE_EINOPS:
            raise RuntimeError(
                "einops is not installed. Please install it with `pip install einops`."
            )

        layout_names = self.dim_names if dim_names is None else dim_names
        layout_shape = self.shape if shape is None else shape

        # Need to reverse order of dim_names to match MCore convention
        dim_names_reverse = layout_names[::-1]

        remaining_dims = []
        for v in dim_names_reverse:
            if v not in dims:
                remaining_dims.append(v)

        rearrange_str = (
            f"({' '.join(dim_names_reverse)}) -> ({' '.join(remaining_dims)}) ({' '.join(dims)})"
        )
        logging.debug(rearrange_str)

        shape_dict = {d: s for d, s in zip(layout_names, layout_shape)}
        return einops.rearrange(
            np.arange(self.rank_offset, self.rank_offset + self.size), rearrange_str, **shape_dict
        ).tolist()

    def _order_dims(
        self, dims: Union[str, list[str]], dim_names: Optional[list[str]] = None
    ) -> Tuple[list[str], str]:
        r"""Reorder dims based on the order of ``dim_names`` (defaults to ``self.dim_names``)."""
        layout_names = self.dim_names if dim_names is None else dim_names
        if not isinstance(dims, list):
            ordered_dims = [dims]
        else:
            dim_names_reverse = layout_names[::-1]
            indices = sorted([dim_names_reverse.index(d) for d in dims])
            if len(indices) == 1:
                ordered_dims = [dim_names_reverse[indices[0]]]
            else:
                ordered_dims = list(itemgetter(*indices)(dim_names_reverse))

        unique_group_key = "-".join(ordered_dims)
        return ordered_dims, unique_group_key

    def is_current_rank_in_grid(self) -> bool:
        """Check if the current rank belongs to this grid.

        Returns:
            True if the current rank is within [rank_offset, rank_offset + size).
        """
        rank = dist.get_rank()
        return bool(self.rank_offset <= rank < self.rank_offset + self.size)
