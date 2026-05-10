# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


class TestHyperCommGrid:
    """Comprehensive tests for HyperCommGrid class."""

    def test_init_basic(self):
        """Test basic initialization of HyperCommGrid."""
        shape = [2, 2, 2]
        dim_names = ["tp", "cp", "dp"]

        grid = HyperCommGrid(shape, dim_names)

        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.rank_offset == 0
        assert grid.backend is None
        assert grid.size == 8  # 2 * 2 * 2
        assert grid._pgs == {}

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        shape = [2, 2]  # Changed from [2, 4] to fit world size 8 with offset 8
        dim_names = ["tp", "dp"]
        rank_offset = 0  # Changed from 8 to 0 to avoid size error
        backend = "nccl"

        grid = HyperCommGrid(shape, dim_names, rank_offset, backend)

        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.rank_offset == rank_offset
        assert grid.backend == backend
        assert grid.size == 4  # 2 * 2

    def test_init_validation_errors(self):
        """Test initialization validation errors."""
        # Shape and dim_names length mismatch
        with pytest.raises(ValueError, match="len\\(shape\\).*!= len\\(dim_names\\)"):
            HyperCommGrid([2, 2], ["tp"])

        # Grid too large for world size
        with pytest.raises(RuntimeError, match="Grid shape.*is over sized"):
            HyperCommGrid([4, 4], ["tp", "dp"])  # 16 > 8 world size

    def test_order_dims_single_dim(self):
        """Test _order_dims with single dimension."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4] to fit world size

        ordered_dims, unique_key = grid._order_dims("cp")

        assert ordered_dims == ["cp"]
        assert unique_key == "cp"

    def test_order_dims_multiple_dims(self):
        """Test _order_dims with multiple dimensions."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4, 5] to fit world size

        # Should order according to reversed dim_names order
        ordered_dims, unique_key = grid._order_dims(["dp", "tp"])

        assert ordered_dims == [
            "dp",
            "tp",
        ]  # Changed: dp comes before tp in reversed order ["dp", "cp", "tp"]
        assert unique_key == "dp-tp"

    def test_order_dims_all_dims(self):
        """Test _order_dims with all dimensions."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4] to fit world size

        ordered_dims, unique_key = grid._order_dims(["dp", "cp", "tp"])

        assert ordered_dims == ["dp", "cp", "tp"]  # Changed: reversed order
        assert unique_key == "dp-cp-tp"

    def test_gen_rank_enum_single_dim(self):
        """Test _gen_rank_enum for single dimension."""
        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        rank_enum = grid._gen_rank_enum(["tp"])

        # Should have 4 groups of 2 ranks each
        expected = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert rank_enum == expected

    def test_gen_rank_enum_multiple_dims(self):
        """Test _gen_rank_enum for multiple dimensions."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        rank_enum = grid._gen_rank_enum(["tp", "cp"])

        # Should have 2 groups (for dp) with 4 ranks each (tp * cp)
        expected = [[0, 2, 1, 3], [4, 6, 5, 7]]  # Updated to match actual einops rearrange result
        assert rank_enum == expected

    def test_gen_rank_enum_with_offset(self):
        """Test _gen_rank_enum with rank offset."""
        grid = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=4)

        rank_enum = grid._gen_rank_enum(["tp"])

        # Should start from rank 4
        expected = [[4, 5], [6, 7]]
        assert rank_enum == expected

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_single_dim(self, mock_new_subgroups):
        """Test create_pg for single dimension."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        result = grid.create_pg("tp")

        assert result == mock_pg
        assert "tp" in grid._pgs
        assert grid._pgs["tp"] == mock_pg

        # Verify the enumeration passed to new_subgroups_by_enumeration
        args, kwargs = mock_new_subgroups.call_args
        expected_enum = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert args[0] == expected_enum
        assert kwargs["backend"] is None

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_multiple_dims(self, mock_new_subgroups):
        """Test create_pg for multiple dimensions."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        result = grid.create_pg(["tp", "cp"])

        assert result == mock_pg
        assert "cp-tp" in grid._pgs

        args, kwargs = mock_new_subgroups.call_args
        expected_enum = [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert args[0] == expected_enum

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_with_options(self, mock_new_subgroups):
        """Test create_pg with additional options."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"], backend="nccl")

        # Mock ProcessGroupNCCL.Options
        mock_options = MagicMock()

        result = grid.create_pg("tp", pg_options=mock_options, group_desc="TEST_GROUP")

        assert result == mock_pg

        args, kwargs = mock_new_subgroups.call_args
        assert kwargs["backend"] == "nccl"
        assert kwargs["pg_options"] == mock_options

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_duplicate_error(self, mock_new_subgroups):
        """Test create_pg raises error when trying to recreate existing process group."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        # Create process group first time
        grid.create_pg("tp")

        # Try to create again should raise KeyError
        with pytest.raises(KeyError, match="Process group.*has already been created"):
            grid.create_pg("tp")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_get_pg_success(self, mock_new_subgroups):
        """Test get_pg returns existing process group."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        # Create process group first
        grid.create_pg("dp")

        # Get should return the same process group
        result = grid.get_pg("dp")
        assert result == mock_pg

    def test_get_pg_not_created_error(self):
        """Test get_pg raises error when process group doesn't exist."""
        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        with pytest.raises(KeyError, match="Process group for.*hasn't been created"):
            grid.get_pg("tp")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_get_pg_multiple_dims(self, mock_new_subgroups):
        """Test get_pg with multiple dimensions."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Create process group with multiple dims
        grid.create_pg(["cp", "dp"])

        # Get should work with different order
        result = grid.get_pg(["dp", "cp"])
        assert result == mock_pg

    def test_complex_grid_scenario(self):
        """Test a complex scenario similar to the docstring example."""
        os.environ["WORLD_SIZE"] = "120"  # Set larger world size for this test

        grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])

        assert grid.size == 120
        assert grid.shape == [2, 3, 4, 5]
        assert grid.dim_names == ["tp", "cp", "pp", "dp"]

        # Test ordering of different dimension combinations
        ordered_dims, key = grid._order_dims(["dp", "pp"])
        assert ordered_dims == ["dp", "pp"]  # Changed: actual order matches reversed dim_names
        assert key == "dp-pp"

        # Test rank enumeration for dp (last dimension)
        rank_enum = grid._gen_rank_enum(["dp"])
        assert len(rank_enum) == 24  # 2 * 3 * 4 = 24 groups
        assert len(rank_enum[0]) == 5  # Each group has 5 ranks

        # Clean up
        os.environ["WORLD_SIZE"] = "8"

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_end_to_end_workflow(self, mock_new_subgroups):
        """Test complete workflow: init -> create -> get."""
        mock_pg1 = MagicMock(spec=dist.ProcessGroup)
        mock_pg2 = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.side_effect = [(mock_pg1, None), (mock_pg2, None)]

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Create different process groups
        tp_pg = grid.create_pg("tp")
        dp_cp_pg = grid.create_pg(["dp", "cp"])

        # Verify they're created correctly
        assert tp_pg == mock_pg1
        assert dp_cp_pg == mock_pg2

        # Verify we can get them back
        assert grid.get_pg("tp") == mock_pg1
        assert grid.get_pg(["cp", "dp"]) == mock_pg2  # Different order should work

        # Verify internal state
        assert len(grid._pgs) == 2
        assert "tp" in grid._pgs
        assert "dp-cp" in grid._pgs  # Changed: actual key order

    def test_edge_case_single_rank_dims(self):
        """Test edge case with dimensions of size 1."""
        grid = HyperCommGrid([1, 2, 4], ["tp", "cp", "dp"])

        # Test with tp dimension (size 1)
        rank_enum = grid._gen_rank_enum(["tp"])
        expected = [[0], [1], [2], [3], [4], [5], [6], [7]]  # 8 groups of 1 rank each
        assert rank_enum == expected

        # Test with multiple dims including size 1
        rank_enum = grid._gen_rank_enum(["tp", "cp"])
        expected = [[0, 1], [2, 3], [4, 5], [6, 7]]  # 4 groups of 2 ranks each
        assert rank_enum == expected

    def test_rank_enumeration_correctness(self):
        """Test that rank enumeration produces correct pattern."""
        grid = HyperCommGrid([2, 2, 2], ["a", "b", "c"])

        # For dimension "a" (first in original order, last in reversed)
        rank_enum_a = grid._gen_rank_enum(["a"])
        expected_a = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert rank_enum_a == expected_a

        # For dimension "c" (last in original order, first in reversed)
        rank_enum_c = grid._gen_rank_enum(["c"])
        expected_c = [[0, 4], [1, 5], [2, 6], [3, 7]]
        assert rank_enum_c == expected_c

        # For dimensions "a" and "b"
        rank_enum_ab = grid._gen_rank_enum(["a", "b"])
        expected_ab = [[0, 2, 1, 3], [4, 6, 5, 7]]
        assert rank_enum_ab == expected_ab


class TestHyperCommGridAltFactorization:
    """Tests for the alt-factorization feature (NMFW-464 expert overlap)."""

    def _expert_grid(self):
        """Standard 8-rank LLM grid with expert overlap: tp=cp=dp=2 / ep=etp=edp=2."""
        return HyperCommGrid(
            shape=[2, 2, 2, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            alt_factorizations={
                "expert": {
                    "shape": [2, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )

    def test_world_size_unchanged_with_alt(self):
        """The alt factorization must not inflate world_size."""
        grid = self._expert_grid()
        # 8 ranks, not 8 * 8 = 64.
        assert grid.size == 8

    def test_alt_dim_names_registered(self):
        """Alt dim names should be discoverable for dispatch."""
        grid = self._expert_grid()
        assert grid._dim_to_alt == {"etp": "expert", "ep": "expert", "edp": "expert"}

    def test_alt_only_rank_enum_matches_primary(self):
        """When alt shape == covered primary shape, alt axes enumerate the same rank groups
        as the corresponding primary axes (just under different names).
        """
        grid = self._expert_grid()
        # In primary, tp / cp / dp produce these enumerations:
        primary_tp = grid._gen_rank_enum(["tp"])
        primary_cp = grid._gen_rank_enum(["cp"])
        primary_dp = grid._gen_rank_enum(["dp"])

        # In alt, etp / ep / edp sit at the same positions in the shadow layout.
        etp_enum = grid.get_rank_enum("etp")
        ep_enum = grid.get_rank_enum("ep")
        edp_enum = grid.get_rank_enum("edp")

        assert etp_enum == primary_tp
        assert ep_enum == primary_cp
        assert edp_enum == primary_dp

    def test_alt_multi_dim_rank_enum(self):
        """Multi-dim alt groups (e.g. tp_ep semantically: ['ep', 'etp']) match primary."""
        grid = self._expert_grid()
        # Combined alt group [ep, etp] matches combined primary [cp, tp].
        assert grid.get_rank_enum(["ep", "etp"]) == grid._gen_rank_enum(["cp", "tp"])

    def test_alt_with_shared_dim(self):
        """Combining alt dims with a primary shared (uncovered) dim should work."""
        grid = HyperCommGrid(
            shape=[2, 2, 2, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            alt_factorizations={
                "expert": {
                    "shape": [2, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        # ep + pp should resolve via alt shadow layout.
        # With pp=1, this collapses to the same as ep alone.
        assert grid.get_rank_enum(["ep", "pp"]) == grid.get_rank_enum("ep")

    def test_alt_with_nontrivial_pp(self):
        """Multi-stage PP keeps the alt factorization confined to per-stage rank slabs."""
        os.environ["WORLD_SIZE"] = "16"
        try:
            grid = HyperCommGrid(
                shape=[2, 2, 2, 2],  # tp=2 cp=2 dp=2 pp=2 -> 16 ranks
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2, 2],
                        "dim_names": ["etp", "ep", "edp"],
                        "replaces": ["tp", "cp", "dp"],
                    }
                },
            )
            ep_enum = grid.get_rank_enum("ep")
            # ep groups must only contain ranks within the same PP stage.
            for group in ep_enum:
                stages = {r // 8 for r in group}
                assert len(stages) == 1, (
                    f"ep group {group} crosses PP stages {stages}; expert axes must be confined "
                    f"to a single PP stage's rank slab"
                )
        finally:
            os.environ["WORLD_SIZE"] = "8"

    def test_alt_unequal_shape(self):
        """Alt factorization may differ in shape from primary covers as long as products match."""
        # Primary tp*cp*dp = 2*2*2 = 8. Alt re-factor as ep=4, etp=2, edp=1 (product 8).
        grid = HyperCommGrid(
            shape=[2, 2, 2, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            alt_factorizations={
                "expert": {
                    "shape": [2, 4, 1],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        # ep has size 4, so 2 groups of 4 ranks each (one per edp value, with edp=1 it's one
        # outer group; but let's check structure).
        ep_enum = grid.get_rank_enum("ep")
        # 8 / 4 = 2 groups, each of size 4.
        assert len(ep_enum) == 2
        for group in ep_enum:
            assert len(group) == 4
        # Together they must cover all 8 ranks exactly once.
        flat = [r for grp in ep_enum for r in grp]
        assert sorted(flat) == list(range(8))

    def test_alt_constraint_violated_product_mismatch(self):
        """Mismatched product must raise."""
        with pytest.raises(ValueError, match="product"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2, 4],  # product 16 != 8
                        "dim_names": ["etp", "ep", "edp"],
                        "replaces": ["tp", "cp", "dp"],
                    }
                },
            )

    def test_alt_constraint_violated_non_contiguous_replaces(self):
        """``replaces`` that isn't a contiguous slice of primary dim_names must raise."""
        with pytest.raises(ValueError, match="contiguous"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2],
                        "dim_names": ["etp", "ep"],
                        "replaces": ["tp", "dp"],  # tp and dp skip over cp -> non-contiguous
                    }
                },
            )

    def test_alt_constraint_violated_unknown_cover(self):
        """Cover entries must be primary dim names."""
        with pytest.raises(ValueError, match="not a primary dim"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {"shape": [2], "dim_names": ["ep"], "replaces": ["xx"]}
                },
            )

    def test_alt_dim_name_collision_with_primary(self):
        """Alt dim_names must not collide with primary dim_names."""
        with pytest.raises(ValueError, match="collides with primary"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2, 2],
                        "dim_names": ["tp", "ep", "edp"],  # tp collides
                        "replaces": ["tp", "cp", "dp"],
                    }
                },
            )

    def test_alt_dim_name_collision_across_alts(self):
        """Two alt factorizations must not share dim names."""
        with pytest.raises(ValueError, match="collides with alt"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2, 2],
                        "dim_names": ["etp", "ep", "edp"],
                        "replaces": ["tp", "cp", "dp"],
                    },
                    "second": {
                        "shape": [2, 2, 2],
                        "dim_names": ["etp", "x", "y"],  # etp re-used
                        "replaces": ["tp", "cp", "dp"],
                    },
                },
            )

    def test_create_pg_rejects_mixing_replaced_primary_and_alt(self):
        """tp + ep together is ambiguous; reject at create_pg time."""
        grid = self._expert_grid()
        with pytest.raises(ValueError, match="combine replaced primary"):
            grid.create_pg(["tp", "ep"])

    def test_get_rank_enum_rejects_unknown_dim(self):
        grid = self._expert_grid()
        with pytest.raises(KeyError, match="not a primary or alt dim"):
            grid.get_rank_enum("zz")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_alt_then_get_pg_alt(self, mock_new_subgroups):
        """create_pg for an alt dim, get_pg returns the same group."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = self._expert_grid()
        ep_pg = grid.create_pg("ep")
        assert ep_pg == mock_pg
        assert grid.get_pg("ep") == mock_pg

        # Verify the rank enumeration that was passed to new_subgroups_by_enumeration matches
        # what the primary "cp" would produce, since etp/ep/edp share shapes with tp/cp/dp.
        args, _ = mock_new_subgroups.call_args
        assert args[0] == grid._gen_rank_enum(["cp"])

    def test_alt_full_primary_no_shared_dims(self):
        """Alt covering the full primary (no shared dims) is allowed."""
        grid = HyperCommGrid(
            shape=[2, 2, 2],  # tp cp dp; no pp
            dim_names=["tp", "cp", "dp"],
            alt_factorizations={
                "expert": {
                    "shape": [2, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        assert grid.get_rank_enum("ep") == grid._gen_rank_enum(["cp"])
        assert grid.get_rank_enum(["ep", "etp"]) == grid._gen_rank_enum(["cp", "tp"])

    def test_disjoint_alts_positive(self):
        """Two alt factorizations may replace disjoint slices of the primary."""
        os.environ["WORLD_SIZE"] = "16"
        try:
            grid = HyperCommGrid(
                shape=[2, 2, 2, 2],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2],
                        "dim_names": ["etp", "ep"],
                        "replaces": ["tp", "cp"],
                    },
                    "video": {"shape": [2], "dim_names": ["vdp"], "replaces": ["dp"]},
                },
            )
            # vdp aliases dp; ep aliases cp.
            assert grid.get_rank_enum("vdp") == grid._gen_rank_enum(["dp"])
            assert grid.get_rank_enum("ep") == grid._gen_rank_enum(["cp"])
        finally:
            os.environ["WORLD_SIZE"] = "8"

    def test_disjoint_alts_overlapping_replaces_rejected(self):
        """Two alt factorizations replacing overlapping primary dims must raise."""
        with pytest.raises(ValueError, match="already replaced"):
            HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2],
                        "dim_names": ["etp", "ep"],
                        "replaces": ["tp", "cp"],
                    },
                    "second": {"shape": [2], "dim_names": ["x"], "replaces": ["cp"]},
                },
            )

    def test_resolve_rejects_mixing_two_alt_factorizations(self):
        """Mixing dims from two different alt factorizations at request time must raise."""
        os.environ["WORLD_SIZE"] = "16"
        try:
            grid = HyperCommGrid(
                shape=[2, 2, 2, 2],
                dim_names=["tp", "cp", "dp", "pp"],
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2],
                        "dim_names": ["etp", "ep"],
                        "replaces": ["tp", "cp"],
                    },
                    "video": {"shape": [2], "dim_names": ["vdp"], "replaces": ["dp"]},
                },
            )
            with pytest.raises(ValueError, match="multiple alt factorizations"):
                grid.get_rank_enum(["ep", "vdp"])
        finally:
            os.environ["WORLD_SIZE"] = "8"

    def test_alt_with_rank_offset(self):
        """rank_offset shifts the alt enumeration by the same amount as the primary."""
        os.environ["WORLD_SIZE"] = "16"
        try:
            grid = HyperCommGrid(
                shape=[2, 2, 2, 1],
                dim_names=["tp", "cp", "dp", "pp"],
                rank_offset=8,  # grid lives on ranks 8..15
                alt_factorizations={
                    "expert": {
                        "shape": [2, 2, 2],
                        "dim_names": ["etp", "ep", "edp"],
                        "replaces": ["tp", "cp", "dp"],
                    }
                },
            )
            ep_enum = grid.get_rank_enum("ep")
            cp_enum = grid._gen_rank_enum(["cp"])
            assert ep_enum == cp_enum
            # Every rank in the enumeration must be in [8, 16).
            for group in ep_enum:
                for r in group:
                    assert 8 <= r < 16
        finally:
            os.environ["WORLD_SIZE"] = "8"

    def test_get_rank_enum_multi_axis_alt_via_public_api(self):
        """Multi-axis alt groups must be reachable through the public API."""
        grid = self._expert_grid()
        assert grid.get_rank_enum(["ep", "etp"]) == grid._gen_rank_enum(["cp", "tp"])
        assert grid.get_rank_enum(["edp", "etp"]) == grid._gen_rank_enum(["dp", "tp"])

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_primary_and_alt_have_distinct_keys(self, mock_new_subgroups):
        """Primary 'cp' and alt 'ep' both create groups; the keys must not collide."""
        mock_pg_cp = MagicMock(spec=dist.ProcessGroup)
        mock_pg_ep = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.side_effect = [(mock_pg_cp, None), (mock_pg_ep, None)]

        grid = self._expert_grid()
        cp_pg = grid.create_pg("cp")
        ep_pg = grid.create_pg("ep")

        assert cp_pg == mock_pg_cp
        assert ep_pg == mock_pg_ep
        assert grid.get_pg("cp") == mock_pg_cp
        assert grid.get_pg("ep") == mock_pg_ep
        assert "cp" in grid._pgs
        assert "ep" in grid._pgs


class TestHyperCommGridIntegration:
    """Integration tests for HyperCommGrid with real distributed initialization."""

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            # Initialize PyTorch distributed with NCCL backend
            # This assumes proper environment variables are set (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
            try:
                dist.init_process_group(backend="nccl")
                cls.distributed_initialized = True
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")
        else:
            cls.distributed_initialized = True

    def test_real_distributed_basic_functionality(self):
        """Test basic HyperCommGrid functionality with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size > 8:
            pytest.skip("Test requires at most 8 GPUs")

        # Test with world_size that fits our constraint
        if world_size == 8:
            shape = [2, 2, 2]
            dim_names = ["tp", "cp", "dp"]
        elif world_size == 4:
            shape = [2, 2]
            dim_names = ["tp", "dp"]
        elif world_size == 2:
            shape = [2]
            dim_names = ["tp"]
        else:
            pytest.skip(f"Unsupported world size: {world_size}")

        grid = HyperCommGrid(shape, dim_names, backend="nccl")

        assert grid.size == world_size
        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.backend == "nccl"

    def test_real_distributed_process_group_creation(self):
        """Test process group creation with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create different types of process groups
        tp_pg = grid.create_pg("tp")
        cp_pg = grid.create_pg("cp")
        dp_pg = grid.create_pg("dp")

        # Verify process groups are real PyTorch ProcessGroup objects
        assert isinstance(tp_pg, dist.ProcessGroup)
        assert isinstance(cp_pg, dist.ProcessGroup)
        assert isinstance(dp_pg, dist.ProcessGroup)

        # Verify we can get the process groups back
        assert grid.get_pg("tp") == tp_pg
        assert grid.get_pg("cp") == cp_pg
        assert grid.get_pg("dp") == dp_pg

        # Test process group sizes
        tp_ranks = dist.get_process_group_ranks(tp_pg)
        cp_ranks = dist.get_process_group_ranks(cp_pg)
        dp_ranks = dist.get_process_group_ranks(dp_pg)

        assert len(tp_ranks) == 2  # tp dimension size
        assert len(cp_ranks) == 2  # cp dimension size
        assert len(dp_ranks) == 2  # dp dimension size

    def test_real_distributed_multi_dimensional_groups(self):
        """Test multi-dimensional process group creation with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create multi-dimensional process groups
        tp_cp_pg = grid.create_pg(["tp", "cp"])
        cp_dp_pg = grid.create_pg(["cp", "dp"])

        # Verify process groups are real
        assert isinstance(tp_cp_pg, dist.ProcessGroup)
        assert isinstance(cp_dp_pg, dist.ProcessGroup)

        # Test process group sizes
        tp_cp_ranks = dist.get_process_group_ranks(tp_cp_pg)
        cp_dp_ranks = dist.get_process_group_ranks(cp_dp_pg)

        assert len(tp_cp_ranks) == 4  # tp * cp = 2 * 2
        assert len(cp_dp_ranks) == 4  # cp * dp = 2 * 2

    def test_real_distributed_all_reduce(self):
        """Test actual communication using the created process groups."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create a process group
        tp_pg = grid.create_pg("tp")

        # Create a tensor for communication test
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        tensor = torch.ones(1, device=device) * rank

        # Perform all-reduce within the tensor parallel group
        dist.all_reduce(tensor, group=tp_pg)

        # Verify the result (sum of ranks in the group)
        tp_ranks = dist.get_process_group_ranks(tp_pg)
        expected_sum = sum(tp_ranks)

        assert tensor.item() == expected_sum

    def test_real_distributed_different_world_sizes(self):
        """Test HyperCommGrid with different valid world sizes."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Test configurations for different world sizes
        configs = {
            1: ([1], ["dp"]),
            2: ([2], ["tp"]),
            4: ([2, 2], ["tp", "dp"]),
            8: ([2, 2, 2], ["tp", "cp", "dp"]),
        }

        if world_size not in configs:
            pytest.skip(f"No test configuration for world size {world_size}")

        shape, dim_names = configs[world_size]
        grid = HyperCommGrid(shape, dim_names, backend="nccl")

        assert grid.size == world_size

        # Create and test first dimension process group
        first_dim_pg = grid.create_pg(dim_names[0])
        assert isinstance(first_dim_pg, dist.ProcessGroup)

        # Test communication if world size > 1
        if world_size > 1:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            tensor = torch.tensor([rank], dtype=torch.float, device=device)

            # All-reduce to verify the process group works
            dist.all_reduce(tensor, group=first_dim_pg)

            # Verify the result
            group_ranks = dist.get_process_group_ranks(first_dim_pg)
            expected_sum = sum(group_ranks)
            assert tensor.item() == expected_sum

    def test_real_distributed_error_handling(self):
        """Test error handling with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size > 8:
            pytest.skip("Test requires at most 8 GPUs")

        # Test shape validation with real world size
        if world_size == 8:
            # This should work
            grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])
            assert grid.size == 8

            # This should fail - too large for world size
            with pytest.raises(RuntimeError, match="Grid shape.*is over sized"):
                HyperCommGrid([4, 4], ["tp", "dp"])  # 16 > 8

        # Test duplicate process group creation
        if world_size >= 2:
            grid = HyperCommGrid([2, world_size // 2], ["tp", "dp"])
            grid.create_pg("tp")

            with pytest.raises(KeyError, match="Process group.*has already been created"):
                grid.create_pg("tp")

    def test_real_distributed_alt_factorization_overlap(self):
        """Verify that alt-factorization expert groups live on the same ranks as the
        primary tp/cp/dp groups (NMFW-464 overlap)."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid(
            shape=[2, 2, 2, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            backend="nccl",
            alt_factorizations={
                "expert": {
                    "shape": [2, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )

        # Build matching groups under each factorization.
        tp_pg = grid.create_pg("tp")
        cp_pg = grid.create_pg("cp")
        dp_pg = grid.create_pg("dp")
        etp_pg = grid.create_pg("etp")
        ep_pg = grid.create_pg("ep")
        edp_pg = grid.create_pg("edp")

        # Each pair must enumerate the SAME physical ranks for the current process.
        assert dist.get_process_group_ranks(tp_pg) == dist.get_process_group_ranks(
            etp_pg
        ), "etp must alias tp ranks under expert overlap"
        assert dist.get_process_group_ranks(cp_pg) == dist.get_process_group_ranks(
            ep_pg
        ), "ep must alias cp ranks under expert overlap"
        assert dist.get_process_group_ranks(dp_pg) == dist.get_process_group_ranks(
            edp_pg
        ), "edp must alias dp ranks under expert overlap"

        # Sanity: communication actually works on the alt group (all-reduce within ep).
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        tensor = torch.tensor([rank], dtype=torch.float, device=device)
        dist.all_reduce(tensor, group=ep_pg)
        ep_ranks = dist.get_process_group_ranks(ep_pg)
        assert tensor.item() == sum(ep_ranks)

    def test_real_distributed_alt_factorization_pp_confined(self):
        """With PP>1, alt-factorization expert groups must stay within a PP stage's rank slab."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        # tp=2 cp=1 dp=2 pp=2 -> 8 ranks; per-PP-stage slab has 4 ranks.
        grid = HyperCommGrid(
            shape=[2, 1, 2, 2],
            dim_names=["tp", "cp", "dp", "pp"],
            backend="nccl",
            alt_factorizations={
                "expert": {
                    "shape": [2, 1, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        edp_pg = grid.create_pg("edp")
        edp_ranks = dist.get_process_group_ranks(edp_pg)
        # All ranks in the edp group must share a PP stage (ranks 0-3 or 4-7).
        stages = {r // 4 for r in edp_ranks}
        assert len(stages) == 1, f"edp group {edp_ranks} crosses PP stages {stages}"

    def test_real_distributed_rank_enumeration_verification(self):
        """Verify rank enumeration produces correct communication patterns."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Test that ranks in the same TP group can communicate
        tp_pg = grid.create_pg("tp")
        tp_ranks = dist.get_process_group_ranks(tp_pg)

        current_rank = dist.get_rank()
        if current_rank in tp_ranks:
            device = torch.device(f"cuda:{current_rank % torch.cuda.device_count()}")

            # Create a unique tensor based on rank
            tensor = torch.tensor([current_rank], dtype=torch.float, device=device)
            original_value = tensor.clone()

            # All-reduce within TP group
            dist.all_reduce(tensor, group=tp_pg)

            # Verify the sum is correct
            expected_sum = sum(tp_ranks)
            assert tensor.item() == expected_sum
