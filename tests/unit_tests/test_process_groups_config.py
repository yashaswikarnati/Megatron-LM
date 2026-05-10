# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


class TestProcessGroupsConfig:
    """Simple tests for process group dataclasses."""

    def test_transformer_process_groups(self, mocker):
        """Test basic functionality of TransformerProcessGroups."""
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        model_pgs.tp = mock_pg1
        model_pgs.pp = mock_pg2

        # Test accessing attributes
        assert model_pgs.tp == mock_pg1
        assert model_pgs.pp == mock_pg2

        # Test attribute existence
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert not hasattr(model_pgs, 'cp')  # Not set yet

    def test_grad_comm_process_groups(self, mocker):
        """Test basic functionality of ProcessGroupCollection."""
        # Create mock process groups
        mock_pg = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        grad_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        grad_pgs.dp = mock_pg

        # Test accessing attributes
        assert grad_pgs.dp == mock_pg

        # Test attribute existence
        assert hasattr(grad_pgs, 'dp')
        assert not hasattr(grad_pgs, 'dp_cp')  # Not set yet

    def test_hierarchical_context_parallel_groups(self, mocker):
        """Test setting and accessing the hierarchical context parallel list."""
        # Create mock process groups
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Set the hierarchical context parallel groups
        model_pgs.hcp = [mock_pg1, mock_pg2]

        # Test list access
        assert isinstance(model_pgs.hcp, list)
        assert len(model_pgs.hcp) == 2
        assert model_pgs.hcp[0] == mock_pg1
        assert model_pgs.hcp[1] == mock_pg2

    def test_repr(self, mocker):
        """Test __repr__ shows active process groups and their sizes."""
        tp_size = 4
        pp_size = 2
        mock_tp = mocker.Mock(spec=dist.ProcessGroup)
        mock_tp.size.return_value = tp_size
        mock_pp = mocker.Mock(spec=dist.ProcessGroup)
        mock_pp.size.return_value = pp_size

        # Test empty collection
        empty_pgs = ProcessGroupCollection()
        assert repr(empty_pgs) == "ProcessGroupCollection(empty)"

        # Test collection with process groups
        model_pgs = ProcessGroupCollection()
        model_pgs.tp = mock_tp
        model_pgs.pp = mock_pp

        repr_str = repr(model_pgs)
        assert "ProcessGroupCollection(" in repr_str
        assert f"tp({tp_size})" in repr_str
        assert f"pp({pp_size})" in repr_str

    def test_repr_with_list_process_groups(self, mocker):
        """Test __repr__ handles list-typed process groups like hcp."""
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg1.size.return_value = 2
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2.size.return_value = 4

        model_pgs = ProcessGroupCollection()
        model_pgs.hcp = [mock_pg1, mock_pg2]

        repr_str = repr(model_pgs)
        assert "ProcessGroupCollection(" in repr_str
        assert "hcp([2, 4])" in repr_str


class TestPGConfigDefaultInitialization:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_initialization(self):
        """Test default initialization of ProcessGroupCollection."""
        # Create instance
        model_pgs = ProcessGroupCollection.use_mpu_process_groups()

        # Test that instance was created successfully
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'dp')
        assert hasattr(model_pgs, 'dp_cp')

        # Test that only required process groups were initialized
        model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'cp'])
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'cp')
        assert not hasattr(model_pgs, 'dp')

        # Test that an error is raised if an invalid process group is requested
        with pytest.raises(ValueError, match=r"Invalid process groups requested"):
            model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'foo'])


class TestPGConfigFromHyperCommGrid:
    """Build ProcessGroupCollection from a HyperCommGrid (no parallel_state init).

    Uses real distributed groups via NCCL on whatever world the runner provides.
    """

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")

    def test_from_hyper_comm_grid_dense_8gpu(self):
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")
        if dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")

        from megatron.core.hyper_comm_grid import HyperCommGrid

        grid = HyperCommGrid(shape=[2, 1, 4, 1], dim_names=["tp", "cp", "dp", "pp"], backend="nccl")
        pg = ProcessGroupCollection.from_hyper_comm_grid(grid)
        assert hasattr(pg, "tp") and pg.tp.size() == 2
        assert hasattr(pg, "cp") and pg.cp.size() == 1
        assert hasattr(pg, "dp") and pg.dp.size() == 4
        assert hasattr(pg, "pp") and pg.pp.size() == 1
        assert hasattr(pg, "dp_cp") and pg.dp_cp.size() == 4
        # No alt factorization -> expert fields are set to ``None`` (so callers like
        # DDP's ``hasattr`` check pass uniformly without distinguishing MoE-grid
        # from non-MoE-grid).
        assert pg.ep is None
        assert pg.expt_tp is None
        assert pg.expt_dp is None

    def test_from_hyper_comm_grid_with_expert_alt_8gpu(self):
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")
        if dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")

        from megatron.core.hyper_comm_grid import HyperCommGrid

        grid = HyperCommGrid(
            shape=[2, 1, 4, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            backend="nccl",
            alt_factorizations={
                "expert": {
                    "shape": [1, 4, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        pg = ProcessGroupCollection.from_hyper_comm_grid(grid)
        # Standard fields populated.
        assert pg.tp.size() == 2
        assert pg.dp.size() == 4
        # Expert fields populated from alt factorization.
        assert pg.ep.size() == 4
        assert pg.expt_tp.size() == 1
        assert pg.expt_dp.size() == 2
        # Combined expert groups.
        assert pg.tp_ep.size() == 4  # etp(1) * ep(4)
        # Structural invariant: ep ranks must live entirely within this rank's
        # primary tp*cp*dp slab (i.e. within the same PP stage). With pp=1, that's
        # the full world; the meaningful check is that the size matches.
        assert len(dist.get_process_group_ranks(pg.ep)) == 4
