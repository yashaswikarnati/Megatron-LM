# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
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

    def test_from_hyper_comm_grid_reads_required_groups(self, mocker):
        """Test mapping from an extended HyperCommGrid to ProcessGroupCollection."""
        grid = mocker.Mock()
        grid.dim_names = ["tp", "cp", "dp", "pp"]

        pgs = {}

        def pg_for(dims):
            key = tuple(dims) if isinstance(dims, list) else dims
            pgs.setdefault(key, mocker.Mock(spec=dist.ProcessGroup))
            return pgs[key]

        grid.get_pg.side_effect = pg_for
        grid.get_alias_dims.return_value = ["expt_tp", "ep", "pp"]

        collection = ProcessGroupCollection.from_hyper_comm_grid(
            grid,
            required_pgs=['tp', 'pp', 'dp', 'dp_cp', 'mp', 'expt_dp', 'tp_ep_pp', 'intra_dist_opt'],
        )

        assert collection.tp is pgs['tp']
        assert collection.pp is pgs['pp']
        assert collection.dp is pgs['dp']
        assert collection.dp_cp is pgs[('dp', 'cp')]
        assert collection.mp is pgs[('tp', 'pp')]
        assert collection.expt_dp is pgs['expt_dp']
        assert collection.tp_ep_pp is pgs['tp_ep_pp']
        assert collection.intra_dist_opt is pgs[('tp', 'cp', 'dp', 'pp')]
        assert collection.intra_dp_cp is collection.dp_cp
        assert collection.intra_expt_dp is collection.expt_dp
        assert collection.inter_dist_opt is None

    def test_from_hyper_comm_grid_rejects_multi_instance_distopt(self, mocker):
        """Phase 1 helper does not support multiple distributed optimizer instances."""
        grid = mocker.Mock()
        with pytest.raises(ValueError, match="num_distributed_optimizer_instances == 1"):
            ProcessGroupCollection.from_hyper_comm_grid(grid, num_distributed_optimizer_instances=2)

    def test_from_hyper_comm_grid_creates_from_real_extended_grid(self, mocker, monkeypatch):
        """Test helper against real HyperCommGrid alias resolution without distributed init."""
        monkeypatch.setenv("WORLD_SIZE", "16")
        mocker.patch('torch.distributed.get_rank', return_value=0)
        mock_new_subgroups = mocker.patch('torch.distributed.new_subgroups_by_enumeration')

        created = []

        def make_pg(rank_enum, **_kwargs):
            pg = mocker.Mock(spec=dist.ProcessGroup)
            pg.size.return_value = len(rank_enum[0])
            created.append((rank_enum, pg))
            return pg, None

        mock_new_subgroups.side_effect = make_pg

        grid = HyperCommGrid([2, 1, 4, 2], ["tp", "cp", "dp", "pp"])
        grid.register_layout(
            "expert",
            [1, 4, 2, 2],
            ["expt_tp", "ep", "expt_dp", "pp"],
            aliases={"tp_ep": ["expt_tp", "ep"], "tp_ep_pp": ["expt_tp", "ep", "pp"]},
        )

        collection = ProcessGroupCollection.from_hyper_comm_grid(
            grid,
            create=True,
            required_pgs=[
                'tp',
                'dp',
                'dp_cp',
                'mp',
                'ep',
                'expt_tp',
                'expt_dp',
                'tp_ep',
                'tp_ep_pp',
                'intra_dist_opt',
            ],
        )

        assert collection.tp is grid.get_pg("tp")
        assert collection.dp is grid.get_pg("dp")
        assert collection.dp_cp is grid.get_pg(["dp", "cp"])
        assert collection.mp is grid.get_pg(["tp", "pp"])
        assert collection.ep is grid.get_pg("ep")
        assert collection.expt_tp is grid.get_pg("expt_tp")
        assert collection.expt_dp is grid.get_pg("expt_dp")
        assert collection.tp_ep is grid.get_pg("tp_ep")
        assert collection.tp_ep_pp is grid.get_pg("tp_ep_pp")
        assert collection.intra_dist_opt is grid.get_pg(["tp", "cp", "dp", "pp"])
        assert collection.intra_dp_cp is collection.dp_cp
        assert collection.intra_expt_dp is collection.expt_dp
        assert collection.inter_dist_opt is None

    def test_from_hyper_comm_grid_rejects_tp_ep_pp_without_shared_pp(self, monkeypatch):
        """tp_ep_pp must include the same pp dimension used by the base layout."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        grid = HyperCommGrid([2, 2], ["tp", "pp"])
        grid.register_layout(
            "expert", [2, 2], ["ep", "expert_pp"], aliases={"tp_ep_pp": ["ep", "expert_pp"]}
        )

        with pytest.raises(ValueError, match="shared pipeline dimension 'pp'"):
            ProcessGroupCollection.from_hyper_comm_grid(
                grid, create=True, required_pgs=['tp_ep_pp']
            )


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
