# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction and real-distributed tests for the hetero MIMO grid topology.

Layout under test: encoder grid tp=2,dp=2 at ranks 0-3, language grid tp=2,pp=2 at ranks 4-7.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import examples.mimo.training.topology as topology_module
from examples.mimo.training.topology import (
    HeteroTopology,
    ModuleGridSpec,
    _validate_grid_layout,
    create_topology,
)
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.parallel_state import default_embedding_ranks
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


def _mock_grid(mocker, rank_offset, size, pp_rank_groups=()):
    grid = mocker.Mock()
    grid.rank_offset = rank_offset
    grid.size = size
    grid.get_rank_enum.return_value = [list(ranks) for ranks in pp_rank_groups]
    return grid


def _specs():
    return [
        ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
        ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4),
    ]


def test_hetero_topology_exposes_only_pg_collection_field():
    fields = set(HeteroTopology.__dataclass_fields__)

    assert "pg_collection" in fields
    assert "schedule_pg_collection" not in fields
    assert not hasattr(HeteroTopology, "schedule_pg_collection")


def test_create_topology_rejects_duplicate_module_names_before_grid_build(mocker):
    build_grid = mocker.patch.object(topology_module, "_build_grid")
    specs = [
        ModuleGridSpec(name="vision", num_ranks=4, tp=2),
        ModuleGridSpec(name="vision", num_ranks=4, tp=2),
        ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2),
    ]

    with pytest.raises(ValueError, match="duplicate module names"):
        create_topology(specs)

    build_grid.assert_not_called()


def test_create_topology_preserves_distinct_module_resources(mocker):
    specs = [
        ModuleGridSpec(name="vision", num_ranks=4, tp=2),
        ModuleGridSpec(name="audio", num_ranks=4, tp=2),
        ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, rank_offset=4),
    ]
    vision_grid = mocker.Mock(name="vision_grid")
    audio_grid = mocker.Mock(name="audio_grid")
    language_grid = mocker.Mock(name="language_grid")
    for grid in (vision_grid, audio_grid):
        grid.is_current_rank_in_grid.return_value = True
    language_grid.is_current_rank_in_grid.return_value = False
    build_grid = mocker.patch.object(
        topology_module, "_build_grid", side_effect=[vision_grid, audio_grid, language_grid]
    )
    mocker.patch.object(topology_module, "_validate_grid_layout")

    vision_pgs = ProcessGroupCollection()
    audio_pgs = ProcessGroupCollection()
    language_pgs = ProcessGroupCollection()
    for pgc in (vision_pgs, audio_pgs, language_pgs):
        pgc.embd = None
        pgc.pos_embd = None
    mocker.patch.object(
        topology_module,
        "pg_collection_from_grid",
        side_effect=[vision_pgs, audio_pgs, language_pgs],
    )

    topology = create_topology(specs)
    try:
        assert [call.args[0] for call in build_grid.call_args_list] == specs
        assert topology.grids["vision"] is vision_grid
        assert topology.grids["audio"] is audio_grid
        assert topology.grids[MIMO_LANGUAGE_MODULE_KEY] is language_grid
        assert topology.module_pgs["vision"] is vision_pgs
        assert topology.module_pgs["audio"] is audio_pgs
        assert topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY] is language_pgs
        assert topology.pg_collection.module_order == (
            "vision",
            "audio",
            MIMO_LANGUAGE_MODULE_KEY,
        )
        assert list(topology.pg_collection.keys()) == ["vision", "audio"]
        assert topology.pg_collection["vision"] is vision_pgs
        assert topology.pg_collection["audio"] is audio_pgs
    finally:
        topology.destroy()


def test_grid_layout_allows_identical_encoder_spans_disjoint_from_language(mocker):
    grids = {
        "vision": _mock_grid(mocker, 0, 4),
        "audio": _mock_grid(mocker, 0, 4),
        MIMO_LANGUAGE_MODULE_KEY: _mock_grid(mocker, 4, 4, ((4, 6), (5, 7))),
    }
    mocker.patch.object(topology_module.dist, "get_world_size", return_value=8)
    mocker.patch.object(
        topology_module.RankRole,
        "build",
        return_value=SimpleNamespace(mode=ModuleLayout.NON_COLOCATED),
    )

    _validate_grid_layout(grids)


def test_grid_layout_rejects_partial_encoder_overlap(mocker):
    grids = {
        "vision": _mock_grid(mocker, 0, 4),
        "audio": _mock_grid(mocker, 2, 4),
        MIMO_LANGUAGE_MODULE_KEY: _mock_grid(mocker, 6, 2, ((6, 7),)),
    }
    mocker.patch.object(topology_module.dist, "get_world_size", return_value=8)

    with pytest.raises(ValueError, match="encoder grids must be identical or disjoint"):
        _validate_grid_layout(grids)


def test_grid_layout_accepts_world_last_on_language_terminal_stage(mocker):
    grids = {
        "vision": _mock_grid(mocker, 0, 4),
        MIMO_LANGUAGE_MODULE_KEY: _mock_grid(mocker, 4, 4, ((4, 6), (5, 7))),
    }
    mocker.patch.object(topology_module.dist, "get_world_size", return_value=8)
    mocker.patch.object(
        topology_module.RankRole,
        "build",
        return_value=SimpleNamespace(mode=ModuleLayout.NON_COLOCATED),
    )

    _validate_grid_layout(grids)


def test_grid_layout_rejects_world_last_outside_language_terminal_stage(mocker):
    grids = {
        MIMO_LANGUAGE_MODULE_KEY: _mock_grid(mocker, 0, 4, ((0, 2), (1, 3))),
        "vision": _mock_grid(mocker, 4, 4),
    }
    mocker.patch.object(topology_module.dist, "get_world_size", return_value=8)

    with pytest.raises(ValueError, match="world-last rank 7.*terminal pipeline stage"):
        _validate_grid_layout(grids)


def test_build_multi_module_pg_collection_filters_in_canonical_order(mocker):
    vision_grid = mocker.Mock()
    audio_grid = mocker.Mock()
    language_grid = mocker.Mock()
    for grid in (vision_grid, audio_grid):
        grid.shape = (2, 2)
        grid.rank_offset = 0
        grid.size = 4
        grid.is_current_rank_in_grid.return_value = True
    language_grid.is_current_rank_in_grid.return_value = False

    vision_pgs = ProcessGroupCollection()
    audio_pgs = ProcessGroupCollection()
    language_pgs = ProcessGroupCollection()
    grids = {"vision": vision_grid, "audio": audio_grid, "language": language_grid}
    module_pgs = {"language": language_pgs, "audio": audio_pgs, "vision": vision_pgs}

    collection = topology_module.build_multi_module_pg_collection(
        grids,
        module_pgs,
        loss_module_name="language",
        module_order=("vision", "audio", "language"),
    )

    assert collection.module_order == ("vision", "audio", "language")
    assert collection.loss_module_name == "language"
    assert list(collection.keys()) == ["vision", "audio"]
    assert collection["vision"] is vision_pgs
    assert collection["audio"] is audio_pgs
    assert collection["vision"] is not collection["audio"]


@pytest.mark.parametrize(
    ("grid_names", "pg_names", "module_order"),
    [
        (("vision",), ("vision", "language"), ("vision", "language")),
        (("vision", "language"), ("vision",), ("vision", "language")),
        (("vision", "language"), ("vision", "language"), ("vision",)),
    ],
)
def test_build_multi_module_pg_collection_rejects_module_name_mismatch(
    mocker, grid_names, pg_names, module_order
):
    grids = {name: mocker.Mock() for name in grid_names}
    for grid in grids.values():
        grid.is_current_rank_in_grid.return_value = True
    module_pgs = {name: ProcessGroupCollection() for name in pg_names}

    with pytest.raises(ValueError, match="same declared module names"):
        topology_module.build_multi_module_pg_collection(
            grids,
            module_pgs,
            loss_module_name="language",
            module_order=module_order,
        )


class TestModuleGridSpecResolution:
    def test_derived_dims_resolve_to_concrete_ints(self):
        # num_ranks=4,tp=2 with default expt_tp=1: dp=2, expt_dp=4.
        spec = ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2)
        assert isinstance(spec.dp, int) and spec.dp == 2
        assert spec.expt_tp == 1
        assert isinstance(spec.expt_dp, int) and spec.expt_dp == 4

    def test_explicit_expert_dims_resolve_correctly(self):
        # num_ranks=4,tp=2,ep=2,expt_tp=2: expt_dp = 4//(2*2*1) = 1.
        spec = ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, ep=2, expt_tp=2)
        assert isinstance(spec.expt_tp, int) and spec.expt_tp == 2
        assert isinstance(spec.expt_dp, int) and spec.expt_dp == 1

    def test_indivisible_dense_raises(self):
        with pytest.raises(ValueError):
            ModuleGridSpec(name=ENCODER, num_ranks=4, tp=3)

    def test_indivisible_expert_raises(self):
        with pytest.raises(ValueError):
            ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, ep=3, expt_tp=2)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestHeteroTopology:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_grids_partition_world(self):
        topo = create_topology(_specs())
        try:
            encoder_ranks = set(range(0, 4))
            llm_ranks = set(range(4, 8))
            assert encoder_ranks & llm_ranks == set()
            assert encoder_ranks | llm_ranks == set(range(dist.get_world_size()))
            assert topo.grids[ENCODER].rank_offset == 0
            assert topo.grids[MIMO_LANGUAGE_MODULE_KEY].rank_offset == 4
        finally:
            topo.destroy()

    def test_exposes_ordered_pg_collection(self):
        specs = _specs()
        topo = create_topology(specs)
        try:
            assert topo.pg_collection.module_order == tuple(spec.name for spec in specs)
            assert topo.pg_collection.loss_module_name == MIMO_LANGUAGE_MODULE_KEY
            assert list(topo.pg_collection.keys()) == [
                spec.name
                for spec in specs
                if topo.grids[spec.name].is_current_rank_in_grid()
            ]
        finally:
            topo.destroy()

    def test_pgc_group_sizes(self):
        topo = create_topology(_specs())
        try:
            rank = dist.get_rank()
            if rank < 4:
                pgc = topo.module_pgs[ENCODER]
                assert pgc.tp.size() == 2
                assert pgc.pp.size() == 1
                assert pgc.dp.size() == 2
                assert pgc.dp_cp.size() == 2
            else:
                pgc = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                assert pgc.tp.size() == 2
                assert pgc.pp.size() == 2
                assert pgc.dp.size() == 1
                assert pgc.dp_cp.size() == 1
        finally:
            topo.destroy()

    def test_embedding_groups(self):
        # Language grid is tp=2,pp=2 at ranks 4-7: each PP group is [first,last] (size 2),
        # so first/last-stage ranks get a 2-rank .embd and the first stage gets .pos_embd.
        topo = create_topology(_specs())
        try:
            rank = dist.get_rank()
            if rank < 4:
                pgc = topo.module_pgs[ENCODER]
                assert pgc.embd is None
                assert pgc.pos_embd is None
            else:
                pgc = topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
                pp_ranks = dist.get_process_group_ranks(pgc.pp)
                pp_rank = pgc.pp.rank()
                expected_embd = len(default_embedding_ranks(pp_ranks))
                assert pgc.embd is not None
                assert pgc.embd.size() == expected_embd
                if pp_rank == 0:
                    assert pgc.pos_embd is not None
                    assert pgc.pos_embd.size() == 1
                else:
                    assert pgc.pos_embd is None
        finally:
            topo.destroy()

    def test_validate_rejects_overlapping_not_equal(self):
        # Illegal: encoder spans 0-3, llm spans 2-5 (overlap, not equal, not disjoint).
        a = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=0, backend="nccl")
        b = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=2, backend="nccl")
        try:
            with pytest.raises(ValueError, match="disjoint"):
                _validate_grid_layout({ENCODER: a, MIMO_LANGUAGE_MODULE_KEY: b})
        finally:
            a.destroy()
            b.destroy()

    def test_validate_rejects_gap_in_world_coverage(self):
        # Illegal: encoder spans 0-3, llm spans 4-7 leaves nothing uncovered, so instead
        # use disjoint grids that fail to span the full 8-rank world (ranks 6-7 uncovered).
        a = HyperCommGrid([2, 1], ["tp", "dp"], rank_offset=0, backend="nccl")
        b = HyperCommGrid([2, 1], ["tp", "dp"], rank_offset=4, backend="nccl")
        try:
            with pytest.raises(ValueError, match="partition the world"):
                _validate_grid_layout({ENCODER: a, MIMO_LANGUAGE_MODULE_KEY: b})
        finally:
            a.destroy()
            b.destroy()
