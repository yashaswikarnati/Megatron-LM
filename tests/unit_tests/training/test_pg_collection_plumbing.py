# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for process-group and pipeline-communicator training API plumbing."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import training as training_mod


def _multi_carrier() -> MultiModuleProcessGroupCollection:
    return MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection()},
        loss_module_name="language",
        module_order=("vision", "language"),
    )


def _ordinary_communicator() -> P2PCommunicator:
    return object.__new__(P2PCommunicator)


def _multimodule_communicator() -> MultiModulePipelineCommunicator:
    return object.__new__(MultiModulePipelineCommunicator)


def _plain_carrier(**overrides) -> ProcessGroupCollection:
    groups = {"pp": "pp", "tp": "tp", "cp": "cp"}
    groups.update(overrides)
    return ProcessGroupCollection(**groups)


def test_training_entrypoints_expose_only_pg_collection():
    for entrypoint in (training_mod.pretrain, training_mod.train, training_mod.train_step):
        parameters = inspect.signature(entrypoint).parameters
        assert "pg_collection" in parameters
        assert "schedule_pg_collection" not in parameters


def test_evaluation_entrypoints_expose_optional_exact_pair():
    for entrypoint in (training_mod.evaluate, training_mod.evaluate_and_print_results):
        parameters = inspect.signature(entrypoint).parameters
        assert parameters["pg_collection"].default is None
        assert parameters["p2p_communicator"].default is None


@pytest.mark.parametrize(
    ("entrypoint", "expected_calls"),
    ((training_mod.pretrain, 2), (training_mod.train, 1)),
)
def test_training_callers_forward_exact_pair_to_evaluation(entrypoint, expected_calls):
    tree = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_and_print_results"
    ]

    assert len(calls) == expected_calls
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["pg_collection"], ast.Name)
        assert keywords["pg_collection"].id == "pg_collection"
        assert isinstance(keywords["p2p_communicator"], ast.Name)
        assert keywords["p2p_communicator"].id == "p2p_communicator"


def test_pretrain_forwards_exact_carrier_to_data_iterator_builders():
    tree = ast.parse(textwrap.dedent(inspect.getsource(training_mod.pretrain)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_train_valid_test_data_iterators"
    ]

    assert len(calls) == 2
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["pg_collection"], ast.Name)
        assert keywords["pg_collection"].id == "pg_collection"


@pytest.mark.parametrize(
    "entrypoint",
    (
        training_mod.setup_model_and_optimizer,
        training_mod.train,
        training_mod.save_checkpoint_and_time,
    ),
)
def test_direct_checkpoint_calls_forward_only_the_union_carrier(entrypoint):
    tree = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"save_checkpoint", "load_checkpoint"}
    ]

    assert calls
    explicit_group_names = {
        "tp_group",
        "pp_group",
        "dp_group",
        "dp_cp_group",
        "expt_dp_group",
    }
    for call in calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "pg_collection" in keyword_names
        assert keyword_names.isdisjoint(explicit_group_names)


@pytest.mark.parametrize(
    ("carrier", "communicator"),
    [
        (_multi_carrier(), _multimodule_communicator()),
        (_plain_carrier(), _ordinary_communicator()),
        (None, None),
    ],
)
def test_resolver_accepts_matching_pairs(carrier, communicator):
    assert (
        training_mod._resolve_pipeline_communicator(carrier, communicator, SimpleNamespace())
        is communicator
    )


def test_plain_carrier_builds_ordinary_communicator_from_exact_inputs():
    carrier = _plain_carrier()
    config = SimpleNamespace()

    with mock.patch.object(training_mod, "P2PCommunicator") as communicator_cls:
        communicator = training_mod._resolve_pipeline_communicator(carrier, None, config)

    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(pp_group=carrier.pp, config=config)


@pytest.mark.parametrize("missing_group", ["pp", "tp", "cp"])
@pytest.mark.parametrize("communicator", [None, _ordinary_communicator()])
def test_plain_carrier_requires_scheduling_groups(missing_group, communicator):
    carrier = _plain_carrier(**{missing_group: None})

    with pytest.raises(ValueError, match=rf"non-None {missing_group}"):
        training_mod._resolve_pipeline_communicator(
            carrier, communicator, SimpleNamespace()
        )


@pytest.mark.parametrize(
    "communicator",
    [None, _ordinary_communicator(), _multimodule_communicator()],
)
def test_resolver_rejects_unknown_carrier_before_communicator_work(communicator):
    with pytest.raises(TypeError, match="pg_collection"):
        training_mod._resolve_pipeline_communicator(
            object(), communicator, SimpleNamespace()
        )


@pytest.mark.parametrize(
    ("carrier", "communicator", "message"),
    [
        (_multi_carrier(), None, "MultiModulePipelineCommunicator"),
        (_multi_carrier(), _ordinary_communicator(), "MultiModulePipelineCommunicator"),
        (
            _plain_carrier(),
            _multimodule_communicator(),
            "MultiModuleProcessGroupCollection",
        ),
        (None, _multimodule_communicator(), "MultiModuleProcessGroupCollection"),
        (None, _ordinary_communicator(), "explicit communicator"),
    ],
)
def test_resolver_rejects_mismatched_pairs(carrier, communicator, message):
    with pytest.raises(ValueError, match=message):
        training_mod._resolve_pipeline_communicator(carrier, communicator, SimpleNamespace())


def test_multimodule_pretrain_skips_process_global_seed_bootstrap():
    class BootstrapStopped(Exception):
        pass

    captured = {}

    def capture_initialize_kwargs(**kwargs):
        captured.update(kwargs)
        raise BootstrapStopped

    with (
        mock.patch.object(training_mod.ft_integration, "setup"),
        mock.patch.object(
            training_mod,
            "initialize_megatron",
            side_effect=capture_initialize_kwargs,
        ),
        pytest.raises(BootstrapStopped),
    ):
        training_mod.pretrain(
            SimpleNamespace(),
            train_valid_test_dataset_provider=None,
            model_type=None,
            forward_step_func=None,
            pg_collection=_multi_carrier(),
        )

    assert captured["skip_random_seed"] is True
    assert all(
        captured[name] is None
        for name in (
            "seed_pp_group",
            "seed_dp_group",
            "seed_tp_group",
            "seed_ep_group",
            "seed_etp_group",
        )
    )


def test_multimodule_only_local_item_preserves_its_name_and_collection():
    vision = ProcessGroupCollection()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"vision": vision},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    assert carrier.get_only_local_item() == ("vision", vision)


def test_multimodule_only_local_item_rejects_multiple_modules_without_selecting_loss_owner():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={
            "vision": ProcessGroupCollection(),
            "language": ProcessGroupCollection(),
        },
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    with pytest.raises(ValueError, match="exactly one local module"):
        carrier.get_only_local_item()
