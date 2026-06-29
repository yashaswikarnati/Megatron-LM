# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for process-group and pipeline-communicator training API plumbing."""

import inspect
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
