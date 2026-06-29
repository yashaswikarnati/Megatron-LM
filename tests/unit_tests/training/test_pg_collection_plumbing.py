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


def test_training_entrypoints_expose_only_pg_collection():
    for entrypoint in (training_mod.pretrain, training_mod.train, training_mod.train_step):
        parameters = inspect.signature(entrypoint).parameters
        assert "pg_collection" in parameters
        assert "schedule_pg_collection" not in parameters


@pytest.mark.parametrize(
    ("carrier", "communicator"),
    [
        (_multi_carrier(), _multimodule_communicator()),
        (ProcessGroupCollection(), _ordinary_communicator()),
        (None, None),
    ],
)
def test_resolver_accepts_matching_pairs(carrier, communicator):
    assert (
        training_mod._resolve_pipeline_communicator(carrier, communicator, SimpleNamespace())
        is communicator
    )


def test_plain_carrier_builds_ordinary_communicator_from_exact_inputs():
    carrier = ProcessGroupCollection(pp="pp")
    config = SimpleNamespace()

    with mock.patch.object(training_mod, "P2PCommunicator") as communicator_cls:
        communicator = training_mod._resolve_pipeline_communicator(carrier, None, config)

    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(pp_group=carrier.pp, config=config)


@pytest.mark.parametrize(
    ("carrier", "communicator", "message"),
    [
        (_multi_carrier(), None, "MultiModulePipelineCommunicator"),
        (_multi_carrier(), _ordinary_communicator(), "MultiModulePipelineCommunicator"),
        (
            ProcessGroupCollection(),
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
