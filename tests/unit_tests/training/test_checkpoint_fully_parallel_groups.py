# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for explicit process groups in distributed checkpoint loading."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import checkpointing


def _checkpoint_collection(label: str) -> ProcessGroupCollection:
    return ProcessGroupCollection(
        tp=f"{label}-tp",
        pp=f"{label}-pp",
        dp=f"{label}-dp",
        dp_cp=f"{label}-dp-cp",
        expt_dp=f"{label}-expt-dp",
    )


def test_checkpoint_resolver_preserves_none_for_stock_mpu_fallback():
    assert checkpointing._resolve_checkpoint_pg_collection(None) == (None, "")


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_multimodule_checkpoint_rejects_nontransparent_global_formats(ckpt_format):
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"vision": _checkpoint_collection("vision")},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    with pytest.raises(ValueError, match="does not preserve module-namespaced logical state"):
        checkpointing._validate_checkpoint_format_for_pg_collection(carrier, ckpt_format)


def test_multimodule_checkpoint_accepts_missing_torch_dist_and_local_checkpointing():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"vision": _checkpoint_collection("vision")},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    checkpointing._validate_checkpoint_format_for_pg_collection(carrier, None)
    checkpointing._validate_checkpoint_format_for_pg_collection(carrier, "torch_dist")
    checkpointing._validate_checkpoint_format_for_pg_collection(carrier, "torch", is_local=True)


@pytest.mark.parametrize("carrier_kind", ["plain", "multimodule"])
def test_checkpoint_resolver_uses_the_exact_caller_collection(carrier_kind):
    local = _checkpoint_collection("local")
    carrier = local
    expected_prefix = ""
    if carrier_kind == "multimodule":
        carrier = MultiModuleProcessGroupCollection(
            module_pgs={"vision": local},
            loss_module_name="language",
            module_order=("vision", "language"),
        )
        expected_prefix = "mimo.vision."

    resolved, key_prefix = checkpointing._resolve_checkpoint_pg_collection(carrier)

    assert resolved is local
    assert key_prefix == expected_prefix


@pytest.mark.parametrize("missing_group", ["tp", "pp", "dp", "dp_cp", "expt_dp"])
def test_checkpoint_resolver_rejects_missing_multimodule_transport_group(missing_group):
    groups = {
        "tp": "tp",
        "pp": "pp",
        "dp": "dp",
        "dp_cp": "dp-cp",
        "expt_dp": "expt-dp",
    }
    groups[missing_group] = None
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"vision": ProcessGroupCollection(**groups)},
        loss_module_name="language",
        module_order=("vision", "language"),
    )

    with pytest.raises(ValueError, match=missing_group):
        checkpointing._resolve_checkpoint_pg_collection(carrier)


@pytest.mark.parametrize(
    ("group_kind", "expected_group_name"),
    (("dp", "dp_cp_group"), ("ep_dp", "expt_dp_group")),
)
def test_fully_parallel_load_uses_explicit_process_group(group_kind, expected_group_name):
    args = SimpleNamespace(
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_load=True,
        ckpt_fully_parallel_load_process_group=group_kind,
        ckpt_fully_parallel_load_exchange_algo="broadcast",
        ckpt_load_validate_sharding_integrity=True,
        dist_ckpt_strictness="assume_ok_unexpected",
        verify_integrity=False,
    )
    dp_cp_group = object()
    expt_dp_group = object()
    explicit_groups = {
        "dp_cp_group": dp_cp_group,
        "expt_dp_group": expt_dp_group,
    }
    base_strategy = object()
    wrapped_strategy = object()
    state_dict = {"model": object()}
    loaded_state_dict = {"loaded": object()}
    context = {}

    with (
        mock.patch.object(
            checkpointing, "get_checkpoint_name", return_value="/checkpoint/iter_0000001"
        ),
        mock.patch.object(
            checkpointing, "TorchDistLoadShardedStrategy", return_value=base_strategy
        ),
        mock.patch.object(
            checkpointing,
            "FullyParallelLoadStrategyWrapper",
            return_value=wrapped_strategy,
        ) as wrapper,
        mock.patch.object(
            checkpointing.dist_checkpointing, "load", return_value=loaded_state_dict
        ) as load,
        mock.patch.object(
            checkpointing.mpu,
            "get_data_parallel_group",
            side_effect=AssertionError("must not read MPU DP group"),
        ),
        mock.patch.object(
            checkpointing.mpu,
            "get_expert_data_parallel_group",
            side_effect=AssertionError("must not read MPU expert DP group"),
        ),
    ):
        result = checkpointing._load_global_dist_base_checkpoint(
            "/checkpoint",
            args,
            rank0=False,
            sharded_state_dict=state_dict,
            iteration=1,
            release=False,
            checkpointing_context=context,
            dp_cp_group=dp_cp_group,
            expt_dp_group=expt_dp_group,
        )

    wrapper.assert_called_once_with(
        base_strategy,
        explicit_groups[expected_group_name],
        exchange_algo="broadcast",
    )
    load.assert_called_once()
    assert load.call_args.args[2] is wrapped_strategy
    assert context["load_strategy"] is wrapped_strategy
    assert result == (
        loaded_state_dict,
        "/checkpoint/iter_0000001",
        False,
        checkpointing.CheckpointType.GLOBAL,
    )


def test_fully_parallel_load_preserves_stock_mpu_fallback():
    args = SimpleNamespace(
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_load=True,
        ckpt_fully_parallel_load_process_group="dp",
        ckpt_fully_parallel_load_exchange_algo="broadcast",
        ckpt_load_validate_sharding_integrity=True,
        dist_ckpt_strictness="assume_ok_unexpected",
        verify_integrity=False,
    )
    fallback_group = object()
    base_strategy = object()
    wrapped_strategy = object()

    with (
        mock.patch.object(
            checkpointing, "get_checkpoint_name", return_value="/checkpoint/iter_0000001"
        ),
        mock.patch.object(
            checkpointing, "TorchDistLoadShardedStrategy", return_value=base_strategy
        ),
        mock.patch.object(
            checkpointing,
            "FullyParallelLoadStrategyWrapper",
            return_value=wrapped_strategy,
        ) as wrapper,
        mock.patch.object(
            checkpointing.dist_checkpointing, "load", return_value={"loaded": object()}
        ),
        mock.patch.object(
            checkpointing.mpu,
            "get_data_parallel_group",
            return_value=fallback_group,
        ) as get_dp_group,
    ):
        checkpointing._load_global_dist_base_checkpoint(
            "/checkpoint",
            args,
            rank0=False,
            sharded_state_dict={"model": object()},
            iteration=1,
            release=False,
        )

    get_dp_group.assert_called_once_with(with_context_parallel=True)
    wrapper.assert_called_once_with(
        base_strategy,
        fallback_group,
        exchange_algo="broadcast",
    )
