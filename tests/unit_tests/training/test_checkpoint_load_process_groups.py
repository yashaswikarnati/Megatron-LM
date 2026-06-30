# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for checkpoint-load process-group forwarding."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import checkpointing


def _fully_parallel_args(process_group):
    return SimpleNamespace(
        auto_detect_ckpt_format=False,
        use_dist_ckpt=True,
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_load=True,
        ckpt_fully_parallel_load_process_group=process_group,
        ckpt_fully_parallel_load_exchange_algo="broadcast",
        ckpt_load_validate_sharding_integrity=True,
        dist_ckpt_strictness="assume_ok_unexpected",
        verify_integrity=False,
    )


@pytest.mark.parametrize(
    "selector,group_keyword,mpu_getter",
    [
        ("dp", "dp_cp_group", "get_data_parallel_group"),
        ("ep_dp", "expt_dp_group", "get_expert_data_parallel_group"),
    ],
)
@pytest.mark.parametrize("use_supplied_group", [True, False])
def test_fully_parallel_load_uses_supplied_group_or_mpu_fallback(
    selector, group_keyword, mpu_getter, use_supplied_group
):
    supplied_group = object() if use_supplied_group else None
    fallback_group = object()
    wrapper = object()
    group_kwargs = {"dp_cp_group": None, "expt_dp_group": None}
    group_kwargs[group_keyword] = supplied_group

    with (
        mock.patch.object(checkpointing, "get_checkpoint_name", return_value="checkpoint"),
        mock.patch.object(checkpointing, "TorchDistLoadShardedStrategy", return_value=object()),
        mock.patch.object(
            checkpointing, "FullyParallelLoadStrategyWrapper", return_value=wrapper
        ) as wrap,
        mock.patch.object(checkpointing.dist_checkpointing, "load", return_value={}),
        mock.patch.object(
            checkpointing.mpu, "get_data_parallel_group", return_value=fallback_group
        ) as get_dp,
        mock.patch.object(
            checkpointing.mpu, "get_expert_data_parallel_group", return_value=fallback_group
        ) as get_expt_dp,
    ):
        checkpointing._load_global_dist_base_checkpoint(
            "load", _fully_parallel_args(selector), False, {}, 1, False, **group_kwargs
        )

    expected_group = supplied_group if use_supplied_group else fallback_group
    assert wrap.call_args.args[1] is expected_group
    selected_getter = get_dp if mpu_getter == "get_data_parallel_group" else get_expt_dp
    other_getter = get_expt_dp if selected_getter is get_dp else get_dp
    if use_supplied_group:
        get_dp.assert_not_called()
        get_expt_dp.assert_not_called()
    else:
        selected_getter.assert_called_once_with(
            **({"with_context_parallel": True} if selector == "dp" else {})
        )
        other_getter.assert_not_called()


def test_nonpersistent_global_load_forwards_supplied_groups():
    args = SimpleNamespace(
        non_persistent_global_ckpt_dir="nonpersistent",
        non_persistent_ckpt_type="global",
        ckpt_step=None,
    )
    dp_cp_group, expt_dp_group = object(), object()
    expected = (object(), "checkpoint", False, checkpointing.CheckpointType.GLOBAL)
    with (
        mock.patch.object(checkpointing, "_get_non_persistent_iteration", return_value=3),
        mock.patch.object(checkpointing, "set_loaded_iteration"),
        mock.patch.object(
            checkpointing, "_load_global_dist_base_checkpoint", return_value=expected
        ) as load_global,
    ):
        result = checkpointing._load_base_checkpoint(
            None,
            args,
            rank0=False,
            sharded_state_dict={},
            checkpointing_context={"context": object()},
            dp_cp_group=dp_cp_group,
            expt_dp_group=expt_dp_group,
        )

    assert result is expected
    assert load_global.call_args.kwargs["dp_cp_group"] is dp_cp_group
    assert load_global.call_args.kwargs["expt_dp_group"] is expt_dp_group


def test_persistent_global_load_forwards_supplied_groups():
    args = SimpleNamespace(
        non_persistent_global_ckpt_dir=None, non_persistent_ckpt_type=None, ckpt_step=None
    )
    dp_cp_group, expt_dp_group = object(), object()
    expected = (object(), "checkpoint", False, checkpointing.CheckpointType.GLOBAL)
    with (
        mock.patch.object(checkpointing, "_get_non_persistent_iteration", return_value=-1),
        mock.patch.object(checkpointing, "get_checkpoint_tracker_filename", return_value="tracker"),
        mock.patch.object(checkpointing, "isfile", return_value=True),
        mock.patch.object(checkpointing, "read_metadata", return_value=(3, False)),
        mock.patch.object(checkpointing, "set_loaded_iteration"),
        mock.patch.object(checkpointing, "get_checkpoint_name", return_value="checkpoint"),
        mock.patch.object(checkpointing, "_get_checkpoint_format", return_value="torch_dist"),
        mock.patch.object(checkpointing, "print_rank_0"),
        mock.patch.object(
            checkpointing, "_load_global_dist_base_checkpoint", return_value=expected
        ) as load_global,
    ):
        result = checkpointing._load_base_checkpoint(
            "load",
            args,
            rank0=False,
            sharded_state_dict={},
            dp_cp_group=dp_cp_group,
            expt_dp_group=expt_dp_group,
        )

    assert result is expected
    assert load_global.call_args.kwargs["dp_cp_group"] is dp_cp_group
    assert load_global.call_args.kwargs["expt_dp_group"] is expt_dp_group


def test_nonpersistent_local_load_uses_supplied_dp_group():
    intermediate_state_dict = mock.Mock()
    manager = mock.Mock()
    manager.load.return_value = (intermediate_state_dict, "checkpoint")
    dp_cp_group = object()
    args = SimpleNamespace(
        non_persistent_ckpt_type="local", non_persistent_local_ckpt_algo="fully_parallel"
    )
    with mock.patch.object(
        checkpointing.mpu,
        "get_data_parallel_group",
        side_effect=AssertionError("supplied DP group must avoid MPU"),
    ):
        checkpointing._load_non_persistent_base_checkpoint(
            "load",
            args,
            False,
            {},
            1,
            {"local_checkpoint_manager": manager},
            dp_cp_group=dp_cp_group,
        )

    assert (
        intermediate_state_dict.to_state_dict.call_args.kwargs["parallelization_group"]
        is dp_cp_group
    )


def test_load_checkpoint_forwards_groups_to_rank_local_base_load():
    tree = ast.parse(Path(checkpointing.__file__).read_text())
    load_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint"
    )
    rank_local_calls = [
        node
        for node in ast.walk(load_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_base_checkpoint"
        and any(
            keyword.arg == "rank0"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
    ]
    assert len(rank_local_calls) == 1
    group_keywords = {keyword.arg: keyword.value for keyword in rank_local_calls[0].keywords}
    for group_name in ("dp_cp_group", "expt_dp_group"):
        assert isinstance(group_keywords[group_name], ast.Name)
        assert group_keywords[group_name].id == group_name
