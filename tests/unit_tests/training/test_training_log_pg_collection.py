# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for training metric process-group routing."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import training as training_mod


def _multi_carrier(*, loss_local=False):
    module_pgs = {"vision": ProcessGroupCollection()}
    if loss_local:
        module_pgs["language"] = ProcessGroupCollection(mp=object(), dp=object())
    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs,
        loss_module_name="language",
        module_order=("vision", "language"),
    )


def _training_log_args(*, log_memory_interval):
    return SimpleNamespace(
        consumed_train_samples=0,
        data_parallel_size=4,
        dsa_indexer_loss_coeff=None,
        freeze_all_layers=False,
        log_energy=False,
        log_interval=1,
        log_memory_interval=log_memory_interval,
        log_throughput=False,
        log_timers_to_tensorboard=False,
        micro_batch_size=1,
        moe_layer_freq=1,
        moe_per_layer_logging=False,
        moe_router_load_balancing_type=[],
        moe_z_loss_coeff=None,
        mtp_num_layers=None,
        num_experts=8,
        num_layers=2,
        perform_rl_step=False,
        record_memory_history=False,
        rl_profile=False,
        rl_use_sequence_packing=False,
        seq_length=16,
        skipped_train_samples=0,
        timing_log_level=0,
        train_iters=10,
        world_size=8,
    )


def _run_training_log(*, pg_collection, report_memory_flag, log_memory_interval):
    args = _training_log_args(log_memory_interval=log_memory_interval)
    timers = mock.MagicMock()
    timers.return_value.elapsed.return_value = 1.0
    tracker = mock.Mock()
    tracker.report.return_value = ""

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "get_loaded_iteration", return_value=0),
        mock.patch.object(training_mod, "is_hybrid_model", return_value=False),
        mock.patch.object(training_mod, "num_floating_point_operations", return_value=0.0),
        mock.patch.object(training_mod, "print_rank_last"),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(training_mod.one_logger_utils, "track_e2e_metrics"),
        mock.patch.object(training_mod.torch.distributed, "get_rank", return_value=1),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            return_value=0.25,
        ) as reduce_learning_rate,
        mock.patch.object(
            training_mod, "get_moe_metrics_tracker", return_value=tracker
        ) as get_tracker,
        mock.patch.object(training_mod, "report_memory") as report_memory,
    ):
        returned_report_memory_flag = training_mod.training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=0.25,
            iteration=2,
            loss_scale=1.0,
            report_memory_flag=report_memory_flag,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            pg_collection=pg_collection,
        )

    return SimpleNamespace(
        get_tracker=get_tracker,
        reduce_learning_rate=reduce_learning_rate,
        report_memory=report_memory,
        returned_report_memory_flag=returned_report_memory_flag,
        tracker=tracker,
    )


@pytest.mark.parametrize(
    ("report_memory_flag", "log_memory_interval"),
    ((True, None), (False, 1)),
    ids=("initial-memory-report", "memory-interval"),
)
def test_plain_collection_routes_training_metrics_to_exact_groups(
    report_memory_flag, log_memory_interval
):
    mp_group, dp_group = object(), object()
    pg_collection = ProcessGroupCollection(mp=mp_group, dp=dp_group)

    run = _run_training_log(
        pg_collection=pg_collection,
        report_memory_flag=report_memory_flag,
        log_memory_interval=log_memory_interval,
    )

    run.reduce_learning_rate.assert_called_once_with(0.25, group=mp_group)
    assert run.tracker.report.call_args.kwargs["pg_collection"] is pg_collection
    run.report_memory.assert_called_once_with(
        "(after 2 iterations)", process_group=dp_group
    )
    assert run.returned_report_memory_flag is False


@pytest.mark.parametrize(
    ("report_memory_flag", "log_memory_interval"),
    ((True, None), (False, 1)),
    ids=("initial-memory-report", "memory-interval"),
)
def test_none_collection_preserves_training_metric_fallbacks(
    report_memory_flag, log_memory_interval
):
    run = _run_training_log(
        pg_collection=None,
        report_memory_flag=report_memory_flag,
        log_memory_interval=log_memory_interval,
    )

    run.reduce_learning_rate.assert_called_once_with(0.25, group=None)
    assert run.tracker.report.call_args.kwargs["pg_collection"] is None
    run.report_memory.assert_called_once_with(
        "(after 2 iterations)", process_group=None
    )
    assert run.returned_report_memory_flag is False


@pytest.mark.parametrize(
    ("report_memory_flag", "log_memory_interval"),
    ((True, None), (False, 1)),
    ids=("initial-memory-report", "memory-interval"),
)
def test_encoder_only_collection_skips_loss_owner_metrics(
    report_memory_flag, log_memory_interval
):
    run = _run_training_log(
        pg_collection=_multi_carrier(),
        report_memory_flag=report_memory_flag,
        log_memory_interval=log_memory_interval,
    )

    run.reduce_learning_rate.assert_not_called()
    run.get_tracker.assert_not_called()
    run.report_memory.assert_not_called()
    assert run.returned_report_memory_flag is False


@pytest.mark.parametrize(
    ("report_memory_flag", "log_memory_interval"),
    ((True, None), (False, 1)),
    ids=("initial-memory-report", "memory-interval"),
)
def test_loss_local_collection_routes_metrics_to_loss_child(
    report_memory_flag, log_memory_interval
):
    pg_collection = _multi_carrier(loss_local=True)
    loss_pg_collection = pg_collection.get_loss_module_collection()

    run = _run_training_log(
        pg_collection=pg_collection,
        report_memory_flag=report_memory_flag,
        log_memory_interval=log_memory_interval,
    )

    run.reduce_learning_rate.assert_not_called()
    assert run.tracker.report.call_args.kwargs["pg_collection"] is loss_pg_collection
    run.report_memory.assert_called_once_with(
        "(after 2 iterations)", process_group=loss_pg_collection.dp
    )
    assert run.returned_report_memory_flag is False


def test_data_parallel_world_size_uses_explicit_plain_collection():
    dp_group = mock.Mock()
    dp_group.size.return_value = 3
    args = SimpleNamespace(data_parallel_size=9)

    with (
        mock.patch.object(training_mod.mpu, "model_parallel_is_initialized") as is_initialized,
        mock.patch.object(training_mod.mpu, "get_data_parallel_world_size") as get_world_size,
    ):
        world_size = training_mod._get_data_parallel_world_size(
            ProcessGroupCollection(dp=dp_group), args
        )

    assert world_size == 3
    is_initialized.assert_not_called()
    get_world_size.assert_not_called()


def test_data_parallel_world_size_uses_local_loss_child():
    pg_collection = _multi_carrier(loss_local=True)
    loss_pg_collection = pg_collection.get_loss_module_collection()
    loss_pg_collection.dp = mock.Mock()
    loss_pg_collection.dp.size.return_value = 5

    world_size = training_mod._get_data_parallel_world_size(
        pg_collection, SimpleNamespace(data_parallel_size=9)
    )

    assert world_size == 5


def test_data_parallel_world_size_uses_args_on_encoder_only_rank():
    args = SimpleNamespace(data_parallel_size=9)

    with (
        mock.patch.object(
            training_mod.mpu,
            "model_parallel_is_initialized",
            side_effect=AssertionError("MPU fallback used"),
        ),
        mock.patch.object(
            training_mod.mpu,
            "get_data_parallel_world_size",
            side_effect=AssertionError("MPU fallback used"),
        ),
    ):
        world_size = training_mod._get_data_parallel_world_size(_multi_carrier(), args)

    assert world_size == 9


@pytest.mark.parametrize(
    ("initialized", "expected"), ((True, 7), (False, 9)), ids=("mpu", "args")
)
def test_data_parallel_world_size_preserves_none_fallback(initialized, expected):
    args = SimpleNamespace(data_parallel_size=9)

    with (
        mock.patch.object(
            training_mod.mpu, "model_parallel_is_initialized", return_value=initialized
        ),
        mock.patch.object(
            training_mod.mpu, "get_data_parallel_world_size", return_value=7
        ) as get_world_size,
    ):
        world_size = training_mod._get_data_parallel_world_size(None, args)

    assert world_size == expected
    assert get_world_size.call_count == int(initialized)
