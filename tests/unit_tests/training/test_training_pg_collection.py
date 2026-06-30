# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the training-loop process-group carrier."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import training as training_mod


class _StopAfterBootstrap(Exception):
    pass


def _pretrain_until_after_jit(pg_collection, p2p_communicator=None):
    cfg = SimpleNamespace(logger=SimpleNamespace(log_progress=False))
    args = SimpleNamespace(fine_grained_activation_offloading=False)
    with (
        mock.patch.object(training_mod.ft_integration, "setup"),
        mock.patch.object(training_mod, "initialize_megatron") as initialize,
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "set_jit_fusion_options") as set_jit,
        mock.patch.object(training_mod, "get_pg_size", return_value=4) as get_pg_size,
        mock.patch.object(training_mod.torch, "tensor", side_effect=_StopAfterBootstrap),
    ):
        with pytest.raises(_StopAfterBootstrap):
            training_mod.pretrain(
                cfg,
                train_valid_test_dataset_provider=None,
                model_type=None,
                forward_step_func=None,
                p2p_communicator=p2p_communicator,
                pg_collection=pg_collection,
            )
    return initialize, set_jit, get_pg_size


def test_multi_bootstrap_skips_seed_and_uses_strict_local_tp_for_jit():
    local = SimpleNamespace(tp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    with mock.patch.object(
        carrier, "get_only_local_collection", wraps=carrier.get_only_local_collection
    ) as get_only_local:
        communicator = object.__new__(MultiModulePipelineCommunicator)
        initialize, set_jit, get_pg_size = _pretrain_until_after_jit(carrier, communicator)
    assert initialize.call_args.kwargs["skip_random_seed"] is True
    get_only_local.assert_called_once_with()
    get_pg_size.assert_called_once_with(local.tp)
    set_jit.assert_called_once_with(tp_size=4)


@pytest.mark.parametrize("p2p_communicator", [None, object()])
def test_multi_bootstrap_requires_multimodule_communicator_before_initialization(
    p2p_communicator,
):
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": SimpleNamespace(tp=object())}, language_model_module_name=None
    )
    cfg = SimpleNamespace(logger=SimpleNamespace(log_progress=False))
    with (
        mock.patch.object(training_mod.ft_integration, "setup") as ft_setup,
        mock.patch.object(training_mod, "initialize_megatron") as initialize,
        pytest.raises(ValueError, match="MultiModulePipelineCommunicator"),
    ):
        training_mod.pretrain(
            cfg,
            train_valid_test_dataset_provider=None,
            model_type=None,
            forward_step_func=None,
            p2p_communicator=p2p_communicator,
            pg_collection=carrier,
        )
    ft_setup.assert_not_called()
    initialize.assert_not_called()


def test_multi_bootstrap_rejects_colocated_local_collections():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": SimpleNamespace(), "language": SimpleNamespace()},
        language_model_module_name="language",
    )
    cfg = SimpleNamespace(logger=SimpleNamespace(log_progress=False))
    with (
        mock.patch.object(training_mod.ft_integration, "setup"),
        mock.patch.object(training_mod, "initialize_megatron"),
        pytest.raises(ValueError, match="exactly one local"),
    ):
        training_mod.pretrain(
            cfg,
            train_valid_test_dataset_provider=None,
            model_type=None,
            forward_step_func=None,
            p2p_communicator=object.__new__(MultiModulePipelineCommunicator),
            pg_collection=carrier,
        )


def test_plain_bootstrap_keeps_default_seed_and_jit_behavior():
    local = SimpleNamespace()
    with mock.patch.object(
        training_mod.ProcessGroupCollection,
        "use_mpu_process_groups",
        return_value=local,
    ) as use_mpu:
        initialize, set_jit, get_pg_size = _pretrain_until_after_jit(None)
    assert initialize.call_args.kwargs["skip_random_seed"] is False
    use_mpu.assert_called_once_with()
    get_pg_size.assert_not_called()
    set_jit.assert_called_once_with(tp_size=None)


def test_explicit_plain_bootstrap_uses_identity_groups_and_tp_size():
    pg_collection = ProcessGroupCollection()
    pg_collection.pp = object()
    pg_collection.dp = object()
    pg_collection.tp = object()
    pg_collection.ep = object()
    pg_collection.expt_tp = object()
    initialize, set_jit, get_pg_size = _pretrain_until_after_jit(pg_collection)
    initialize_kwargs = initialize.call_args.kwargs
    assert initialize_kwargs["seed_pp_group"] is pg_collection.pp
    assert initialize_kwargs["seed_dp_group"] is pg_collection.dp
    assert initialize_kwargs["seed_tp_group"] is pg_collection.tp
    assert initialize_kwargs["seed_ep_group"] is pg_collection.ep
    assert initialize_kwargs["seed_etp_group"] is pg_collection.expt_tp
    get_pg_size.assert_called_once_with(pg_collection.tp)
    set_jit.assert_called_once_with(tp_size=4)


def _eval_args(**overrides):
    values = dict(
        eval_global_batch_size=1,
        eval_micro_batch_size=1,
        data_parallel_size=1,
        cuda_graph_impl="none",
        moe_expert_rank_capacity_factor=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        modelopt_enabled=False,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=1,
        empty_unused_memory_level=0,
        sft=False,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_evaluate_multi_forwards_exact_pair_and_encoder_avoids_mpu_loss_group():
    local = SimpleNamespace(pp=object(), dp_cp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    communicator = object()
    forward_backward = mock.Mock(side_effect=[[{"unexpected": object()}], object()])
    model = [SimpleNamespace(eval=mock.Mock(), train=mock.Mock())]
    rerun = SimpleNamespace(
        get_mode=mock.Mock(return_value=object()), set_mode=mock.Mock()
    )
    with (
        mock.patch.object(training_mod, "get_args", return_value=_eval_args()),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(
            training_mod, "get_forward_backward_func", return_value=forward_backward
        ) as get_schedule,
        mock.patch.object(training_mod.ft_integration, "on_eval_step_start"),
        mock.patch.object(training_mod.ft_integration, "on_eval_step_end"),
        mock.patch.object(training_mod, "has_nvidia_modelopt", True),
        mock.patch.object(
            training_mod, "get_tensor_shapes_adjust_fn_for_distillation"
        ) as shape_adjust,
        mock.patch.object(training_mod, "is_pp_last_stage") as is_last,
        mock.patch.object(training_mod, "is_last_rank", return_value=False),
        mock.patch.object(
            training_mod.mpu,
            "is_pipeline_last_stage",
            side_effect=AssertionError("encoder rank must not consult MPU loss ownership"),
        ),
    ):
        result = training_mod.evaluate(
            None,
            iter([]),
            model,
            object(),
            SimpleNamespace(),
            pg_collection=carrier,
            p2p_communicator=communicator,
        )
    get_schedule.assert_called_once_with(pg_collection=carrier)
    assert forward_backward.call_count == 2
    non_loss_kwargs = forward_backward.call_args_list[1].kwargs
    assert non_loss_kwargs["pg_collection"] is carrier
    assert non_loss_kwargs["p2p_communicator"] is communicator
    assert non_loss_kwargs["collect_non_loss_data"] is True
    shape_adjust.assert_not_called()
    is_last.assert_not_called()
    assert result[0] == {}
    assert result[1] is None


def _log_args(**overrides):
    values = dict(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        freeze_all_layers=False,
        tensorboard_log_interval=100,
        skipped_train_samples=0,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_interval=100,
        consumed_train_samples=1,
        record_memory_history=False,
        train_iters=10,
        rl_use_sequence_packing=False,
        log_timers_to_tensorboard=False,
        log_throughput=False,
        log_energy=False,
        log_memory_interval=None,
        rl_profile=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _training_log(args, carrier, *, report_memory_flag=False):
    timer = mock.MagicMock()
    timer.elapsed.return_value = 1.0
    timers = mock.MagicMock(return_value=timer)
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(training_mod.one_logger_utils, "track_e2e_metrics"),
        mock.patch.object(training_mod, "num_floating_point_operations", return_value=1.0),
        mock.patch.object(training_mod, "print_rank_last"),
        mock.patch.object(training_mod, "get_loaded_iteration", return_value=0),
        mock.patch.object(training_mod, "report_theoretical_memory"),
        mock.patch.object(training_mod, "report_memory") as report_memory,
        mock.patch.object(training_mod.torch.distributed, "get_rank", return_value=1),
        mock.patch.object(
            training_mod, "reduce_max_stat_across_model_parallel_group"
        ) as reduce_max,
    ):
        training_mod.training_log(
            {},
            {},
            0.1,
            1,
            1.0,
            report_memory_flag,
            0,
            None,
            None,
            None,
            None,
            pg_collection=carrier,
        )
    return reduce_max, report_memory


def test_training_log_multi_skips_lr_reduction():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": SimpleNamespace(dp=object())}, language_model_module_name=None
    )
    reduce_max, _ = _training_log(_log_args(), carrier)
    reduce_max.assert_not_called()


def test_training_log_multi_reports_memory_on_strict_local_dp_group():
    local = SimpleNamespace(dp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    _, report_memory = _training_log(
        _log_args(log_interval=1), carrier, report_memory_flag=True
    )
    report_memory.assert_called_once_with(
        "(after 1 iterations)", process_group=local.dp
    )


def test_training_log_multi_reports_moe_only_on_language_rank():
    tracker = mock.MagicMock()
    language = SimpleNamespace(dp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"language": language}, language_model_module_name="language"
    )
    args = _log_args(
        num_experts=8,
        moe_router_load_balancing_type=[],
        moe_z_loss_coeff=None,
        num_layers=2,
        moe_per_layer_logging=False,
        moe_layer_freq=1,
    )
    with (
        mock.patch.object(training_mod, "get_moe_metrics_tracker", return_value=tracker),
        mock.patch.object(training_mod, "is_hybrid_model", return_value=False),
    ):
        _training_log(args, carrier)
    assert tracker.report.call_args.kwargs["pg_collection"] is language
