# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``train_step`` schedule plumbing."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import training as training_mod


class _Rerun:
    """Run forward/backward once and optionally exit before ``optimizer.step``."""

    def __init__(self, *, exit_before_optimizer=False):
        self._ran = False
        self._exit_before_optimizer = exit_before_optimizer

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, self._exit_before_optimizer, 0


def _multi_carrier(*, loss_local=False):
    module_pgs = {"vision": ProcessGroupCollection()}
    if loss_local:
        module_pgs["language"] = ProcessGroupCollection(pp=object(), dp_cp=object())
    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs,
        loss_module_name="language",
        module_order=("vision", "language"),
    )


def _run_train_step(
    *,
    pg_collection=None,
    p2p_communicator=None,
    losses_reduced=None,
    optimizer_values=(True, 7.0, 9.0),
    log_num_zeros_in_grad=False,
    exit_before_optimizer=False,
    schedule=None,
):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        vision_pretraining_type=None,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=log_num_zeros_in_grad,
        data_parallel_size=1,
    )
    captured = {}
    scheduler = mock.Mock()
    optimizer = SimpleNamespace(
        zero_grad=mock.Mock(), step=mock.Mock(return_value=optimizer_values)
    )

    def capture_schedule(**kwargs):
        captured.update(kwargs)
        return [] if losses_reduced is None else losses_reduced

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod,
            "get_rerun_state_machine",
            return_value=_Rerun(exit_before_optimizer=exit_before_optimizer),
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=iter(()),
            model=[SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(),
            forward_backward_func=schedule or capture_schedule,
            iteration=0,
            pg_collection=pg_collection,
            p2p_communicator=p2p_communicator,
        )
    return SimpleNamespace(captured=captured, result=result, scheduler=scheduler)


def test_train_step_forwards_the_exact_carrier_and_resolved_communicator():
    carrier = _multi_carrier()
    communicator = object.__new__(MultiModulePipelineCommunicator)
    resolved_communicator = object()

    with mock.patch.object(
        training_mod,
        "_resolve_pipeline_communicator",
        return_value=resolved_communicator,
    ) as resolve:
        run = _run_train_step(
            pg_collection=carrier,
            p2p_communicator=communicator,
            exit_before_optimizer=True,
        )

    resolve.assert_called_once_with(carrier, communicator, mock.ANY)
    assert run.captured["pg_collection"] is carrier
    assert run.captured["p2p_communicator"] is resolved_communicator


@pytest.mark.parametrize(
    ("carrier", "communicator", "message"),
    (
        (
            _multi_carrier(),
            object.__new__(P2PCommunicator),
            "MultiModulePipelineCommunicator",
        ),
        (
            ProcessGroupCollection(pp=object(), tp=object(), cp=object()),
            object.__new__(MultiModulePipelineCommunicator),
            "MultiModuleProcessGroupCollection",
        ),
    ),
)
def test_train_step_rejects_mismatched_pair_before_schedule(
    carrier, communicator, message
):
    schedule = mock.Mock(return_value=[])

    with pytest.raises(ValueError, match=message):
        _run_train_step(
            pg_collection=carrier,
            p2p_communicator=communicator,
            exit_before_optimizer=True,
            schedule=schedule,
        )

    schedule.assert_not_called()


def test_plain_train_step_uses_exact_groups_and_reduced_optimizer_stats():
    mp_group, pp_group, dp_cp_group = object(), object(), object()
    carrier = ProcessGroupCollection(
        mp=mp_group, pp=pp_group, dp_cp=dp_cp_group, tp=object(), cp=object()
    )
    losses = [{"lm loss": torch.tensor([6.0, 3.0])}]

    with (
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            return_value=False,
        ) as reduce_success,
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=(11.0, 13.0),
        ) as reduce_max,
        mock.patch.object(
            training_mod, "is_pp_last_stage", return_value=True
        ) as is_last_stage,
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
    ):
        run = _run_train_step(
            pg_collection=carrier,
            p2p_communicator=object.__new__(P2PCommunicator),
            losses_reduced=losses,
            log_num_zeros_in_grad=True,
        )

    reduce_success.assert_called_once_with(True, group=mp_group)
    assert reduce_max.call_args_list == [
        mock.call(7.0, group=mp_group),
        mock.call(9.0, group=mp_group),
    ]
    assert run.result[1] == 1
    assert run.result[5:7] == (11.0, 13.0)
    is_last_stage.assert_called_once_with(pp_group)
    assert all_reduce.call_args.kwargs["group"] is dp_cp_group
    run.scheduler.step.assert_not_called()


def test_multimodule_train_step_keeps_optimizer_stats_without_generic_reductions():
    grad_norm, num_zeros_in_grad = object(), object()

    with (
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=AssertionError("generic success reduction used"),
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=AssertionError("generic stat reduction used"),
        ),
    ):
        run = _run_train_step(
            pg_collection=_multi_carrier(),
            p2p_communicator=object.__new__(MultiModulePipelineCommunicator),
            optimizer_values=(False, grad_norm, num_zeros_in_grad),
            log_num_zeros_in_grad=True,
        )

    assert run.result[1] == 1
    assert run.result[5] is grad_norm
    assert run.result[6] is num_zeros_in_grad


def test_encoder_only_train_step_does_not_read_plain_groups():
    with (
        mock.patch.object(
            training_mod.ProcessGroupCollection,
            "use_mpu_process_groups",
            side_effect=AssertionError("MPU fallback used"),
        ),
        mock.patch.object(
            training_mod,
            "is_pp_last_stage",
            side_effect=AssertionError("plain PP group used"),
        ),
    ):
        run = _run_train_step(
            pg_collection=_multi_carrier(),
            p2p_communicator=object.__new__(MultiModulePipelineCommunicator),
        )

    assert run.result[0] == {}


def test_local_loss_child_controls_terminal_stage_and_loss_reduction_group():
    carrier = _multi_carrier(loss_local=True)
    loss_collection = carrier.get_loss_module_collection()
    losses = [{"lm loss": torch.tensor([6.0, 3.0])}]

    with (
        mock.patch.object(
            training_mod, "is_pp_last_stage", return_value=True
        ) as is_last_stage,
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
    ):
        run = _run_train_step(
            pg_collection=carrier,
            p2p_communicator=object.__new__(MultiModulePipelineCommunicator),
            losses_reduced=losses,
        )

    is_last_stage.assert_called_once_with(loss_collection.pp)
    assert all_reduce.call_args.kwargs["group"] is loss_collection.dp_cp
    assert run.result[0]["lm loss"].item() == pytest.approx(2.0)


def test_plain_carrier_missing_required_group_raises_value_error():
    carrier = ProcessGroupCollection(
        mp=None, pp=object(), dp_cp=object(), tp=object(), cp=object()
    )

    with pytest.raises(ValueError, match="plain pg_collection must define mp"):
        _run_train_step(
            pg_collection=carrier,
            p2p_communicator=object.__new__(P2PCommunicator),
        )


def test_train_step_without_carrier_keeps_legacy_mpu_fallback():
    carrier = ProcessGroupCollection(mp=object(), pp=object(), dp_cp=object())

    with (
        mock.patch.object(
            training_mod.ProcessGroupCollection,
            "use_mpu_process_groups",
            return_value=carrier,
        ) as use_mpu,
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            return_value=True,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            return_value=7.0,
        ),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
    ):
        run = _run_train_step()

    use_mpu.assert_called_once_with()
    run.scheduler.step.assert_called_once_with(increment=1)
