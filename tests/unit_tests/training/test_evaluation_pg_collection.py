# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for heterogeneous evaluation process-group plumbing."""

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


class _RerunState:
    def get_mode(self):
        return "previous"

    def set_mode(self, mode):
        pass


def _multi_carrier(*, loss_local=False):
    module_pgs = {"vision": ProcessGroupCollection()}
    if loss_local:
        module_pgs["language"] = ProcessGroupCollection(
            pp=object(), cp=object(), dp_cp=object()
        )
    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs,
        loss_module_name="language",
        module_order=("vision", "language"),
    )


def _plain_carrier():
    return ProcessGroupCollection(
        pp=object(), tp=object(), cp=object(), dp_cp=object()
    )


def _eval_args(**overrides):
    values = dict(
        eval_global_batch_size=1,
        eval_micro_batch_size=1,
        data_parallel_size=1,
        cuda_graph_impl=None,
        moe_expert_rank_capacity_factor=None,
        seq_length=8,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        sft=False,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_one_iteration(
    *,
    carrier,
    communicator,
    losses=None,
    process_non_loss_data_func=None,
    world_last=True,
):
    args = _eval_args()
    payload = object()

    def schedule_call(**kwargs):
        if kwargs.get("collect_non_loss_data"):
            return payload
        return [] if losses is None else losses

    schedule = mock.Mock(side_effect=schedule_call)
    selector = mock.Mock(return_value=schedule)
    timers = mock.MagicMock()
    config = SimpleNamespace()

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunState()
        ),
        mock.patch.object(training_mod, "get_forward_backward_func", selector),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_last_rank", return_value=world_last),
        mock.patch.object(training_mod.ft_integration, "on_eval_step_start"),
        mock.patch.object(training_mod.ft_integration, "on_eval_step_end"),
    ):
        result = training_mod.evaluate(
            forward_step_func=mock.Mock(),
            data_iterator=iter(()),
            model=[mock.Mock()],
            process_non_loss_data_func=process_non_loss_data_func,
            config=config,
            eval_iters=1,
            pg_collection=carrier,
            p2p_communicator=communicator,
        )

    return SimpleNamespace(
        payload=payload, result=result, schedule=schedule, selector=selector
    )


def _call_wrapper(carrier, communicator):
    training_mod.evaluate_and_print_results(
        "prefix",
        mock.Mock(),
        [iter(()), iter(())],
        [mock.Mock()],
        1,
        None,
        SimpleNamespace(),
        pg_collection=carrier,
        p2p_communicator=communicator,
    )


def test_evaluate_forwards_exact_multimodule_pair_without_mpu_access():
    carrier = _multi_carrier()
    communicator = object.__new__(MultiModulePipelineCommunicator)

    with (
        mock.patch.object(
            training_mod.mpu,
            "is_pipeline_last_stage",
            side_effect=AssertionError("MPU terminal lookup used"),
        ),
        mock.patch.object(
            training_mod.mpu,
            "get_data_parallel_group",
            side_effect=AssertionError("MPU data-parallel lookup used"),
        ),
    ):
        run = _run_one_iteration(carrier=carrier, communicator=communicator)

    run.selector.assert_called_once_with(pg_collection=carrier)
    assert run.schedule.call_count == 1
    assert run.schedule.call_args.kwargs["pg_collection"] is carrier
    assert run.schedule.call_args.kwargs["p2p_communicator"] is communicator
    assert run.result[0] == {}


def test_local_loss_child_controls_terminal_stage_and_reduction_group():
    carrier = _multi_carrier(loss_local=True)
    loss_collection = carrier.get_loss_module_collection()
    losses = [{"lm loss": torch.tensor([6.0, 3.0])}]

    with (
        mock.patch.object(
            training_mod, "is_pp_last_stage", return_value=True
        ) as is_last_stage,
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
    ):
        run = _run_one_iteration(
            carrier=carrier,
            communicator=object.__new__(MultiModulePipelineCommunicator),
            losses=losses,
        )

    is_last_stage.assert_called_once_with(loss_collection.pp)
    assert all_reduce.call_args.kwargs["group"] is loss_collection.dp_cp
    assert run.result[0]["lm loss"].item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("world_last", "expected_data"), ((False, None), (True, "payload"))
)
def test_every_multimodule_rank_joins_non_loss_schedule(world_last, expected_data):
    run = _run_one_iteration(
        carrier=_multi_carrier(),
        communicator=object.__new__(MultiModulePipelineCommunicator),
        process_non_loss_data_func=mock.Mock(),
        world_last=world_last,
    )

    assert run.schedule.call_count == 2
    bridge_call = run.schedule.call_args_list[1]
    assert bridge_call.kwargs["collect_non_loss_data"] is True
    assert bridge_call.kwargs["pg_collection"] is run.schedule.call_args_list[0].kwargs[
        "pg_collection"
    ]
    assert bridge_call.kwargs["p2p_communicator"] is run.schedule.call_args_list[
        0
    ].kwargs["p2p_communicator"]
    assert run.result[1] is (run.payload if expected_data == "payload" else None)


def test_plain_nonreporting_rank_keeps_legacy_non_loss_behavior():
    carrier = _plain_carrier()

    with mock.patch.object(training_mod, "is_pp_last_stage", return_value=False):
        run = _run_one_iteration(
            carrier=carrier,
            communicator=object.__new__(P2PCommunicator),
            process_non_loss_data_func=mock.Mock(),
            world_last=False,
        )

    assert run.schedule.call_count == 1
    assert run.result[1] is None


def test_mismatched_pair_is_rejected_before_selector_or_collective():
    selector = mock.Mock()

    with (
        mock.patch.object(training_mod, "get_args", return_value=_eval_args()),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_forward_backward_func", selector),
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
        pytest.raises(ValueError, match="MultiModulePipelineCommunicator"),
    ):
        training_mod.evaluate(
            mock.Mock(),
            iter(()),
            [mock.Mock()],
            None,
            SimpleNamespace(),
            eval_iters=1,
            pg_collection=_multi_carrier(),
            p2p_communicator=object.__new__(P2PCommunicator),
        )

    selector.assert_not_called()
    all_reduce.assert_not_called()


def test_wrapper_forwards_exact_pair_to_every_validation_set():
    carrier = _multi_carrier()
    communicator = object.__new__(MultiModulePipelineCommunicator)
    args = SimpleNamespace(
        multiple_validation_sets=True,
        eval_iters=[1, 2],
        full_validation=False,
        validation_set_names=None,
    )

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(
            training_mod, "evaluate", return_value=({}, None, False)
        ) as evaluate,
        mock.patch.object(training_mod, "print_rank_last"),
    ):
        _call_wrapper(carrier, communicator)

    assert evaluate.call_count == 2
    for evaluate_call in evaluate.call_args_list:
        assert evaluate_call.kwargs["pg_collection"] is carrier
        assert evaluate_call.kwargs["p2p_communicator"] is communicator


def test_multimodule_full_validation_uses_global_rank_zero_as_source():
    carrier = _multi_carrier()
    args = SimpleNamespace(
        multiple_validation_sets=True,
        eval_iters=[1, 2],
        full_validation=True,
        validation_set_names=None,
    )
    eval_iters_tensor = mock.Mock()
    eval_iters_tensor.tolist.return_value = [1, 2]

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod.torch, "tensor", return_value=eval_iters_tensor),
        mock.patch.object(
            training_mod.torch.distributed, "get_rank", return_value=0
        ) as get_rank,
        mock.patch.object(training_mod.torch.distributed, "broadcast") as broadcast,
        mock.patch.object(
            training_mod.mpu,
            "get_tensor_model_parallel_rank",
            side_effect=AssertionError("MPU TP rank used"),
        ),
        mock.patch.object(training_mod, "evaluate", return_value=({}, None, False)),
        mock.patch.object(training_mod, "print_rank_last"),
    ):
        _call_wrapper(
            carrier, object.__new__(MultiModulePipelineCommunicator)
        )

    get_rank.assert_called_once_with()
    broadcast.assert_called_once_with(eval_iters_tensor, 0)
