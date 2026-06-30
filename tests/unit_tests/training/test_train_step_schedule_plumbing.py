# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for train_step process-group plumbing."""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(*, exit_before_optimizer=True, losses_reduced=None, is_last_stage=False, **kwargs):
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
        qk_clip=False,
        log_max_attention_logit=False,
        barrier_with_L1_time=False,
        log_num_zeros_in_grad=True,
        data_parallel_size=2,
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    rerun = _Rerun()
    if not exit_before_optimizer:
        rerun.should_checkpoint_and_exit = lambda: (False, False, 0)
    optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        step=mock.Mock(return_value=(True, 2.0, 3)),
    )
    scheduler = SimpleNamespace(step=mock.Mock())
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(
            training_mod, "is_pp_last_stage", return_value=is_last_stage
        ) as is_last,
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
        mock.patch.object(
            training_mod, "logical_and_across_model_parallel_group", return_value=True
        ) as logical_and,
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group=None: value,
        ) as reduce_max,
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or (losses_reduced or []),
            iteration=0,
            **kwargs,
        )
    return SimpleNamespace(
        captured=captured,
        result=result,
        is_last=is_last,
        logical_and=logical_and,
        reduce_max=reduce_max,
        all_reduce=all_reduce,
    )


def test_train_step_forwards_multi_module_schedule_plumbing():
    p2p = object()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": SimpleNamespace()}, language_model_module_name=None
    )
    run = _run(p2p_communicator=p2p, pg_collection=carrier)
    assert run.captured["p2p_communicator"] is p2p
    assert run.captured["pg_collection"] is carrier


def test_train_step_plain_schedule_keeps_legacy_call_shape():
    run = _run(pg_collection=SimpleNamespace(mp=object(), pp=object(), dp_cp=object()))
    assert "p2p_communicator" not in run.captured
    assert "pg_collection" not in run.captured


def test_train_step_plain_reduces_optimizer_outputs():
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    run = _run(exit_before_optimizer=False, pg_collection=pg_collection)
    run.logical_and.assert_called_once_with(True, group=pg_collection.mp)
    assert run.reduce_max.call_count == 2


def test_train_step_multi_trusts_optimizer_outputs_and_encoder_has_no_loss():
    local = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    run = _run(
        exit_before_optimizer=False,
        losses_reduced=[{"unexpected": object()}],
        pg_collection=carrier,
        p2p_communicator=object(),
    )
    run.logical_and.assert_not_called()
    run.reduce_max.assert_not_called()
    run.is_last.assert_not_called()
    assert run.result[0] == {}
    assert run.result[-3:-1] == (2.0, 3)


def test_train_step_multi_reduces_terminal_language_loss_on_language_dp_cp():
    language = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"language": language}, language_model_module_name="language"
    )
    run = _run(
        exit_before_optimizer=False,
        losses_reduced=[{"loss": torch.tensor([6.0, 2.0])}],
        is_last_stage=True,
        pg_collection=carrier,
        p2p_communicator=object(),
    )
    run.is_last.assert_called_once_with(language.pp)
    run.all_reduce.assert_called_once()
    assert run.all_reduce.call_args.kwargs["group"] is language.dp_cp
    assert run.result[0]["loss"].item() == 3.0
