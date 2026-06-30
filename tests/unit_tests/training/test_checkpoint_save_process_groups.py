# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for checkpoint-save process-group forwarding."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import checkpointing


def _save_args(**overrides):
    values = dict(
        async_save=False,
        async_strategy="mcore",
        use_dist_ckpt=True,
        save="checkpoints",
        ckpt_format="torch_dist",
        use_distributed_optimizer=False,
        ckpt_assume_constant_structure=False,
        ckpt_fully_parallel_save=False,
        verify_integrity=False,
        log_progress=False,
        non_persistent_ckpt_type=None,
        non_persistent_local_ckpt_algo="atomic",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@contextmanager
def _common_save_patches(args, *, distributed):
    rerun_state_machine = mock.Mock()
    rerun_state_machine.state_dict.return_value = {}
    patchers = (
        mock.patch.object(checkpointing, "get_args", return_value=args),
        mock.patch.object(checkpointing, "is_empty_async_queue", return_value=True),
        mock.patch.object(checkpointing, "on_save_checkpoint_start", return_value={}),
        mock.patch.object(checkpointing.ft_integration, "on_checkpointing_start"),
        mock.patch.object(checkpointing.ft_integration, "on_checkpointing_end"),
        mock.patch.object(checkpointing, "unwrap_model", side_effect=lambda model: model),
        mock.patch.object(checkpointing, "get_rng_state", return_value={}),
        mock.patch.object(
            checkpointing, "get_rerun_state_machine", return_value=rerun_state_machine
        ),
        mock.patch.object(checkpointing, "get_checkpoint_name", return_value="checkpoint"),
        mock.patch.object(checkpointing, "get_checkpoint_tracker_filename", return_value="tracker"),
        mock.patch.object(checkpointing, "ensure_directory_exists"),
        mock.patch.object(checkpointing, "maybe_save_dataloader_state"),
        mock.patch.object(checkpointing, "_build_sharded_state_dict_metadata", return_value={}),
        mock.patch.object(checkpointing, "generate_state_dict", return_value={}),
        mock.patch.object(checkpointing, "TorchDistSaveShardedStrategy", return_value=object()),
        mock.patch.object(checkpointing, "has_nvidia_modelopt", False),
        mock.patch.object(checkpointing, "is_last_rank", return_value=False),
        mock.patch.object(checkpointing, "on_save_checkpoint_success"),
        mock.patch.object(checkpointing.wandb_utils, "on_save_checkpoint_success"),
        mock.patch.object(
            checkpointing.torch.distributed, "is_initialized", return_value=distributed
        ),
        mock.patch.object(checkpointing.torch.distributed, "get_rank", return_value=0),
        mock.patch.object(checkpointing.torch.distributed, "barrier"),
    )
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield stack.enter_context(mock.patch.object(checkpointing, "print_rank_0"))


@pytest.mark.parametrize(
    "tensor_rank,pipeline_rank,expected_ranks",
    [(None, None, "[ t 3/4, p 2/2 ]"), (6, 7, "[ t 7/4, p 8/2 ]")],
)
def test_async_finalize_uses_supplied_group_numbers_without_mpu(
    tensor_rank, pipeline_rank, expected_ranks
):
    args = _save_args(async_save=True)
    tp_group, pp_group, dp_group, expt_dp_group = object(), object(), object(), object()
    ranks = {tp_group: 2, pp_group: 1, dp_group: 0, expt_dp_group: 0}
    sizes = {tp_group: 4, pp_group: 2}
    async_request = mock.Mock()
    with (
        _common_save_patches(args, distributed=True) as print_rank_0,
        mock.patch.object(checkpointing, "get_pg_rank", side_effect=ranks.__getitem__) as get_rank,
        mock.patch.object(checkpointing, "get_pg_size", side_effect=sizes.__getitem__) as get_size,
        mock.patch.object(
            checkpointing.mpu, "get_tensor_model_parallel_rank", side_effect=AssertionError
        ),
        mock.patch.object(
            checkpointing.mpu, "get_pipeline_model_parallel_rank", side_effect=AssertionError
        ),
        mock.patch.object(
            checkpointing.mpu, "get_tensor_model_parallel_world_size", side_effect=AssertionError
        ),
        mock.patch.object(
            checkpointing.mpu, "get_pipeline_model_parallel_world_size", side_effect=AssertionError
        ),
        mock.patch.object(checkpointing.dist_checkpointing, "save", return_value=async_request),
        mock.patch.object(checkpointing, "schedule_async_save"),
        mock.patch("megatron.training.distillation.get_logits_saver", return_value=None),
        mock.patch.object(checkpointing, "open_file", mock.mock_open()),
    ):
        checkpointing.save_checkpoint(
            5,
            [object()],
            None,
            None,
            0,
            tp_group=tp_group,
            pp_group=pp_group,
            dp_group=dp_group,
            expt_dp_group=expt_dp_group,
            tensor_rank=tensor_rank,
            pipeline_rank=pipeline_rank,
        )
        finalize = async_request.add_finalize_fn.call_args.args[0]
        get_rank.reset_mock()
        get_size.reset_mock()
        get_rank.side_effect = AssertionError("callback must capture integer ranks")
        get_size.side_effect = AssertionError("callback must capture integer sizes")
        print_rank_0.reset_mock()
        finalize()

    assert expected_ranks in print_rank_0.call_args.args[0]
    get_rank.assert_not_called()
    get_size.assert_not_called()


def test_nonpersistent_local_save_uses_supplied_dp_group():
    args = _save_args(use_dist_ckpt=False, ckpt_format="torch", non_persistent_ckpt_type="local")
    dp_cp_group = object()
    manager = mock.Mock(local_ckpt_dir="local-checkpoints")
    state_dict_for_save = object()
    tensor_aware_state_dict = mock.Mock()
    tensor_aware_state_dict.from_state_dict.return_value = (state_dict_for_save, object())
    checkpointing_context = {"local_checkpoint_manager": manager}
    with (
        _common_save_patches(args, distributed=False),
        mock.patch(
            "megatron.core.dist_checkpointing.tensor_aware_state_dict.MCoreTensorAwareStateDict",
            tensor_aware_state_dict,
        ),
        mock.patch.object(
            checkpointing.mpu,
            "get_data_parallel_group",
            side_effect=AssertionError("supplied DP group must avoid MPU"),
        ),
    ):
        checkpointing.save_checkpoint(
            5,
            [object()],
            None,
            None,
            0,
            checkpointing_context=checkpointing_context,
            non_persistent_ckpt=True,
            dp_cp_group=dp_cp_group,
        )

    assert (
        tensor_aware_state_dict.from_state_dict.call_args.kwargs["parallelization_group"]
        is dp_cp_group
    )
    manager.save.assert_called_once_with(state_dict_for_save, 5, is_async=False)
