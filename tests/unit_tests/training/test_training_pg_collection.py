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


def test_resolve_local_pg_collection_preserves_plain_input():
    pg_collection = ProcessGroupCollection()
    assert training_mod._resolve_local_pg_collection(pg_collection) == (pg_collection, "")


def test_resolve_local_pg_collection_materializes_mpu_groups_for_none():
    local = ProcessGroupCollection()
    with mock.patch.object(
        training_mod.ProcessGroupCollection, "use_mpu_process_groups", return_value=local
    ) as use_mpu:
        assert training_mod._resolve_local_pg_collection(None) == (local, "")
    use_mpu.assert_called_once_with()


def test_resolve_local_pg_collection_namespaces_multi_module_rng():
    local = ProcessGroupCollection()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )

    assert training_mod._resolve_local_pg_collection(carrier) == (local, "encoder.")


def test_resolve_local_pg_collection_rejects_colocated_modules():
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": ProcessGroupCollection(), "language": ProcessGroupCollection()},
        language_model_module_name="language",
    )

    with pytest.raises(ValueError, match="exactly one local"):
        training_mod._resolve_local_pg_collection(carrier)


def _local_pg_collection():
    return ProcessGroupCollection(
        tp=object(), pp=object(), dp=object(), dp_cp=object(), expt_dp=object()
    )


def test_setup_model_and_optimizer_uses_local_groups_for_build_load_and_batch_check():
    local = _local_pg_collection()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    args = SimpleNamespace(
        skip_train=True,
        perform_rl_step=False,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        load="checkpoint",
        pretrained_checkpoint=None,
        use_torch_fsdp2=False,
        ckpt_format="torch_dist",
        micro_batch_size=1,
        fp16=False,
        ckpt_convert_format=None,
    )
    builder = mock.Mock()
    model = [object()]
    builder.build_distributed_models.return_value = model
    model_config = SimpleNamespace(
        get_builder_cls=mock.Mock(return_value=mock.Mock(return_value=builder))
    )
    cfg = SimpleNamespace(
        profiling=object(),
        model=model_config,
        ddp=object(),
        optimizer=SimpleNamespace(overlap_param_gather_with_optimizer_step=False),
        dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=True),
    )
    checkpointing_context = {"context": object()}
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "unwrap_model", return_value=model),
        mock.patch.object(training_mod, "load_checkpoint", return_value=(2, 3)) as load,
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "get_current_global_batch_size", return_value=1),
        mock.patch.object(training_mod, "get_pg_size", return_value=4) as get_pg_size,
        mock.patch.object(
            training_mod.mpu,
            "get_data_parallel_world_size",
            side_effect=AssertionError("MIMO setup must use the local DP group"),
        ),
        mock.patch("megatron.training.utils.start_memory_history_recording"),
    ):
        training_mod.setup_model_and_optimizer(
            None,
            None,
            checkpointing_context=checkpointing_context,
            cfg_container=cfg,
            pg_collection=carrier,
        )

    assert builder.build_distributed_models.call_args.kwargs["pg_collection"] is local
    assert load.call_args.args == (model, None, None)
    assert load.call_args.kwargs == {
        "checkpointing_context": checkpointing_context,
        "skip_load_to_model_and_opt": False,
        "tp_group": local.tp,
        "pp_group": local.pp,
        "dp_cp_group": local.dp_cp,
        "dp_group": local.dp,
        "expt_dp_group": local.expt_dp,
        "rng_state_key_prefix": "encoder.",
    }
    get_pg_size.assert_called_once_with(local.dp)


def test_save_checkpoint_and_time_forwards_local_groups_and_rng_prefix():
    local = _local_pg_collection()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"language": local}, language_model_module_name="language"
    )
    args = SimpleNamespace(fp8=False, async_save=True, log_progress=False)
    timers = mock.MagicMock()
    timers.return_value.elapsed.return_value = 1.0
    train_data_iterator = object()
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "should_disable_forward_pre_hook", return_value=False),
        mock.patch.object(training_mod.one_logger_utils, "track_e2e_metrics"),
        mock.patch.object(training_mod.one_logger_utils, "on_save_checkpoint_end"),
        mock.patch.object(training_mod.torch.cuda, "empty_cache"),
        mock.patch.object(training_mod, "num_checkpoints_memory_reported", 0),
        mock.patch.object(training_mod, "MAX_NUM_CHECKPOINTS_MEMORY_REPORTED", 1),
        mock.patch.object(training_mod, "report_memory") as report_memory,
        mock.patch.object(training_mod, "save_checkpoint") as save,
    ):
        training_mod.save_checkpoint_and_time(
            7,
            [object()],
            object(),
            object(),
            12.0,
            {"context": object()},
            train_data_iterator=train_data_iterator,
            pg_collection=carrier,
        )

    save_kwargs = save.call_args.kwargs
    assert save_kwargs["tp_group"] is local.tp
    assert save_kwargs["pp_group"] is local.pp
    assert save_kwargs["dp_group"] is local.dp
    assert save_kwargs["dp_cp_group"] is local.dp_cp
    assert save_kwargs["expt_dp_group"] is local.expt_dp
    assert save_kwargs["rng_state_key_prefix"] == "language."
    assert save_kwargs["train_data_iterator"] is train_data_iterator
    assert report_memory.call_args_list == [
        mock.call("(before save_checkpoint for iteration 7)", process_group=local.dp),
        mock.call("(after save_checkpoint for iteration 7)", process_group=local.dp),
    ]


def test_multi_bootstrap_seeds_and_configures_jit_with_strict_local_groups():
    local = SimpleNamespace(pp=object(), dp=object(), tp=object(), ep=object(), expt_tp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": local}, language_model_module_name=None
    )
    cfg = SimpleNamespace(logger=SimpleNamespace(log_progress=False))
    args = SimpleNamespace(fine_grained_activation_offloading=False)
    with (
        mock.patch.object(
            carrier, "get_only_local_collection", wraps=carrier.get_only_local_collection
        ) as get_only_local,
        mock.patch.object(training_mod.ft_integration, "setup"),
        mock.patch.object(training_mod, "initialize_megatron") as initialize,
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "set_jit_fusion_options") as set_jit,
        mock.patch.object(training_mod, "get_pg_size", return_value=4) as get_pg_size,
        mock.patch.object(training_mod.torch, "tensor", side_effect=_StopAfterBootstrap),
        pytest.raises(_StopAfterBootstrap),
    ):
        training_mod.pretrain(
            cfg,
            train_valid_test_dataset_provider=None,
            model_type=None,
            forward_step_func=None,
            p2p_communicator=object.__new__(MultiModulePipelineCommunicator),
            pg_collection=carrier,
        )

    initialize_kwargs = initialize.call_args.kwargs
    assert "skip_random_seed" not in initialize_kwargs
    assert initialize_kwargs["seed_pp_group"] is local.pp
    assert initialize_kwargs["seed_dp_group"] is local.dp
    assert initialize_kwargs["seed_tp_group"] is local.tp
    assert initialize_kwargs["seed_ep_group"] is local.ep
    assert initialize_kwargs["seed_etp_group"] is local.expt_tp
    get_only_local.assert_called_once_with()
    get_pg_size.assert_called_once_with(local.tp)
    set_jit.assert_called_once_with(tp_size=4)
