# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO per-rank model construction and distributed preparation."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.training.runtime import prepare_active_modules_for_distributed_training
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import unwrap_model
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    get_language_model_spec,
    get_vision_submodules_spec,
)
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


def _args(**overrides):
    base = dict(
        seed=1234, image_token_id=100, fp32=True, ddp_num_buckets=None, ddp_bucket_size=None
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_unwrapped_mimo_model(topo, bf16=False):
    """Build a bare (un-DDP-wrapped) MimoModel over a HeteroTopology's per-module PGCs."""
    mimo_config = MimoModelConfig(
        language_model_spec=get_language_model_spec(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            vocab_size=128,
            seq_len=8,
            pg_collection=topo.module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            bf16=bf16,
        ),
        modality_submodules_spec={
            ENCODER: get_vision_submodules_spec(
                num_layers=2,
                hidden_size=16,
                num_attention_heads=4,
                language_hidden_size=16,
                pg_collection=topo.module_pgs[ENCODER],
                bf16=bf16,
            )
        },
        special_token_ids={ENCODER: 50257},
        module_to_grid_map=topo.grids,
    )
    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda"))
    return mimo_model


def _eight_gpu_topology():
    """Encoder dp=4 at ranks 0-3; language dp=4 at ranks 4-7 (non-colocated, tiles world)."""
    return create_topology(
        [
            ModuleGridSpec(name=ENCODER, num_ranks=4, rank_offset=0),
            ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, rank_offset=4),
        ]
    )


def test_builder_seeds_meta_build_and_forwards_lifecycle(mocker):
    """The non-colocated builder seeds and prepares the one active child."""
    from examples.mimo.training.builder import (
        _LANGUAGE_SEED_OFFSET,
        MimoBuildConfig,
        MimoModelBuilder,
    )

    args = _args(init_model_with_meta_device=True)
    groups = mocker.Mock()
    child = SimpleNamespace(config=SimpleNamespace(init_model_with_meta_device=True))
    model = SimpleNamespace(language_model=child, modality_submodules={})
    topology, ddp_config = mocker.Mock(), DistributedDataParallelConfig()
    builder = MimoModelBuilder(MimoBuildConfig(_topology=topology, _args=args))
    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(None, True, False, groups, None),
    )
    mocker.patch.object(builder, "build_model", return_value=model)
    prepare = mocker.patch(
        "examples.mimo.training.builder.prepare_active_modules_for_distributed_training"
    )
    mocker.patch("examples.mimo.training.builder.configure_grad_sync")
    torch_device = mocker.patch(
        "examples.mimo.training.builder.torch.device", return_value=mocker.MagicMock()
    )
    set_seed = mocker.patch("examples.mimo.training.runtime._set_random_seed")

    assert builder.build_distributed_models(
        mocker.Mock(), ddp_config=ddp_config, data_parallel_random_init=True
    ) == [model]

    torch_device.assert_called_once_with("meta")
    set_seed.assert_called_once_with(
        args.seed + _LANGUAGE_SEED_OFFSET,
        True,
        False,
        False,
        use_cudagraphable_rng=False,
        pp_group=groups.pp,
        dp_group=groups.dp,
        tp_group=groups.tp,
        ep_group=groups.ep,
        etp_group=groups.expt_tp,
    )
    call = prepare.call_args
    assert call.args == (args, model, topology)
    assert call.kwargs["ddp_config"] is ddp_config
    assert call.kwargs["built_with_meta_device"] is True
    assert call.kwargs["data_parallel_random_init"] is True


@pytest.mark.parametrize(
    ("role", "wrapper_kind"),
    [
        ("language", "custom"),
        ("encoder", "default"),
    ],
)
def test_active_runtime_uses_shared_lifecycle_with_copied_role_config(
    mocker, role, wrapper_kind
):
    """Each active child keeps its PGC, wrapper policy, and a role-safe DDP config copy."""
    from examples.mimo.training.runtime import _EncoderFloat16Module

    module = torch.nn.Linear(2, 2)
    module.config = SimpleNamespace(
        fp16=False, bf16=role == "encoder", init_model_with_meta_device=False
    )
    if role == "language":
        mimo_model = SimpleNamespace(language_model=module, modality_submodules={})
        module_name = MIMO_LANGUAGE_MODULE_KEY
    else:
        mimo_model = SimpleNamespace(language_model=None, modality_submodules={ENCODER: module})
        module_name = ENCODER
    mixed_precision_wrapper = {
        "default": Float16Module,
        "custom": mocker.sentinel.mixed_precision_wrapper,
        "none": None,
    }[wrapper_kind]
    pg_collection = mocker.Mock()
    topology = SimpleNamespace(module_pgs={module_name: pg_collection})
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, overlap_param_gather=True)
    shared_lifecycle = mocker.patch(
        "examples.mimo.training.runtime.prepare_existing_model_chunks_for_distributed_training",
        return_value=[mocker.Mock()],
    )

    prepare_active_modules_for_distributed_training(
        _args(),
        mimo_model,
        topology,
        ddp_config=ddp_config,
        built_with_meta_device=False,
        data_parallel_random_init=True,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )

    call = shared_lifecycle.call_args
    assert call.args[2] is pg_collection
    copied_config = call.kwargs["ddp_config"]
    assert copied_config is not ddp_config
    expected_overlap = role == "language"
    assert copied_config.overlap_grad_reduce is expected_overlap
    assert copied_config.overlap_param_gather is expected_overlap
    assert ddp_config.overlap_grad_reduce is True
    assert ddp_config.overlap_param_gather is True
    assert call.kwargs["data_parallel_random_init"] is True
    expected_wrapper = (
        _EncoderFloat16Module
        if role == "encoder" and mixed_precision_wrapper is Float16Module
        else mixed_precision_wrapper
    )
    assert call.kwargs["mixed_precision_wrapper"] is expected_wrapper


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestRuntimeDistributed:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_active_module_is_ddp_over_its_own_grid(self):
        topo = _eight_gpu_topology()
        try:
            mimo_model = _build_unwrapped_mimo_model(topo)
            prepare_active_modules_for_distributed_training(
                _args(),
                mimo_model,
                topo,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                built_with_meta_device=False,
            )
            # Non-colocated: each rank owns exactly one active module (language XOR encoder).
            if torch.distributed.get_rank() < 4:
                active = mimo_model.modality_submodules[ENCODER]
                assert mimo_model.language_model is None
            else:
                active = mimo_model.language_model
                assert ENCODER not in mimo_model.modality_submodules
            assert isinstance(active, DistributedDataParallel)
        finally:
            topo.destroy()

    def test_bf16_wraps_in_float16module_and_freezes_targets(self):
        topo = _eight_gpu_topology()
        try:
            # bf16 -> Float16Module wrap; --freeze-vit freezes the encoder backbone only.
            mimo_model = _build_unwrapped_mimo_model(topo, bf16=True)
            prepare_active_modules_for_distributed_training(
                _args(fp32=False, freeze_vit=True),
                mimo_model,
                topo,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                built_with_meta_device=False,
            )

            if torch.distributed.get_rank() < 4:
                active = mimo_model.modality_submodules[ENCODER]
                # Float16Module sits under DDP, above the bare submodule.
                assert isinstance(active.module, Float16Module)
                submodule = unwrap_model(active)
                # --freeze-vit froze the encoder backbone, not the projector.
                assert all(not p.requires_grad for p in submodule.encoders.parameters())
                assert all(p.requires_grad for p in submodule.input_projections.parameters())
            else:
                assert isinstance(mimo_model.language_model.module, Float16Module)
        finally:
            topo.destroy()
