# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO per-rank runtime setup (RNG seeding, bucket sizing, DDP wrapping)."""

import argparse
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.training.builder import MimoBuildConfig, MimoModelBuilder
from examples.mimo.training.runtime import configure_module_rng, wrap_active_modules_with_ddp
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import unwrap_model
from megatron.training.training import resolve_ddp_bucket_size
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


@pytest.mark.parametrize(
    "config, overlap, num_params, expected",
    [
        # num_buckets divides the param count.
        (DistributedDataParallelConfig(num_buckets=4), True, 128, 128 // 4),
        # explicit bucket_size passes through.
        (DistributedDataParallelConfig(bucket_size=4096), True, 256, 4096),
        # overlap off -> None, regardless of bucket_size.
        (DistributedDataParallelConfig(bucket_size=4096), False, 256, None),
        # no explicit size with group=None (dp size 1) -> the sane default.
        (DistributedDataParallelConfig(), True, 256, max(40_000_000, 1_000_000)),
    ],
)
def test_resolve_ddp_bucket_size(config, overlap, num_params, expected):
    """The MIMO wrap delegates bucket sizing to this shared get_model helper."""
    assert resolve_ddp_bucket_size(config, None, overlap, num_params) == expected


def test_builder_forwards_caller_ddp_config(mocker):
    """The model builder must not replace the config selected by the training container."""
    args = _args()
    topology = mocker.Mock()
    language_pg = mocker.Mock()
    model = mocker.Mock()
    builder = MimoModelBuilder(MimoBuildConfig(_topology=topology, _args=args))
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)

    mocker.patch(
        "examples.mimo.training.builder._resolve_role",
        return_value=(None, True, False, language_pg, None),
    )
    mocker.patch("examples.mimo.training.builder._seed_module_rng")
    mocker.patch.object(builder, "build_model", return_value=model)
    wrap = mocker.patch("examples.mimo.training.builder.wrap_active_modules_with_ddp")
    mocker.patch("examples.mimo.training.builder.configure_grad_sync")
    mocker.patch("examples.mimo.training.builder._mimo_branch_name", return_value="language")

    assert builder.build_distributed_models(
        mocker.Mock(),
        ddp_config=ddp_config,
        overlap_param_gather_with_optimizer_step=True,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        data_parallel_random_init=True,
    ) == [model]
    wrap.assert_called_once_with(
        args,
        model,
        topology,
        ddp_config,
        overlap_param_gather_with_optimizer_step=True,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        data_parallel_random_init=True,
    )


def test_builder_requires_ddp_config_before_building(mocker):
    """DDP wrapping must fail clearly before allocating a MIMO model without a config."""
    builder = MimoModelBuilder(MimoBuildConfig(_topology=mocker.Mock(), _args=_args()))
    build_model = mocker.patch.object(builder, "build_model")

    with pytest.raises(ValueError, match="ddp_config is required"):
        builder.build_distributed_models(mocker.Mock(), ddp_config=None, wrap_with_ddp=True)

    build_model.assert_not_called()


def test_runtime_delegates_to_shared_ddp_wrapper(mocker):
    """MIMO supplies the role model/PGC and caller flags to the shared wrapper lifecycle."""
    module = torch.nn.Linear(2, 2)
    module.config = SimpleNamespace(fp16=False, bf16=False)
    mimo_model = SimpleNamespace(language_model=module, modality_submodules={})
    pg_collection = mocker.Mock()
    topology = SimpleNamespace(module_pgs={MIMO_LANGUAGE_MODULE_KEY: pg_collection})
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    wrapped = mocker.Mock()
    shared_ddp_wrap = mocker.patch(
        "examples.mimo.training.runtime._ddp_wrap", create=True, return_value=[wrapped]
    )
    mocker.patch("examples.mimo.training.runtime.print_rank_0")

    wrap_active_modules_with_ddp(
        _args(),
        mimo_model,
        topology,
        ddp_config,
        overlap_param_gather_with_optimizer_step=True,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        data_parallel_random_init=True,
    )

    assert mimo_model.language_model is wrapped
    shared_ddp_wrap.assert_called_once()
    call = shared_ddp_wrap.call_args
    assert call.args[0] == [module]
    assert call.args[1] is True
    assert call.args[2] is not ddp_config
    assert call.args[2] == ddp_config
    assert call.args[3] is True
    assert call.args[4] is True
    assert call.args[5] is False
    assert call.kwargs == {"pg_collection": pg_collection}


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
class TestRuntimeDistributed:
    def setup_method(self, method):
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_distinct_offsets_give_distinct_rng_states(self):
        # Encoder tp=2,dp=2 at 0-3; language tp=2,pp=2 at 4-7. Each rank seeds the one
        # module it participates in; distinct role offsets must reseed every tracked state.
        topo = create_topology(
            [
                ModuleGridSpec(name=ENCODER, num_ranks=4, tp=2, rank_offset=0),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, tp=2, pp=2, rank_offset=4
                ),
            ]
        )
        try:
            module = MIMO_LANGUAGE_MODULE_KEY if torch.distributed.get_rank() >= 4 else ENCODER
            pgc = topo.module_pgs[module]
            configure_module_rng(_args(), pgc, role_seed_offset=10)
            states_a = get_cuda_rng_tracker().get_states()
            configure_module_rng(_args(), pgc, role_seed_offset=20)
            states_b = get_cuda_rng_tracker().get_states()
            assert set(states_a) == set(states_b)
            for name in states_a:
                assert not torch.equal(states_a[name], states_b[name])
        finally:
            topo.destroy()

    @pytest.mark.parametrize("use_distributed_optimizer", [False, True])
    def test_active_module_preserves_caller_ddp_config(self, use_distributed_optimizer):
        topo = _eight_gpu_topology()
        try:
            mimo_model = _build_unwrapped_mimo_model(topo)
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=use_distributed_optimizer,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                bucket_size=12345,
                check_for_nan_in_grad=True,
                check_for_large_grads=True,
                average_in_collective=True,
            )
            original_values = {
                field.name: getattr(ddp_config, field.name)
                for field in dataclasses.fields(DistributedDataParallelConfig)
            }
            wrap_active_modules_with_ddp(_args(), mimo_model, topo, ddp_config)
            # Non-colocated: each rank owns exactly one active module (language XOR encoder).
            if torch.distributed.get_rank() < 4:
                active = mimo_model.modality_submodules[ENCODER]
                assert mimo_model.language_model is None
                expected_role_overrides = {
                    "overlap_grad_reduce": False,
                    "overlap_param_gather": False,
                    "bucket_size": None,
                }
            else:
                active = mimo_model.language_model
                assert ENCODER not in mimo_model.modality_submodules
                expected_role_overrides = {}
            assert isinstance(active, DistributedDataParallel)
            assert active.ddp_config is not ddp_config
            for field in dataclasses.fields(DistributedDataParallelConfig):
                expected = expected_role_overrides.get(field.name, original_values[field.name])
                assert getattr(active.ddp_config, field.name) == expected
            for name, original_value in original_values.items():
                assert getattr(ddp_config, name) == original_value
        finally:
            topo.destroy()

    def test_data_parallel_random_init_broadcasts_active_module(self):
        topo = _eight_gpu_topology()
        try:
            mimo_model = _build_unwrapped_mimo_model(topo)
            rank = torch.distributed.get_rank()
            if rank < 4:
                module_name = ENCODER
                active = mimo_model.modality_submodules[ENCODER]
            else:
                module_name = MIMO_LANGUAGE_MODULE_KEY
                active = mimo_model.language_model
            with torch.no_grad():
                next(active.parameters()).fill_(rank)

            wrap_active_modules_with_ddp(
                _args(),
                mimo_model,
                topo,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                data_parallel_random_init=True,
            )

            active = (
                mimo_model.language_model
                if module_name == MIMO_LANGUAGE_MODULE_KEY
                else mimo_model.modality_submodules[module_name]
            )
            value = next(active.parameters()).flatten()[0].detach().clone()
            gathered = [torch.empty_like(value) for _ in range(topo.module_pgs[module_name].dp.size())]
            torch.distributed.all_gather(
                gathered, value, group=topo.module_pgs[module_name].dp
            )
            assert all(torch.equal(item, gathered[0]) for item in gathered)
        finally:
            topo.destroy()

    def test_bf16_wraps_in_float16module_and_freezes_targets(self):
        topo = _eight_gpu_topology()
        try:
            # bf16 -> Float16Module wrap; --freeze-vit freezes the encoder backbone only.
            mimo_model = _build_unwrapped_mimo_model(topo, bf16=True)
            wrap_active_modules_with_ddp(
                _args(fp32=False, freeze_vit=True),
                mimo_model,
                topo,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
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
