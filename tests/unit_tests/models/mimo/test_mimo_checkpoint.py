# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for MIMO distributed checkpoint save/load in non-colocated mode.

Run with 8 GPUs:
    uv run python -m torch.distributed.run --nproc-per-node=8 \
        -m pytest tests/unit_tests/models/mimo/test_mimo_checkpoint.py -v -s
"""

import os
import shutil
import tempfile
from collections import OrderedDict

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
    is_rank_in_grid,
)
from tests.unit_tests.test_utilities import Utils

ENCODER_NAME = "images"


class _CheckpointLeaf(torch.nn.Module):
    """Small parameterized module for exercising nested checkpoint wrappers."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Return the logical checkpoint key for the leaf parameter."""
        return {f'{prefix}weight': self.weight}


class _VersionedCheckpointLeaf(_CheckpointLeaf):
    """Leaf that records state-dict metadata forwarded by wrapper loaders."""

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.loaded_metadata = dict(local_metadata)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _wrap_without_runtime_init(wrapper_type, module):
    """Build only the registered ``module`` nesting needed by checkpoint methods."""
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.module = module
    return wrapper


def _nested_checkpoint_module(leaf, with_float16=True):
    """Build DDP with the optional mixed-precision wrapper used by production MIMO."""
    module = _wrap_without_runtime_init(Float16Module, leaf) if with_float16 else leaf
    ddp_module = _wrap_without_runtime_init(DistributedDataParallel, module)
    ddp_module.tp_group = None
    return ddp_module


def _checkpoint_model(encoder=None, language=None):
    """Build only the logical child structure required by MIMO checkpoint methods."""
    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    model.modality_submodules = torch.nn.ModuleDict(
        {ENCODER_NAME: encoder} if encoder is not None else {}
    )
    model.language_model = language
    return model


@pytest.mark.parametrize('with_float16', [False, True])
@pytest.mark.parametrize('module_name', [ENCODER_NAME, 'language_model'])
def test_nested_ddp_float16_sharded_state_loads_strictly(module_name, with_float16):
    """MIMO checkpoint keys must stay logical across nested runtime wrappers."""
    leaf = _CheckpointLeaf()
    ddp_module = _nested_checkpoint_module(leaf, with_float16=with_float16)
    if module_name == ENCODER_NAME:
        model = _checkpoint_model(encoder=ddp_module)
    else:
        model = _checkpoint_model(language=ddp_module)

    sharded_state = model.sharded_state_dict(metadata={'dp_cp_group': object()})
    logical_prefix = (
        f'modality_submodules.{module_name}.'
        if module_name == ENCODER_NAME
        else 'language_model.'
    )
    assert list(sharded_state) == [f'{logical_prefix}weight']
    loaded_state = {key: torch.full_like(value, 7) for key, value in sharded_state.items()}

    model.load_state_dict(loaded_state, strict=True)

    torch.testing.assert_close(leaf.weight, torch.full_like(leaf.weight, 7))


def test_ddp_sharded_state_dict_falls_back_for_plain_module(mocker):
    """Transparent DDP checkpointing must retain support for ordinary nn.Module children."""
    leaf = torch.nn.Linear(2, 2)
    ddp_module = _wrap_without_runtime_init(DistributedDataParallel, leaf)
    ddp_module.tp_group = object()
    metadata = {'dp_cp_group': object()}
    sharded_offsets = ((0, 0, 1),)
    logical_state = {'logical.weight': leaf.weight}
    default_sharded_state = mocker.patch(
        'megatron.core.distributed.distributed_data_parallel.sharded_state_dict_default',
        return_value=logical_state,
        create=True,
    )

    assert (
        ddp_module.sharded_state_dict('logical.', sharded_offsets, metadata) is logical_state
    )
    default_sharded_state.assert_called_once_with(
        leaf,
        'logical.',
        sharded_offsets,
        metadata,
        tp_group=ddp_module.tp_group,
    )


def test_mimo_load_state_dict_aggregates_logical_incompatibilities():
    """Non-strict loads report child and root incompatibilities with logical prefixes."""
    encoder = _nested_checkpoint_module(_CheckpointLeaf())
    language = _nested_checkpoint_module(_CheckpointLeaf())
    model = _checkpoint_model(encoder=encoder, language=language)

    incompatible = model.load_state_dict(
        {
            'language_model.weight': torch.ones(1),
            'language_model.extra': torch.ones(1),
            'root_extra': torch.ones(1),
        },
        strict=False,
    )

    assert incompatible.missing_keys == [f'modality_submodules.{ENCODER_NAME}.weight']
    assert incompatible.unexpected_keys == ['language_model.extra', 'root_extra']


def test_mimo_load_state_dict_rebases_metadata_and_forwards_assign():
    """Logical child loads retain version metadata and standard assign semantics."""
    leaf = _VersionedCheckpointLeaf()
    model = _checkpoint_model(language=_nested_checkpoint_module(leaf))
    state_dict = OrderedDict({'language_model.weight': torch.full_like(leaf.weight, 9)})
    state_dict._metadata = {'language_model': {'version': 17}}

    model.load_state_dict(state_dict, assign=True)

    assert leaf.loaded_metadata == {'version': 17, 'assign_to_params_buffers': True}
    torch.testing.assert_close(leaf.weight, torch.full_like(leaf.weight, 9))


def _get_shared_tmpdir():
    """Create a shared temp directory across all ranks."""
    tmpdir_list = [None]
    if dist.get_rank() == 0:
        tmpdir_list[0] = tempfile.mkdtemp(prefix="mimo_ckpt_test_")
    dist.broadcast_object_list(tmpdir_list, src=0)
    return tmpdir_list[0]


def _cleanup_tmpdir(tmpdir):
    """Clean up temp directory (rank 0 only)."""
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _randomize_params(model, seed):
    """Set all model parameters to deterministic random values."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.random_()


def _create_model_and_optimizer(encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed):
    """Create MIMO model with DDP + optimizer, do a fake step to populate optimizer state.

    Caller must call create_all_embedding_groups() before this function.
    """
    torch.manual_seed(seed)

    mimo_model, _, _, _, _ = get_mimo_model(
        encoder_name=ENCODER_NAME,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=64,
    )
    _randomize_params(mimo_model, seed)

    # Use Float16Optimizer (not DistributedOptimizer) to exercise the MIMO-specific
    # param_groups/grad_scaler extraction in sharded_state_dict. DistributedOptimizer
    # handles its own checkpointing internally and our code is transparent to it.
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=False,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    # Fake backward + step to populate optimizer state (Adam m/v)
    for param in mimo_model.parameters():
        param.grad = torch.randn_like(param)
    optimizer.step()

    return mimo_model, optimizer


def run_checkpoint_test(
    encoder_tp,
    encoder_pp,
    encoder_dp,
    encoder_offset,
    llm_tp,
    llm_pp,
    llm_dp,
    llm_offset,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
):
    """Save model + optimizer checkpoint, load into fresh instances, verify match."""
    # Clear NVTE env vars that the conftest set_env fixture sets to '0'.
    # GPTModel (LanguageModule) asserts these are unset or match the attention backend.
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_grid = create_hypercomm_grid(
        offset=encoder_offset, tp=encoder_tp, cp=1, pp=encoder_pp, dp=encoder_dp
    )
    llm_grid = create_hypercomm_grid(offset=llm_offset, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
    create_all_embedding_groups([encoder_grid, llm_grid])

    # --- Create model A + optimizer, snapshot state ---
    model_a, optimizer_a = _create_model_and_optimizer(
        encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed=1
    )
    params_a = {name: p.clone() for name, p in model_a.named_parameters()}

    ckpt_dir = _get_shared_tmpdir()
    try:
        model_ckpt = os.path.join(ckpt_dir, 'model')
        optim_ckpt = os.path.join(ckpt_dir, 'optimizer')
        if dist.get_rank() == 0:
            os.makedirs(model_ckpt)
            os.makedirs(optim_ckpt)
        dist.barrier()

        # Save model
        save(model_a.sharded_state_dict(), model_ckpt)

        # Save optimizer (needs fresh model sharded_state_dict since save() consumes tensor refs).
        # validate_access_integrity=True is the regression guard for the _get_replica_id fix:
        # without including TP rank in replica_id, every TP rank at (pp=0, dp=0) would emit
        # the same `_mimo_*` ShardedObject as a main replica, producing duplicate-key errors.
        optim_sd_a = optimizer_a.sharded_state_dict(model_a.sharded_state_dict(), is_loading=False)
        save(optim_sd_a, optim_ckpt, validate_access_integrity=True)

        dist.barrier()

        # --- Create model B + optimizer with different weights (reuse same grids) ---
        model_b, optimizer_b = _create_model_and_optimizer(
            encoder_grid, llm_grid, hidden_size, num_layers, vocab_size, seed=2
        )

        # Load model
        model_sd_b = model_b.sharded_state_dict()
        loaded_model_sd, missing, unexpected = load(
            model_sd_b, model_ckpt, strict=StrictHandling.RETURN_ALL
        )
        real_missing = [k for k in missing if '_extra_state' not in k]
        real_unexpected = [k for k in unexpected if '_extra_state' not in k]
        assert not real_missing, f"Missing keys: {real_missing}"
        assert not real_unexpected, f"Unexpected keys: {real_unexpected}"
        model_b.load_state_dict(loaded_model_sd)

        # Load optimizer
        optim_sd_b = optimizer_b.sharded_state_dict(model_b.sharded_state_dict(), is_loading=True)
        loaded_optim_sd = load(optim_sd_b, optim_ckpt, validate_access_integrity=False)
        optimizer_b.load_state_dict(loaded_optim_sd)

        # --- Verify model params match ---
        mismatches = [
            name
            for name, p in model_b.named_parameters()
            if name in params_a and not torch.equal(p, params_a[name])
        ]
        assert not mismatches, f"Model param mismatch after load: {mismatches}"

        # --- Verify optimizer state matches (param_groups + Adam m/v tensors) ---
        for name, info_b in optimizer_b.module_infos.items():
            if not (info_b.is_active and info_b.optimizer):
                continue
            info_a = optimizer_a.module_infos[name]
            sd_a = info_a.optimizer.state_dict()
            sd_b = info_b.optimizer.state_dict()

            # Verify param_groups
            pg_a = sd_a.get('optimizer', {}).get('param_groups', [])
            pg_b = sd_b.get('optimizer', {}).get('param_groups', [])
            assert len(pg_a) == len(pg_b), f"Optimizer {name}: param_groups count mismatch"
            for i, (ga, gb) in enumerate(zip(pg_a, pg_b)):
                assert ga['lr'] == gb['lr'], f"Optimizer {name} group[{i}]: lr mismatch"

            # Verify Adam state tensors (exp_avg, exp_avg_sq)
            state_a = sd_a.get('optimizer', {}).get('state', {})
            state_b = sd_b.get('optimizer', {}).get('state', {})
            for param_id in state_a:
                if param_id not in state_b:
                    continue
                for key in ('exp_avg', 'exp_avg_sq'):
                    if key in state_a[param_id] and key in state_b[param_id]:
                        assert torch.equal(
                            state_a[param_id][key], state_b[param_id][key]
                        ), f"Optimizer {name} param {param_id} {key} mismatch"

    finally:
        _cleanup_tmpdir(ckpt_dir)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimoCheckpoint:
    """Distributed checkpoint save/load tests for non-colocated MiMo (8 GPUs)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_expert_view_shares_pipeline_groups_with_base_view(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        grid = create_hypercomm_grid(tp=2, dp=2, pp=2)

        base_pp_ranks = dist.get_process_group_ranks(grid.get_pg("pp"))
        expert_pp_ranks = dist.get_process_group_ranks(grid.get_pg("pp", view="expert"))

        assert base_pp_ranks == expert_pp_ranks

    def test_encoder_tp2_llm_tp2_pp3(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=2,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=3,
            llm_dp=1,
            llm_offset=2,
            hidden_size=256,
            num_layers=3,
        )

    def test_encoder_tp1_llm_pp7(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=1,
            encoder_pp=1,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=1,
            llm_pp=7,
            llm_dp=1,
            llm_offset=1,
            hidden_size=256,
            num_layers=7,
        )

    def test_encoder_tp2_pp2_llm_tp2_pp2(self):
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_checkpoint_test(
            encoder_tp=2,
            encoder_pp=2,
            encoder_dp=1,
            encoder_offset=0,
            llm_tp=2,
            llm_pp=2,
            llm_dp=1,
            llm_offset=4,
            hidden_size=256,
            num_layers=2,
        )


class TestOptimizerCheckpointHelpers:
    """CPU-only coverage for the dist-checkpoint extract/restore helpers."""

    def test_replica_id_includes_context_parallel_rank(self):
        from unittest.mock import Mock

        from megatron.core.models.mimo.optimizer import _get_replica_id

        pg_collection = ProcessGroupCollection(
            tp=Mock(rank=Mock(return_value=1)),
            cp=Mock(rank=Mock(return_value=2)),
            pp=Mock(rank=Mock(return_value=3)),
            dp=Mock(rank=Mock(return_value=4)),
        )

        assert _get_replica_id(pg_collection) == (1, 3, 4, 2)

    @staticmethod
    def _extract_param_state_sharding_type(*args, **kwargs):
        from megatron.core.models.mimo.optimizer import _extract_param_state_sharding_type

        return _extract_param_state_sharding_type(*args, **kwargs)

    @staticmethod
    def _restore_param_state_sharding_type(*args, **kwargs):
        from megatron.core.models.mimo.optimizer import _restore_param_state_sharding_type

        return _restore_param_state_sharding_type(*args, **kwargs)

    @staticmethod
    def _extract_param_groups(*args, **kwargs):
        from megatron.core.models.mimo.optimizer import _extract_param_groups

        return _extract_param_groups(*args, **kwargs)

    @staticmethod
    def _extract_grad_scaler(*args, **kwargs):
        from megatron.core.models.mimo.optimizer import _extract_grad_scaler

        return _extract_grad_scaler(*args, **kwargs)

    @staticmethod
    def _restore_param_groups(*args, **kwargs):
        from megatron.core.models.mimo.optimizer import _restore_param_groups

        return _restore_param_groups(*args, **kwargs)

    def test_extract_param_state_sharding_type_wraps_value_into_sharded_object(self):
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        sub_sd = {'param_state_sharding_type': 'fully_sharded_bucket_space'}

        self._extract_param_state_sharding_type(sub_sd, 'images', '.1', replica_id=(0, 0, 0))

        assert 'param_state_sharding_type' not in sub_sd
        wrapped = sub_sd['_mimo_param_state_sharding_type.1']
        assert isinstance(wrapped, ShardedObject)
        assert wrapped.key == 'optimizer.mimo.images.1.param_state_sharding_type'
        assert wrapped.data == 'fully_sharded_bucket_space'
        assert wrapped.replica_id == (0, 0, 0)

    def test_extract_param_state_sharding_type_noop_when_missing(self):
        sub_sd = {'unrelated': 1}

        self._extract_param_state_sharding_type(sub_sd, 'images', '', replica_id=0)

        assert sub_sd == {'unrelated': 1}

    def test_restore_param_state_sharding_type_renames_suffixed_key(self):
        sub_sd = {'_mimo_param_state_sharding_type.0': 'fully_sharded_bucket_space'}

        self._restore_param_state_sharding_type(sub_sd)

        assert sub_sd == {'param_state_sharding_type': 'fully_sharded_bucket_space'}

    def test_restore_param_state_sharding_type_noop_when_missing(self):
        sub_sd = {'unrelated': 1}

        self._restore_param_state_sharding_type(sub_sd)

        assert sub_sd == {'unrelated': 1}

    def test_extract_param_groups_deletes_empty_optimizer_dict(self):
        sub_sd = {'optimizer': {'param_groups': [{'lr': 0.1, 'params': [0]}]}}

        self._extract_param_groups(sub_sd, 'images', '', replica_id=0)

        assert 'optimizer' not in sub_sd
        assert '_mimo_param_groups' in sub_sd

    def test_extract_param_groups_keeps_optimizer_when_other_keys_remain(self):
        sub_sd = {
            'optimizer': {'param_groups': [{'lr': 0.1, 'params': [0]}], 'state': {0: {'step': 5}}}
        }

        self._extract_param_groups(sub_sd, 'images', '', replica_id=0)

        assert sub_sd['optimizer'] == {'state': {0: {'step': 5}}}
        assert '_mimo_param_groups' in sub_sd

    def test_extract_grad_scaler_preserves_distributed_optimizer_sharded_object(self):
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        grad_scaler = ShardedObject(
            key='optimizer.distributed.dp_group_idx_0.grad_scaler',
            data={'scale': 65536.0},
            global_shape=(1,),
            global_offset=(0,),
            replica_id=(0, 0, 0),
        )
        sub_sd = {'grad_scaler': grad_scaler}

        self._extract_grad_scaler(sub_sd, 'images', '', replica_id=(0, 0, 0, 0))

        assert sub_sd == {'grad_scaler': grad_scaler}

    def test_extract_grad_scaler_wraps_raw_scaler_state(self):
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        sub_sd = {'grad_scaler': {'scale': 65536.0}}

        self._extract_grad_scaler(sub_sd, 'images', '', replica_id=(0, 0, 0, 0))

        wrapped = sub_sd['_mimo_grad_scaler']
        assert isinstance(wrapped, ShardedObject)
        assert wrapped.data == {'scale': 65536.0}
        assert 'grad_scaler' not in sub_sd

    def test_restore_param_groups_recreates_missing_optimizer_wrapper(self):
        from unittest.mock import MagicMock

        inner_optimizer = MagicMock()
        inner_optimizer.optimizer.state_dict.return_value = {
            'param_groups': [{'lr': 0.1, 'params': [42, 43]}]
        }
        sub_sd = {'_mimo_param_groups': [{'lr': 0.1, 'params': []}]}

        self._restore_param_groups(sub_sd, inner_optimizer, 'images')

        assert sub_sd['optimizer']['param_groups'][0]['params'] == [42, 43]
        assert '_mimo_param_groups' not in sub_sd
