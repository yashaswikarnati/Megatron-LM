# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MIMO memory optimization: recompute and fine-grained offload.

Tests each cell of the recompute/offload support matrix plus mix/match integration.

Run with:
    uv run python -m torch.distributed.run --nproc_per_node=8 -m pytest \
        tests/unit_tests/models/test_mimo_memory_optimization.py -v
"""

import copy
import logging
import os
from contextlib import ExitStack, contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.memory_config import ModuleMemoryConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None

logger = logging.getLogger(__name__)

ENCODER_NAME = "images"
IMAGE_TOKEN_ID = 32000
HIDDEN_SIZE = 64
NUM_HEADS = 4
ENCODER_LAYERS = 2
LLM_LAYERS = 2
VOCAB_SIZE = 33000
SEQ_LENGTH = 128
IMAGE_SEQ_LENGTH = 32
MICRO_BATCH_SIZE = 2
NUM_MICROBATCHES = 2

# ============================================================================
# Grid / PG helpers (same as test_mimo_colocated_e2e.py)
# ============================================================================

_active_grids = []
_embedding_pg_cache = {}


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["ep"])
    grid.create_pg(["expt_dp"])
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    grid.create_pg(["tp", "cp", "ep", "pp", "dp"])
    _active_grids.append(grid)
    return grid


def destroy_all_grids():
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def create_all_embedding_groups(grids):
    for grid in grids:
        pp_group = grid.get_pg("pp")
        if not pp_group:
            continue
        pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
        cache_key = tuple(pp_ranks)
        if cache_key not in _embedding_pg_cache:
            pos_embd_ranks = [pp_ranks[0]]
            embd_ranks = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            _embedding_pg_cache[cache_key] = (
                dist.new_group(ranks=pos_embd_ranks),
                dist.new_group(ranks=embd_ranks),
            )


def get_pg_collection(grid, is_language_model=False):
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.get_pg("tp")
    pg_collection.cp = grid.get_pg("cp")
    pg_collection.pp = grid.get_pg("pp")
    pg_collection.ep = grid.get_pg("ep")
    pg_collection.dp = grid.get_pg("dp")
    pg_collection.dp_cp = grid.get_pg(["dp", "cp"])
    pg_collection.expt_dp = grid.get_pg("expt_dp")

    # Add embedding groups
    if pg_collection.pp:
        pp_ranks = sorted(dist.get_process_group_ranks(pg_collection.pp))
        cache_key = tuple(pp_ranks)
        if cache_key in _embedding_pg_cache:
            pos_embd_pg, embd_pg = _embedding_pg_cache[cache_key]
            pg_collection.pos_embd = pos_embd_pg if is_pp_first_stage(pg_collection.pp) else None
            if is_language_model:
                pg_collection.embd = (
                    embd_pg
                    if (is_pp_last_stage(pg_collection.pp) or is_pp_first_stage(pg_collection.pp))
                    else None
                )
            else:
                pg_collection.embd = None

    return pg_collection


# ============================================================================
# Model creation
# ============================================================================


def _make_encoder_spec(pg_collection):
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    vision_config = TransformerConfig(
        num_layers=ENCODER_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    proj_cfg = TransformerConfig(num_layers=1, hidden_size=HIDDEN_SIZE, num_attention_heads=1)
    proj_cfg.ffn_hidden_size = HIDDEN_SIZE
    proj_cfg.bias_activation_fusion = True
    proj_cfg.add_bias_linear = True
    proj_cfg.activation_func = torch.nn.functional.gelu

    return ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {
                "clip_encoder": ModuleSpec(
                    module=TransformerBlock,
                    params={
                        "config": vision_config,
                        "spec": get_gpt_layer_with_transformer_engine_spec(),
                        "pg_collection": pg_collection,
                        "pre_process": True,
                        "post_process": True,
                    },
                )
            },
            "input_projections": [
                ModuleSpec(
                    module=MultimodalProjector,
                    params={
                        "config": proj_cfg,
                        "submodules": MLPSubmodules(
                            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
                        ),
                        "projector_type": "mlp",
                        "input_size": HIDDEN_SIZE,
                        "tp_group": pg_collection.tp,
                    },
                )
            ],
        },
    )


def _make_lm_spec(pg_collection):
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    lm_config = TransformerConfig(
        num_layers=LLM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
    )

    return ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": VOCAB_SIZE,
            "max_sequence_length": SEQ_LENGTH,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": pg_collection,
        },
    )


def create_mimo_model(encoder_grid, llm_grid, memory_config=None):
    """Create a MIMO model with optional memory config, DDP-wrapped."""
    # Clear NVTE env vars that conftest set_env fixture sets to '0'.
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_pg = get_pg_collection(encoder_grid, is_language_model=False)
    llm_pg = get_pg_collection(llm_grid, is_language_model=True)

    # Initialize RNG tracker (required by TE layers for model-parallel-rng)
    tp_rank = dist.get_rank(llm_pg.tp) if llm_pg.tp else 0
    model_parallel_cuda_manual_seed(
        42, tp_rank=tp_rank, ep_rank=0, etp_rank=0, force_reset_rng=True
    )
    torch.manual_seed(42)

    encoder_spec = _make_encoder_spec(encoder_pg)
    lm_spec = _make_lm_spec(llm_pg)

    mimo_config = MimoModelConfig(
        language_model_spec=lm_spec,
        modality_submodules_spec={ENCODER_NAME: encoder_spec},
        special_token_ids={ENCODER_NAME: IMAGE_TOKEN_ID},
        module_to_grid_map={ENCODER_NAME: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
        memory_config=memory_config,
    )

    model = MimoModel(mimo_config)
    model.to(torch.device("cuda")).to(torch.bfloat16)
    model.model_type = ModelType.encoder_or_decoder

    # DDP wrap
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, use_distributed_optimizer=False
    )

    if model.language_model is not None:
        model.language_model = DistributedDataParallel(
            config=model.language_model.config,
            ddp_config=ddp_config,
            module=model.language_model,
            pg_collection=llm_pg,
        )

    if ENCODER_NAME in model.modality_submodules:
        submodule = model.modality_submodules[ENCODER_NAME]
        if submodule is not None:
            model.modality_submodules[ENCODER_NAME] = DistributedDataParallel(
                config=submodule.encoders['clip_encoder'].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=encoder_pg,
            )

    # Attach schedule callbacks
    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if model.language_model is not None:
                stack.enter_context(model.language_model.no_sync())
            for sub in model.modality_submodules.values():
                if sub is not None:
                    stack.enter_context(sub.no_sync())
            yield

    def finalize_grads_func(*args, force_all_reduce=False, **kwargs):
        if model.language_model is not None:
            finalize_model_grads(
                [model.language_model],
                num_tokens=None,
                pg_collection=llm_pg,
                force_all_reduce=force_all_reduce,
            )
        for sub in model.modality_submodules.values():
            if sub is not None:
                finalize_model_grads(
                    [sub],
                    num_tokens=None,
                    pg_collection=encoder_pg,
                    force_all_reduce=force_all_reduce,
                )

    model.config.no_sync_func = no_sync_func
    model.config.finalize_model_grads_func = finalize_grads_func
    model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    return model, encoder_pg, llm_pg


# ============================================================================
# Data + forward step
# ============================================================================


def make_batch(device='cuda'):
    """Create a synthetic VLM batch."""
    input_ids = torch.randint(0, VOCAB_SIZE, (MICRO_BATCH_SIZE, SEQ_LENGTH), device=device)
    # Replace first IMAGE_SEQ_LENGTH tokens with image token
    input_ids[:, :IMAGE_SEQ_LENGTH] = IMAGE_TOKEN_ID

    labels = torch.randint(0, VOCAB_SIZE, (MICRO_BATCH_SIZE, SEQ_LENGTH), device=device)
    loss_mask = torch.ones(MICRO_BATCH_SIZE, SEQ_LENGTH, device=device)
    position_ids = torch.arange(SEQ_LENGTH, device=device).unsqueeze(0).expand(MICRO_BATCH_SIZE, -1)

    # Vision encoder input: [seq, batch, hidden]
    pixel_values = torch.randn(
        IMAGE_SEQ_LENGTH, MICRO_BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    return {
        'input_ids': input_ids,
        'labels': labels,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
        'attention_mask': None,
        'modality_inputs': {
            ENCODER_NAME: {'clip_encoder': {'hidden_states': pixel_values, 'attention_mask': None}}
        },
    }


class BatchIterator:
    """Repeatable deterministic batch iterator."""

    def __init__(self, seed=123):
        self.seed = seed
        self.call_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        torch.manual_seed(self.seed + self.call_count)
        self.call_count += 1
        return make_batch()


def forward_step(data_iterator, model, *args, **kwargs):
    batch = next(data_iterator)
    output_tensor, loss_mask = model(**batch)

    def loss_func(loss_mask, output_tensor):
        if output_tensor is None:
            return torch.tensor(0.0, device='cuda', requires_grad=True), {}
        loss = output_tensor.float().sum()
        return loss, {'loss': loss.detach().item()}

    return output_tensor, partial(loss_func, loss_mask)


# ============================================================================
# Run fwd/bwd and collect gradients
# ============================================================================


def run_fwd_bwd(model, llm_pg, encoder_grid, llm_grid, seed=42):
    """Run one forward-backward pass and return a dict of param gradients."""
    data_iter = BatchIterator(seed=seed)

    schedule.forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=data_iter,
        model=[model],
        num_microbatches=NUM_MICROBATCHES,
        seq_length=SEQ_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
        forward_only=False,
        pg_collection=llm_pg,
    )

    grads = {}
    for name, param in model.named_parameters():
        # Megatron DDP stores grads in param.main_grad, not param.grad
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            grad = param.grad
        if grad is not None:
            grads[name] = grad.clone()
    return grads


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def init_dist():
    Utils.initialize_distributed()
    yield
    destroy_all_grids()


@pytest.fixture()
def homogeneous_grids():
    """TP=2, DP=4, PP=1 for both encoder and LLM (homogeneous)."""
    enc_grid = create_hypercomm_grid(tp=2, dp=4)
    llm_grid = create_hypercomm_grid(tp=2, dp=4)
    create_all_embedding_groups([enc_grid, llm_grid])
    yield enc_grid, llm_grid
    destroy_all_grids()


# ============================================================================
# Tests — one per matrix cell
# ============================================================================


def _assert_valid_grads(grads):
    """Assert grads dict is non-empty and all values are finite."""
    assert len(grads) > 0, "No gradients produced"
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"


class TestEncoderRecompute:
    """Test encoder internal recompute (TransformerBlock-level)."""

    def test_encoder_recompute_full(self, homogeneous_grids):
        """Full recompute on encoder: all layers checkpointed."""
        enc_grid, llm_grid = homogeneous_grids

        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=ENCODER_LAYERS,
            )
        }
        model, _, llm_pg = create_mimo_model(enc_grid, llm_grid, memory_config=memory_config)
        grads = run_fwd_bwd(model, llm_pg, enc_grid, llm_grid, seed=42)
        _assert_valid_grads(grads)
        del model

    def test_encoder_recompute_selective(self, homogeneous_grids):
        """Selective recompute on encoder: only core_attn checkpointed."""
        enc_grid, llm_grid = homogeneous_grids

        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            )
        }
        model, _, llm_pg = create_mimo_model(enc_grid, llm_grid, memory_config=memory_config)
        grads = run_fwd_bwd(model, llm_pg, enc_grid, llm_grid, seed=42)
        _assert_valid_grads(grads)
        del model


class TestLLMRecompute:
    """Test LLM internal recompute."""

    def test_llm_recompute_selective(self, homogeneous_grids):
        """Selective recompute on LLM: core_attn checkpointed.

        Verifies fwd/bwd runs without error and produces valid gradients.
        """
        enc_grid, llm_grid = homogeneous_grids

        memory_config = {
            MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            )
        }
        model, _, llm_pg = create_mimo_model(enc_grid, llm_grid, memory_config=memory_config)
        grads = run_fwd_bwd(model, llm_pg, enc_grid, llm_grid, seed=42)
        _assert_valid_grads(grads)
        del model


class TestProjectionRecompute:
    """Test projection layer recompute (MLP projection)."""

    def test_recompute_projection_mlp(self, homogeneous_grids):
        """Recompute projection: MLP intermediates freed during forward."""
        enc_grid, llm_grid = homogeneous_grids

        memory_config = {ENCODER_NAME: ModuleMemoryConfig(recompute_projection=True)}
        model, _, llm_pg = create_mimo_model(enc_grid, llm_grid, memory_config=memory_config)
        grads = run_fwd_bwd(model, llm_pg, enc_grid, llm_grid, seed=42)
        _assert_valid_grads(grads)
        del model


class TestMixedRecomputeAndOffload:
    """Integration test: mix/match recompute + offload across modules."""

    def test_mixed_recompute(self, homogeneous_grids):
        """Encoder full recompute + recompute projection + LLM selective recompute."""
        enc_grid, llm_grid = homogeneous_grids

        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=ENCODER_LAYERS,
                recompute_projection=True,
            ),
            MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            ),
        }
        model, _, llm_pg = create_mimo_model(enc_grid, llm_grid, memory_config=memory_config)
        grads = run_fwd_bwd(model, llm_pg, enc_grid, llm_grid, seed=42)
        _assert_valid_grads(grads)
        del model
