# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MIMO memory optimization: recompute and fine-grained offload.

Tests each cell of the recompute/offload support matrix plus mix/match integration.
Each test validates:
  1. Correctness: gradients match a no-optimization baseline (torch.allclose)
  2. Functionality: forward/backward completes without error

Run with:
    uv run python -m torch.distributed.run --nproc_per_node=8 -m pytest \
        tests/unit_tests/models/test_mimo_memory_optimization.py -v
"""

import gc
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
from megatron.core.models.mimo.colocated_schedule import colocated_forward_backward_with_pp
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLPSubmodules
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
# Grid / PG helpers
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


def _make_encoder_spec(
    pg_collection, num_layers=ENCODER_LAYERS, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS
):
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    proj_cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    proj_cfg.ffn_hidden_size = hidden_size
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
                        "input_size": hidden_size,
                        "tp_group": pg_collection.tp,
                    },
                )
            ],
        },
    )


def _make_lm_spec(
    pg_collection,
    num_layers=LLM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    seq_length=SEQ_LENGTH,
):
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1

    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
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
            "vocab_size": vocab_size,
            "max_sequence_length": seq_length,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": pg_collection,
        },
    )


def create_mimo_model(
    encoder_grid,
    llm_grid,
    memory_config=None,
    use_ddp=True,
    enc_layers=ENCODER_LAYERS,
    enc_hidden=HIDDEN_SIZE,
    enc_heads=NUM_HEADS,
    llm_layers=LLM_LAYERS,
    llm_hidden=HIDDEN_SIZE,
    llm_heads=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    seq_length=SEQ_LENGTH,
):
    """Create a MIMO model with optional memory config, optionally DDP-wrapped."""
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_pg = get_pg_collection(encoder_grid, is_language_model=False)
    llm_pg = get_pg_collection(llm_grid, is_language_model=True)

    tp_rank = dist.get_rank(llm_pg.tp) if llm_pg.tp else 0
    model_parallel_cuda_manual_seed(
        42, tp_rank=tp_rank, ep_rank=0, etp_rank=0, force_reset_rng=True
    )
    torch.manual_seed(42)

    encoder_spec = _make_encoder_spec(
        encoder_pg, num_layers=enc_layers, hidden_size=enc_hidden, num_heads=enc_heads
    )
    lm_spec = _make_lm_spec(
        llm_pg,
        num_layers=llm_layers,
        hidden_size=llm_hidden,
        num_heads=llm_heads,
        vocab_size=vocab_size,
        seq_length=seq_length,
    )

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

    if use_ddp:
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

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if use_ddp:
                if model.language_model is not None:
                    stack.enter_context(model.language_model.no_sync())
                for sub in model.modality_submodules.values():
                    if sub is not None:
                        stack.enter_context(sub.no_sync())
            yield

    def finalize_grads_func(*args, force_all_reduce=False, **kwargs):
        if use_ddp:
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


def make_batch(
    device='cuda',
    seq_length=SEQ_LENGTH,
    image_seq_length=IMAGE_SEQ_LENGTH,
    hidden_size=HIDDEN_SIZE,
    micro_batch_size=MICRO_BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    image_token_id=IMAGE_TOKEN_ID,
):
    """Create a synthetic VLM batch."""
    input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
    input_ids[:, :image_seq_length] = image_token_id

    labels = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
    loss_mask = torch.ones(micro_batch_size, seq_length, device=device)
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(micro_batch_size, -1)

    pixel_values = torch.randn(
        image_seq_length, micro_batch_size, hidden_size, device=device, dtype=torch.bfloat16
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

    def __init__(self, seed=123, **batch_kwargs):
        self.seed = seed
        self.call_count = 0
        self.batch_kwargs = batch_kwargs

    def __iter__(self):
        return self

    def __next__(self):
        torch.manual_seed(self.seed + self.call_count)
        self.call_count += 1
        return make_batch(**self.batch_kwargs)


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


def collect_grads(model):
    """Collect gradients from all parameters (handles Megatron DDP main_grad)."""
    grads = {}
    for name, param in model.named_parameters():
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            grad = param.grad
        if grad is not None:
            grads[name] = grad.float().clone()
    return grads


def zero_grads(model):
    """Zero all gradients including main_grad."""
    for param in model.parameters():
        if hasattr(param, 'main_grad') and param.main_grad is not None:
            param.main_grad.zero_()
        if param.grad is not None:
            param.grad.zero_()


def run_fwd_bwd(model, llm_pg, seed=42, num_microbatches=NUM_MICROBATCHES, **batch_kwargs):
    """Run forward-backward and return collected gradients."""
    seq_length = batch_kwargs.get('seq_length', SEQ_LENGTH)
    micro_batch_size = batch_kwargs.get('micro_batch_size', MICRO_BATCH_SIZE)
    data_iter = BatchIterator(seed=seed, **batch_kwargs)

    schedule.forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=data_iter,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        pg_collection=llm_pg,
    )

    return collect_grads(model)


def _reset_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _run_one_mimo_iter(model, llm_pg, seed, batch_kwargs, num_microbatches):
    """Run one fwd/bwd iteration on the same model instance, return (grads, peak_bytes).

    Mirrors the pattern from test_fine_grained_activation_offloading.py:
    reset_peak_memory_stats -> fwd/bwd -> synchronize -> capture peak.
    """
    # Zero existing grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    torch.cuda.reset_peak_memory_stats()
    grads = run_fwd_bwd(model, llm_pg, seed=seed, num_microbatches=num_microbatches, **batch_kwargs)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return grads, peak_bytes


def run_baseline_and_optimized(enc_grid, llm_grid, memory_config, seed=42, **model_kwargs):
    """Create baseline and optimized models, warmup + measure steady-state on each.

    Follows the same pattern as test_fine_grained_activation_offloading.py:
      1. Build model
      2. Warmup iteration (allocator warm, caches populated)
      3. _reset_cuda_memory (gc + empty_cache)
      4. Measurement iteration (reset_peak -> fwd/bwd -> capture peak)
      5. Delete model + cleanup

    Returns (grads_baseline, grads_optimized, peak_mem_baseline, peak_mem_optimized).
    """
    seq_length = model_kwargs.get('seq_length', SEQ_LENGTH)
    enc_hidden = model_kwargs.get('enc_hidden', HIDDEN_SIZE)
    image_seq_length = model_kwargs.get('image_seq_length', IMAGE_SEQ_LENGTH)
    micro_batch_size = model_kwargs.get('micro_batch_size', MICRO_BATCH_SIZE)
    num_microbatches = model_kwargs.get('num_microbatches', NUM_MICROBATCHES)
    vocab_size = model_kwargs.get('vocab_size', VOCAB_SIZE)

    batch_kwargs = dict(
        seq_length=seq_length,
        hidden_size=enc_hidden,
        image_seq_length=image_seq_length,
        micro_batch_size=micro_batch_size,
        vocab_size=vocab_size,
    )

    create_kwargs = {
        k: v
        for k, v in model_kwargs.items()
        if k
        in (
            'enc_layers',
            'enc_hidden',
            'enc_heads',
            'llm_layers',
            'llm_hidden',
            'llm_heads',
            'vocab_size',
            'seq_length',
        )
    }

    # --- Optimized first (avoids CUDA allocator caching bias on baseline) ---
    model_opt, _, llm_pg = create_mimo_model(
        enc_grid, llm_grid, memory_config=memory_config, use_ddp=False, **create_kwargs
    )
    model_opt.train()

    # Warmup iteration (same model instance — allocator warm, TE caches populated)
    _run_one_mimo_iter(
        model_opt,
        llm_pg,
        seed=seed + 1000,
        batch_kwargs=batch_kwargs,
        num_microbatches=num_microbatches,
    )
    _reset_cuda_memory()

    # Steady-state measurement iteration
    grads_opt, peak_mem_opt = _run_one_mimo_iter(
        model_opt, llm_pg, seed=seed, batch_kwargs=batch_kwargs, num_microbatches=num_microbatches
    )

    del model_opt
    _reset_cuda_memory()

    # --- Baseline ---
    model_base, _, llm_pg = create_mimo_model(
        enc_grid, llm_grid, memory_config=None, use_ddp=False, **create_kwargs
    )
    model_base.train()

    # Warmup
    _run_one_mimo_iter(
        model_base,
        llm_pg,
        seed=seed + 1000,
        batch_kwargs=batch_kwargs,
        num_microbatches=num_microbatches,
    )
    _reset_cuda_memory()

    # Steady-state measurement
    grads_base, peak_mem_base = _run_one_mimo_iter(
        model_base, llm_pg, seed=seed, batch_kwargs=batch_kwargs, num_microbatches=num_microbatches
    )

    del model_base
    _reset_cuda_memory()

    return grads_base, grads_opt, peak_mem_base, peak_mem_opt


def assert_grads_match(grads_base, grads_opt, atol=1e-2, rtol=1e-2):
    """Assert gradient correctness: optimized grads match baseline within tolerance.

    Uses relaxed tolerance (1e-2) because bf16 recomputation can introduce
    small numerical differences due to non-deterministic TE kernels.
    """
    # Verify we got gradients from both runs
    assert len(grads_base) > 0, "Baseline produced no gradients"
    assert len(grads_opt) > 0, "Optimized model produced no gradients"

    matched = 0
    for name in grads_base:
        if name not in grads_opt:
            continue
        g_base = grads_base[name]
        g_opt = grads_opt[name]
        assert (
            g_base.shape == g_opt.shape
        ), f"Shape mismatch for {name}: {g_base.shape} vs {g_opt.shape}"
        assert torch.isfinite(g_opt).all(), f"Non-finite gradient in optimized model for {name}"
        assert torch.allclose(g_base, g_opt, atol=atol, rtol=rtol), (
            f"Gradient mismatch for {name}: "
            f"max_diff={torch.max(torch.abs(g_base - g_opt)).item():.6f}, "
            f"base_norm={g_base.norm().item():.6f}, opt_norm={g_opt.norm().item():.6f}"
        )
        matched += 1

    assert matched > 0, "No matching parameter names between baseline and optimized"


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
# Correctness tests — baseline comparison for each recompute config
# ============================================================================


class TestEncoderRecomputeCorrectness:
    """Verify encoder recompute produces identical gradients to baseline."""

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
        grads_base, grads_opt, mem_base, mem_opt = run_baseline_and_optimized(
            enc_grid, llm_grid, memory_config
        )
        assert_grads_match(grads_base, grads_opt)

    def test_encoder_recompute_selective(self, homogeneous_grids):
        """Selective recompute on encoder: only core_attn checkpointed."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            )
        }
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestLLMRecomputeCorrectness:
    """Verify LLM recompute produces identical gradients to baseline."""

    def test_llm_recompute_selective(self, homogeneous_grids):
        """Selective recompute on LLM: core_attn checkpointed."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {
            MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            )
        }
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestProjectionRecomputeCorrectness:
    """Verify projection recompute produces identical gradients to baseline."""

    def test_recompute_projection_mlp(self, homogeneous_grids):
        """Recompute projection: MLP intermediates freed during forward."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(recompute_projection=True)}
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestMixedRecomputeCorrectness:
    """Verify combined encoder + LLM + projection recompute produces correct gradients."""

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
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestMemoryReduction:
    """Verify recompute saves GPU memory with analytically-bounded expectations.

    Uses a larger model (hidden=1024, 8 encoder layers, seq=512) where
    activation savings dominate over checkpoint overhead.

    Analytical model for full recompute savings on the encoder:
      Without recompute: each layer saves ~N activation tensors for backward.
        With TE flash attention, per-layer cost ≈ (10*S*B*H + 2*S*B*4H) * dtype_bytes
        = 18 * S * B * H * dtype_bytes  (dominant terms)
      With full recompute (recompute_num_layers=L): only 1 input tensor saved
        = S * B * H * dtype_bytes for the entire block

      Expected savings ≈ L * 18 * S * B * H * dtype_bytes (minus the one saved input)

    We use a generous tolerance (50%) since TE kernels may save additional
    tensors and the allocator has jitter.
    """

    # Larger model dimensions for measurable memory signal
    MEM_ENC_LAYERS = 8
    MEM_ENC_HIDDEN = 1024
    MEM_ENC_HEADS = 8
    MEM_LLM_LAYERS = 2  # Keep LLM small — we're measuring encoder savings
    MEM_LLM_HIDDEN = 1024
    MEM_LLM_HEADS = 8
    MEM_SEQ_LENGTH = 512
    MEM_IMAGE_SEQ = 256
    MEM_BATCH = 2

    def test_encoder_full_recompute_saves_memory(self, homogeneous_grids):
        """Full encoder recompute saves measurable GPU memory."""
        enc_grid, llm_grid = homogeneous_grids
        L = self.MEM_ENC_LAYERS

        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full', recompute_method='uniform', recompute_num_layers=L
            )
        }

        model_kwargs = dict(
            enc_layers=L,
            enc_hidden=self.MEM_ENC_HIDDEN,
            enc_heads=self.MEM_ENC_HEADS,
            llm_layers=self.MEM_LLM_LAYERS,
            llm_hidden=self.MEM_LLM_HIDDEN,
            llm_heads=self.MEM_LLM_HEADS,
            seq_length=self.MEM_SEQ_LENGTH,
            image_seq_length=self.MEM_IMAGE_SEQ,
            micro_batch_size=self.MEM_BATCH,
            num_microbatches=1,  # Single microbatch for cleaner measurement
        )

        grads_base, grads_opt, mem_base, mem_opt = run_baseline_and_optimized(
            enc_grid, llm_grid, memory_config, **model_kwargs
        )

        mem_base_mb = mem_base / (1024**2)
        mem_opt_mb = mem_opt / (1024**2)
        savings_mb = mem_base_mb - mem_opt_mb

        # Analytical expected savings (conservative lower bound)
        # Per-layer activation: ~18 * S * B * H * 2 bytes (bf16)
        # With TP=2, effective H per rank = H/2 for some tensors, but many are full H
        # Use ~10 * S * B * H * 2 as conservative per-layer estimate
        S, B, H = self.MEM_IMAGE_SEQ, self.MEM_BATCH, self.MEM_ENC_HIDDEN
        conservative_per_layer_bytes = 10 * S * B * H * 2
        expected_savings_mb = (L * conservative_per_layer_bytes) / (1024**2)

        if dist.get_rank() == 0:
            print(
                f"\nMemory test (enc {L}L h={H} seq={S} bs={B}):\n"
                f"  Baseline peak:  {mem_base_mb:.1f} MB\n"
                f"  Optimized peak: {mem_opt_mb:.1f} MB\n"
                f"  Actual savings: {savings_mb:.1f} MB\n"
                f"  Expected (analytical lower bound): {expected_savings_mb:.1f} MB"
            )

        # Assert positive savings
        assert savings_mb > 0, (
            f"Recompute did not reduce memory: "
            f"baseline={mem_base_mb:.1f}MB, optimized={mem_opt_mb:.1f}MB"
        )

        # Assert savings are in the right ballpark (within 50% of analytical estimate)
        assert savings_mb >= expected_savings_mb * 0.5, (
            f"Memory savings ({savings_mb:.1f}MB) less than 50% of expected "
            f"({expected_savings_mb:.1f}MB). Something may be wrong with recompute."
        )

        # Correctness must still hold
        assert_grads_match(grads_base, grads_opt)


class TestOffloadProjectionCorrectness:
    """Verify offload_projection produces identical gradients to baseline.

    Uses FineGrainedActivationOffloadingInterface to offload the projection's
    saved-for-backward tensors (including encoder output) to CPU.
    """

    def test_offload_projection(self, homogeneous_grids):
        """Offload projection saved tensors to CPU via fine-grained interface."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(offload_projection=True)}
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestOffloadEncoderOutputCorrectness:
    """Verify offload_encoder_output produces identical gradients to baseline.

    Uses FineGrainedActivationOffloadingInterface to offload the encoder's
    saved-for-backward tensors to CPU.
    """

    def test_offload_encoder_output(self, homogeneous_grids):
        """Offload encoder saved tensors to CPU via fine-grained interface."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(offload_encoder_output=True)}
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestProjectionOutputRecomputeCorrectness:
    """Verify recompute_projection_output produces identical gradients."""

    def test_recompute_projection_output(self, homogeneous_grids):
        """Checkpoint around align_embeddings: discards combined_embeddings intermediate."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(recompute_projection_output=True)}
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestProjectionOutputOffloadCorrectness:
    """Verify offload_projection_output produces identical gradients."""

    def test_offload_projection_output(self, homogeneous_grids):
        """Offload align_embeddings saved tensors to CPU."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(offload_projection_output=True)}
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


class TestMixedRecomputeAndOffloadCorrectness:
    """Verify combined recompute + offload produces correct gradients."""

    def test_encoder_recompute_plus_offload_projection(self, homogeneous_grids):
        """Encoder full recompute + projection offload combined."""
        enc_grid, llm_grid = homogeneous_grids
        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=ENCODER_LAYERS,
                offload_projection=True,
            )
        }
        grads_base, grads_opt, _, _ = run_baseline_and_optimized(enc_grid, llm_grid, memory_config)
        assert_grads_match(grads_base, grads_opt)


# ============================================================================
# PP>1 colocated tests
# ============================================================================

PP_HIDDEN = 256
PP_HEADS = 8
PP_LAYERS = 2
PP_VOCAB = 1000
PP_SEQ = 64
PP_IMAGE_SEQ = 32
PP_MBS = 2
PP_NMB = 4
PP_IMAGE_TOKEN = 50257


def _run_pp_colocated(enc_tp, enc_dp, llm_tp, llm_dp, llm_pp, memory_config=None, num_iterations=2):
    """Run colocated PP>1 training and return param grads from all ranks.

    Creates model, runs colocated_forward_backward_with_pp for num_iterations,
    and collects gradients. Uses no DDP for clean gradient access.
    """
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_grid = create_hypercomm_grid(offset=0, tp=enc_tp, dp=enc_dp)
    llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, pp=llm_pp, dp=llm_dp)
    create_all_embedding_groups([encoder_grid, llm_grid])

    encoder_pg = get_pg_collection(encoder_grid, is_language_model=False)
    llm_pg = get_pg_collection(llm_grid, is_language_model=True)

    tp_rank = dist.get_rank(llm_pg.tp) if llm_pg.tp else 0
    model_parallel_cuda_manual_seed(
        42, tp_rank=tp_rank, ep_rank=0, etp_rank=0, force_reset_rng=True
    )
    torch.manual_seed(42)

    encoder_spec = _make_encoder_spec(
        encoder_pg, num_layers=PP_LAYERS, hidden_size=PP_HIDDEN, num_heads=PP_HEADS
    )
    lm_spec = _make_lm_spec(
        llm_pg,
        num_layers=PP_LAYERS,
        hidden_size=PP_HIDDEN,
        num_heads=PP_HEADS,
        vocab_size=PP_VOCAB,
        seq_length=PP_SEQ,
    )

    mimo_config = MimoModelConfig(
        language_model_spec=lm_spec,
        modality_submodules_spec={ENCODER_NAME: encoder_spec},
        special_token_ids={ENCODER_NAME: PP_IMAGE_TOKEN},
        module_to_grid_map={ENCODER_NAME: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
        memory_config=memory_config,
    )

    model = MimoModel(mimo_config)
    model.to(torch.device("cuda")).to(torch.bfloat16)
    model.model_type = ModelType.encoder_or_decoder
    model.train()

    # Schedule callbacks (no DDP for clean gradients)
    @contextmanager
    def no_sync_func():
        yield

    def finalize_grads_func(*args, **kwargs):
        pass

    model.config.no_sync_func = no_sync_func
    model.config.finalize_model_grads_func = finalize_grads_func
    model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    pp_group = llm_grid.get_pg("pp")
    p2p = P2PCommunicator(pp_group=pp_group, config=model.config)

    data_iter = BatchIterator(
        seed=42,
        seq_length=PP_SEQ,
        image_seq_length=PP_IMAGE_SEQ,
        hidden_size=PP_HIDDEN,
        micro_batch_size=PP_MBS,
        vocab_size=PP_VOCAB,
        image_token_id=PP_IMAGE_TOKEN,
    )

    for iteration in range(num_iterations):
        losses = colocated_forward_backward_with_pp(
            mimo_model=model,
            data_iterator=data_iter,
            num_microbatches=PP_NMB,
            encoder_grid=encoder_grid,
            llm_grid=llm_grid,
            encoder_name=ENCODER_NAME,
            seq_length=PP_SEQ,
            micro_batch_size=PP_MBS,
            p2p_communicator=p2p,
            pg_collection=llm_pg,
        )

    # Verify losses on last PP stage
    if is_pp_last_stage(pp_group):
        assert losses is not None and len(losses) > 0, "No losses on last PP stage"
        for loss_dict in losses:
            val = loss_dict.get('loss_reduced', 0)
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert val == val, "NaN loss"

    grads = collect_grads(model)

    del model
    _reset_cuda_memory()
    destroy_all_grids()

    return grads


class TestColocatedPPMemoryOptimization:
    """Test recompute and offload with colocated PP>1 (three-phase schedule).

    Uses encoder TP2/DP4/PP1 + LLM TP2/DP2/PP2 on 8 GPUs.
    The three-phase schedule detaches encoder output between Phase 1 and Phase 2.
    Recompute on encoder works because Phase 3's backward replays through checkpoint.
    Offload works because we defer off_interface.reset() until after Phase 3.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()

    def test_pp_encoder_full_recompute(self):
        """Encoder full recompute with PP>1: verify training completes with valid grads."""
        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=PP_LAYERS,
            )
        }
        grads = _run_pp_colocated(
            enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=2, llm_pp=2, memory_config=memory_config
        )
        assert len(grads) > 0, "No gradients produced"
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"Non-finite gradient for {name}"

    def test_pp_recompute_projection(self):
        """Projection recompute with PP>1."""
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(recompute_projection=True)}
        grads = _run_pp_colocated(
            enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=2, llm_pp=2, memory_config=memory_config
        )
        assert len(grads) > 0, "No gradients produced"
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"Non-finite gradient for {name}"

    def test_pp_offload_projection(self):
        """Projection offload with PP>1: tests deferred reset across phases."""
        memory_config = {ENCODER_NAME: ModuleMemoryConfig(offload_projection=True)}
        grads = _run_pp_colocated(
            enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=2, llm_pp=2, memory_config=memory_config
        )
        assert len(grads) > 0, "No gradients produced"
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"Non-finite gradient for {name}"

    def test_pp_mixed_recompute_and_offload(self):
        """Encoder recompute + projection offload + LLM recompute with PP>1."""
        memory_config = {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=PP_LAYERS,
                offload_projection=True,
            ),
            MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            ),
        }
        grads = _run_pp_colocated(
            enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=2, llm_pp=2, memory_config=memory_config
        )
        assert len(grads) > 0, "No gradients produced"
        for name, g in grads.items():
            assert torch.isfinite(g).all(), f"Non-finite gradient for {name}"
