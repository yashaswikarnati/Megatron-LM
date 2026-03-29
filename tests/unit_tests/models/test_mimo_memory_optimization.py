# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MIMO memory optimization: recompute and fine-grained offload.

Each test in the support matrix is a (memory_config, schedule_mode) pair driven
by ``pytest.mark.parametrize``.  A single oracle function handles model creation,
warmup, steady-state measurement, gradient correctness, and optional memory checks.

Run with:
    uv run python -m torch.distributed.run --nproc_per_node=8 -m pytest \
        tests/unit_tests/models/test_mimo_memory_optimization.py -v
"""

import gc
import logging
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import pytest
import torch
import torch.distributed as dist

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.colocated_schedule import colocated_forward_backward_with_pp
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.memory_config import ModuleMemoryConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
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

# ============================================================================
# Test configuration: schedule modes + model sizing
# ============================================================================


@dataclass(frozen=True)
class ScheduleConfig:
    """Defines parallelism layout and model sizing for a test run."""

    enc_tp: int
    enc_dp: int
    llm_tp: int
    llm_dp: int
    llm_pp: int = 1
    hidden_size: int = 64
    num_heads: int = 4
    enc_layers: int = 2
    llm_layers: int = 2
    vocab_size: int = 33000
    seq_length: int = 128
    image_seq_length: int = 32
    micro_batch_size: int = 2
    num_microbatches: int = 2
    image_token_id: int = 32000

    @property
    def is_pp(self) -> bool:
        return self.llm_pp > 1


# Standard configs used across tests
PP1_COLOCATED = ScheduleConfig(enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=4)
PP2_COLOCATED = ScheduleConfig(
    enc_tp=2,
    enc_dp=4,
    llm_tp=2,
    llm_dp=2,
    llm_pp=2,
    hidden_size=256,
    num_heads=8,
    vocab_size=1000,
    seq_length=64,
    image_seq_length=32,
    num_microbatches=4,
    image_token_id=50257,
)
# Larger model for memory measurement
PP1_MEMORY = ScheduleConfig(
    enc_tp=2,
    enc_dp=4,
    llm_tp=2,
    llm_dp=4,
    hidden_size=1024,
    num_heads=8,
    enc_layers=8,
    llm_layers=2,
    seq_length=512,
    image_seq_length=256,
    num_microbatches=1,
)


# ============================================================================
# Memory configs from the support matrix
# ============================================================================

MEMORY_CONFIGS: Dict[str, Dict[str, ModuleMemoryConfig]] = {
    "enc_recompute_full": {
        ENCODER_NAME: ModuleMemoryConfig(
            recompute_granularity='full', recompute_method='uniform', recompute_num_layers=2
        )
    },
    "enc_recompute_selective": {
        ENCODER_NAME: ModuleMemoryConfig(
            recompute_granularity='selective', recompute_modules=['core_attn']
        )
    },
    "llm_recompute_selective": {
        MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
            recompute_granularity='selective', recompute_modules=['core_attn']
        )
    },
    "recompute_projection": {ENCODER_NAME: ModuleMemoryConfig(recompute_projection=True)},
    "recompute_projection_output": {
        ENCODER_NAME: ModuleMemoryConfig(recompute_projection_output=True)
    },
    "offload_projection": {ENCODER_NAME: ModuleMemoryConfig(offload_projection=True)},
    "offload_encoder_output": {ENCODER_NAME: ModuleMemoryConfig(offload_encoder_output=True)},
    "offload_projection_output": {ENCODER_NAME: ModuleMemoryConfig(offload_projection_output=True)},
    "enc_offload_attn_norm": {ENCODER_NAME: ModuleMemoryConfig(offload_modules=['attn_norm'])},
    "llm_offload_attn_norm": {
        MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(offload_modules=['attn_norm'])
    },
    "mixed_recompute": {
        ENCODER_NAME: ModuleMemoryConfig(
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=2,
            recompute_projection=True,
        ),
        MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
            recompute_granularity='selective', recompute_modules=['core_attn']
        ),
    },
    "mixed_recompute_and_offload": {
        ENCODER_NAME: ModuleMemoryConfig(
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=2,
            offload_projection=True,
        )
    },
}

# ============================================================================
# Infrastructure: grids, model creation, data, measurement
# ============================================================================

_active_grids = []
_embedding_pg_cache = {}


def _create_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    for pg_dims in (
        ["tp"],
        ["cp"],
        ["pp"],
        ["dp"],
        ["dp", "cp"],
        ["ep"],
        ["expt_dp"],
        ["tp", "pp"],
        ["tp", "ep", "pp"],
        ["dp", "ep"],
        ["tp", "cp", "ep", "pp", "dp"],
    ):
        grid.create_pg(pg_dims)
    _active_grids.append(grid)
    return grid


def _destroy_all_grids():
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def _create_embedding_groups(grids):
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


def _get_pg_collection(grid, is_language_model=False):
    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.ep = grid.get_pg("ep")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.expt_dp = grid.get_pg("expt_dp")
    if pg.pp:
        pp_ranks = sorted(dist.get_process_group_ranks(pg.pp))
        cache_key = tuple(pp_ranks)
        if cache_key in _embedding_pg_cache:
            pos_embd_pg, embd_pg = _embedding_pg_cache[cache_key]
            pg.pos_embd = pos_embd_pg if is_pp_first_stage(pg.pp) else None
            pg.embd = (
                embd_pg
                if is_language_model and (is_pp_last_stage(pg.pp) or is_pp_first_stage(pg.pp))
                else None
            )
    return pg


def _build_mimo_model(sc: ScheduleConfig, memory_config=None):
    """Build a MIMO model (no DDP) from a ScheduleConfig + optional memory config."""
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    enc_grid = _create_grid(tp=sc.enc_tp, dp=sc.enc_dp)
    llm_grid = _create_grid(tp=sc.llm_tp, dp=sc.llm_dp, pp=sc.llm_pp)
    _create_embedding_groups([enc_grid, llm_grid])

    enc_pg = _get_pg_collection(enc_grid, is_language_model=False)
    llm_pg = _get_pg_collection(llm_grid, is_language_model=True)

    tp_rank = dist.get_rank(llm_pg.tp) if llm_pg.tp else 0
    model_parallel_cuda_manual_seed(
        42, tp_rank=tp_rank, ep_rank=0, etp_rank=0, force_reset_rng=True
    )
    torch.manual_seed(42)

    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_enc = enc_pg.tp.size() if enc_pg.tp else 1
    enc_cfg = TransformerConfig(
        num_layers=sc.enc_layers,
        hidden_size=sc.hidden_size,
        num_attention_heads=sc.num_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_enc,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    proj_cfg = TransformerConfig(num_layers=1, hidden_size=sc.hidden_size, num_attention_heads=1)
    proj_cfg.ffn_hidden_size = sc.hidden_size
    proj_cfg.bias_activation_fusion = True
    proj_cfg.add_bias_linear = True
    proj_cfg.activation_func = torch.nn.functional.gelu

    encoder_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {
                "clip_encoder": ModuleSpec(
                    module=TransformerBlock,
                    params={
                        "config": enc_cfg,
                        "spec": get_gpt_layer_with_transformer_engine_spec(),
                        "pg_collection": enc_pg,
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
                        "input_size": sc.hidden_size,
                        "tp_group": enc_pg.tp,
                    },
                )
            ],
        },
    )

    pp_rank = dist.get_rank(llm_pg.pp)
    pp_size = dist.get_world_size(llm_pg.pp)
    tp_llm = llm_pg.tp.size() if llm_pg.tp else 1
    lm_cfg = TransformerConfig(
        num_layers=sc.llm_layers,
        hidden_size=sc.hidden_size,
        num_attention_heads=sc.num_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_llm,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
    )
    lm_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_cfg,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": sc.vocab_size,
            "max_sequence_length": sc.seq_length,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": llm_pg,
        },
    )

    # Adapt recompute_num_layers to actual encoder layer count
    if memory_config:
        enc_mcfg = memory_config.get(ENCODER_NAME)
        if enc_mcfg and enc_mcfg.recompute_num_layers is not None:
            memory_config = dict(memory_config)  # shallow copy
            memory_config[ENCODER_NAME] = ModuleMemoryConfig(
                recompute_granularity=enc_mcfg.recompute_granularity,
                recompute_method=enc_mcfg.recompute_method,
                recompute_num_layers=sc.enc_layers,
                recompute_modules=enc_mcfg.recompute_modules,
                offload_modules=enc_mcfg.offload_modules,
                recompute_projection=enc_mcfg.recompute_projection,
                offload_projection=enc_mcfg.offload_projection,
                offload_encoder_output=enc_mcfg.offload_encoder_output,
                recompute_projection_output=enc_mcfg.recompute_projection_output,
                offload_projection_output=enc_mcfg.offload_projection_output,
            )

    mimo_config = MimoModelConfig(
        language_model_spec=lm_spec,
        modality_submodules_spec={ENCODER_NAME: encoder_spec},
        special_token_ids={ENCODER_NAME: sc.image_token_id},
        module_to_grid_map={ENCODER_NAME: enc_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
        memory_config=memory_config,
    )

    model = MimoModel(mimo_config)
    model.to(torch.device("cuda")).to(torch.bfloat16)
    model.model_type = ModelType.encoder_or_decoder
    model.train()

    # Schedule callbacks (no DDP — correctness tests use param.grad directly)
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

    return model, enc_grid, llm_grid, enc_pg, llm_pg


def _make_batch(sc: ScheduleConfig, seed):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, sc.vocab_size, (sc.micro_batch_size, sc.seq_length), device='cuda')
    input_ids[:, : sc.image_seq_length] = sc.image_token_id
    return {
        'input_ids': input_ids,
        'labels': torch.randint(
            0, sc.vocab_size, (sc.micro_batch_size, sc.seq_length), device='cuda'
        ),
        'loss_mask': torch.ones(sc.micro_batch_size, sc.seq_length, device='cuda'),
        'position_ids': torch.arange(sc.seq_length, device='cuda')
        .unsqueeze(0)
        .expand(sc.micro_batch_size, -1),
        'attention_mask': None,
        'modality_inputs': {
            ENCODER_NAME: {
                'clip_encoder': {
                    'hidden_states': torch.randn(
                        sc.image_seq_length,
                        sc.micro_batch_size,
                        sc.hidden_size,
                        device='cuda',
                        dtype=torch.bfloat16,
                    ),
                    'attention_mask': None,
                }
            }
        },
    }


class _BatchIterator:
    def __init__(self, sc: ScheduleConfig, seed=123):
        self.sc, self.seed, self.i = sc, seed, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        return _make_batch(self.sc, self.seed + self.i)


def _forward_step(data_iterator, model, *args, **kwargs):
    batch = next(data_iterator)
    output_tensor, loss_mask = model(**batch)

    def loss_func(loss_mask, output_tensor):
        if output_tensor is None:
            return torch.tensor(0.0, device='cuda', requires_grad=True), {}
        return output_tensor.float().sum(), {'loss': output_tensor.float().sum().detach().item()}

    return output_tensor, partial(loss_func, loss_mask)


def _collect_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        grad = getattr(param, 'main_grad', None)
        if grad is None:
            grad = param.grad
        if grad is not None:
            grads[name] = grad.float().clone()
    return grads


def _reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================================
# Oracle: single function that runs baseline vs optimized and validates
# ============================================================================


def _has_offload(memory_config):
    """Check if a memory_config dict contains any offload flags."""
    if not memory_config:
        return False
    for mcfg in memory_config.values():
        if (
            mcfg.offload_projection
            or mcfg.offload_encoder_output
            or mcfg.offload_projection_output
            or mcfg.offload_modules
        ):
            return True
    return False


# Tolerance for offload memory expectation (matches test_fine_grained_activation_offloading.py)
_OFFLOAD_REL_TOL = 0.30  # 30% relative
_OFFLOAD_ABS_TOL = 20  # MiB absolute


def run_and_validate(
    sc: ScheduleConfig, memory_config: Dict[str, ModuleMemoryConfig], check_memory: bool = False
):
    """Oracle test function: build baseline + optimized, compare grads, optionally check memory.

    Works for both PP=1 (forward_backward_no_pipelining) and PP>1
    (colocated_forward_backward_with_pp) based on sc.is_pp.

    For offload configs, validates peak memory savings match
    PipelineOffloadManager.offload_summary_bytes within tolerance
    (same approach as test_fine_grained_activation_offloading.py).
    """

    def _run_one_iteration(model, sc, enc_grid, llm_grid, llm_pg, seed):
        """Run one fwd/bwd iteration, return grads."""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        if sc.is_pp:
            pp_group = llm_grid.get_pg("pp")
            p2p = P2PCommunicator(pp_group=pp_group, config=model.config)
            data_iter = _BatchIterator(sc, seed=seed)
            colocated_forward_backward_with_pp(
                mimo_model=model,
                data_iterator=data_iter,
                num_microbatches=sc.num_microbatches,
                encoder_grid=enc_grid,
                llm_grid=llm_grid,
                encoder_name=ENCODER_NAME,
                seq_length=sc.seq_length,
                micro_batch_size=sc.micro_batch_size,
                p2p_communicator=p2p,
                pg_collection=llm_pg,
            )
        else:
            data_iter = _BatchIterator(sc, seed=seed)
            schedule.forward_backward_no_pipelining(
                forward_step_func=_forward_step,
                data_iterator=data_iter,
                model=[model],
                num_microbatches=sc.num_microbatches,
                seq_length=sc.seq_length,
                micro_batch_size=sc.micro_batch_size,
                forward_only=False,
                pg_collection=llm_pg,
            )

        return _collect_grads(model)

    def _build_and_measure(sc, memory_config, seed):
        """Build model, warmup, measure steady-state. Returns (grads, peak, offload_bytes)."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_iface,
            PipelineOffloadManager,
        )

        is_offload = _has_offload(memory_config)
        if is_offload:
            off_iface.reset_instance()

        model, enc_grid, llm_grid, _, llm_pg = _build_mimo_model(sc, memory_config)

        # Warmup iteration
        _run_one_iteration(model, sc, enc_grid, llm_grid, llm_pg, seed=seed + 1000)

        # For offload: read expected offload bytes after warmup triggers post_warmup_callback
        expected_offload_bytes = 0
        if is_offload:
            mgr = PipelineOffloadManager.get_instance()
            if mgr is not None and hasattr(mgr, 'offload_summary_bytes'):
                expected_offload_bytes = int(sum(mgr.offload_summary_bytes.values()))

        _reset_cuda()

        # Steady-state measurement
        torch.cuda.reset_peak_memory_stats()
        grads = _run_one_iteration(model, sc, enc_grid, llm_grid, llm_pg, seed=seed)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        del model
        _reset_cuda()
        _destroy_all_grids()
        if is_offload:
            off_iface.reset_instance()

        return grads, peak, expected_offload_bytes

    # Run optimized first (avoids CUDA allocator caching bias)
    grads_opt, peak_opt, expected_offload_bytes = _build_and_measure(sc, memory_config, seed=42)
    grads_base, peak_base, _ = _build_and_measure(sc, None, seed=42)

    # --- Gradient correctness ---
    assert len(grads_base) > 0, "Baseline produced no gradients"
    assert len(grads_opt) > 0, "Optimized model produced no gradients"

    matched = 0
    for name in grads_base:
        if name not in grads_opt:
            continue
        g_base, g_opt = grads_base[name], grads_opt[name]
        assert g_base.shape == g_opt.shape, f"Shape mismatch for {name}"
        assert torch.isfinite(g_opt).all(), f"Non-finite gradient for {name}"
        assert torch.allclose(g_base, g_opt, atol=1e-2, rtol=1e-2), (
            f"Gradient mismatch for {name}: "
            f"max_diff={torch.max(torch.abs(g_base - g_opt)).item():.6f}"
        )
        matched += 1
    assert matched > 0, "No matching parameter names between baseline and optimized"

    # --- Memory check ---
    if check_memory:
        saved_mib = (peak_base - peak_opt) / (1024**2)
        expected_offload_mib = expected_offload_bytes / (1024**2)

        if dist.get_rank() == 0:
            print(
                f"\n  Memory: baseline={peak_base / (1024**2):.1f}MB, "
                f"optimized={peak_opt / (1024**2):.1f}MB, saved={saved_mib:.1f}MB"
                + (
                    f", expected_offload={expected_offload_mib:.1f}MB"
                    if expected_offload_mib > 0
                    else ""
                )
            )

        # Positive savings required
        assert saved_mib > 0, (
            f"Expected memory reduction but got: "
            f"baseline={peak_base / (1024**2):.1f}MB, optimized={peak_opt / (1024**2):.1f}MB"
        )

        # For offload: match expected savings within tolerance
        # (same pattern as test_fine_grained_activation_offloading.py)
        if expected_offload_mib >= 2.0:
            rel_err = abs(saved_mib - expected_offload_mib) / max(expected_offload_mib, 1e-6)
            abs_err = abs(saved_mib - expected_offload_mib)
            assert rel_err <= _OFFLOAD_REL_TOL or abs_err <= _OFFLOAD_ABS_TOL, (
                f"Offload memory saving mismatch: saved={saved_mib:.2f}MiB "
                f"expected~={expected_offload_mib:.2f}MiB "
                f"(rel_err={rel_err:.2f}, abs_err={abs_err:.2f}MiB)"
            )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def init_dist():
    Utils.initialize_distributed()
    yield
    _destroy_all_grids()


# ============================================================================
# Parametrized correctness tests — PP=1
# ============================================================================


@pytest.mark.parametrize(
    "config_name",
    [
        "enc_recompute_full",
        "enc_recompute_selective",
        "llm_recompute_selective",
        "recompute_projection",
        "recompute_projection_output",
        "offload_projection",
        "offload_encoder_output",
        "offload_projection_output",
        "enc_offload_attn_norm",
        "llm_offload_attn_norm",
        "mixed_recompute",
        "mixed_recompute_and_offload",
    ],
)
def test_pp1_correctness(config_name):
    """PP=1 colocated: verify optimized grads match baseline."""
    run_and_validate(PP1_COLOCATED, MEMORY_CONFIGS[config_name])


# ============================================================================
# Parametrized correctness tests — PP>1 colocated
# ============================================================================


@pytest.mark.parametrize(
    "config_name",
    [
        "enc_recompute_full",
        "recompute_projection",
        "offload_projection",
        "mixed_recompute_and_offload",
    ],
)
def test_pp2_colocated_correctness(config_name):
    """PP>1 colocated (3-phase schedule): verify valid grads."""
    run_and_validate(PP2_COLOCATED, MEMORY_CONFIGS[config_name])


# ============================================================================
# Memory reduction test (larger model)
# ============================================================================


def test_encoder_full_recompute_memory():
    """Verify encoder full recompute saves measurable GPU memory."""
    mc = {
        ENCODER_NAME: ModuleMemoryConfig(
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=PP1_MEMORY.enc_layers,
        )
    }
    run_and_validate(PP1_MEMORY, mc, check_memory=True)


def test_offload_projection_memory():
    """Verify offload_projection saves memory matching offload_summary_bytes."""
    mc = {ENCODER_NAME: ModuleMemoryConfig(offload_projection=True)}
    run_and_validate(PP1_MEMORY, mc, check_memory=True)


def test_encoder_offload_attn_norm_memory():
    """Verify encoder offload_modules=['attn_norm'] saves memory matching offload_summary_bytes."""
    mc = {ENCODER_NAME: ModuleMemoryConfig(offload_modules=['attn_norm'])}
    run_and_validate(PP1_MEMORY, mc, check_memory=True)
