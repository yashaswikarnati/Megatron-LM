# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MIMO memory optimization: recompute and offload.

4 tests covering the support matrix:
  1. PP=1  recompute (encoder full + combined_embeddings) — correctness + memory
  2. PP=1  offload (combined_embeddings) — correctness + memory
  3. PP>1  recompute (encoder full + combined_embeddings) — correctness
  4. PP=1  mixed (encoder + LLM + combined_embeddings) — correctness + memory

Each test verifies bitwise identical gradients (torch.equal) with unfused
attention for determinism, and asserts positive memory savings.

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
from typing import Dict

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
from megatron.core.transformer.enums import AttnBackend, ModelType
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


@dataclass(frozen=True)
class ScheduleConfig:
    enc_tp: int
    enc_dp: int
    llm_tp: int
    llm_dp: int
    llm_pp: int = 1
    hidden_size: int = 128
    num_heads: int = 4
    enc_layers: int = 2
    llm_layers: int = 2
    vocab_size: int = 1024
    seq_length: int = 64
    image_seq_length: int = 16
    micro_batch_size: int = 2
    num_microbatches: int = 2
    image_token_id: int = 32000

    @property
    def is_pp(self):
        return self.llm_pp > 1


PP1 = ScheduleConfig(enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=4)
PP2 = ScheduleConfig(
    enc_tp=2, enc_dp=4, llm_tp=2, llm_dp=2, llm_pp=2, num_microbatches=2, image_token_id=50257
)

# ============================================================================
# Infrastructure
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
    for dims in (
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
        grid.create_pg(dims)
    _active_grids.append(grid)
    return grid


def _destroy_all_grids():
    for g in _active_grids:
        g.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def _create_embedding_groups(grids):
    for grid in grids:
        pp_group = grid.get_pg("pp")
        if not pp_group:
            continue
        pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
        key = tuple(pp_ranks)
        if key not in _embedding_pg_cache:
            first, last = [pp_ranks[0]], [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                last.append(pp_ranks[-1])
            _embedding_pg_cache[key] = (dist.new_group(ranks=first), dist.new_group(ranks=last))


def _get_pg(grid, is_lm=False):
    pg = ProcessGroupCollection()
    pg.tp, pg.cp, pg.pp = grid.get_pg("tp"), grid.get_pg("cp"), grid.get_pg("pp")
    pg.ep, pg.dp = grid.get_pg("ep"), grid.get_pg("dp")
    pg.dp_cp, pg.expt_dp = grid.get_pg(["dp", "cp"]), grid.get_pg("expt_dp")
    if pg.pp:
        key = tuple(sorted(dist.get_process_group_ranks(pg.pp)))
        if key in _embedding_pg_cache:
            pos_pg, embd_pg = _embedding_pg_cache[key]
            pg.pos_embd = pos_pg if is_pp_first_stage(pg.pp) else None
            pg.embd = (
                embd_pg if is_lm and (is_pp_last_stage(pg.pp) or is_pp_first_stage(pg.pp)) else None
            )
    return pg


def _build_model(sc, memory_config=None):
    import dataclasses as _dc
    from megatron.core.transformer.transformer_block import TransformerBlock

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    enc_grid = _create_grid(tp=sc.enc_tp, dp=sc.enc_dp)
    llm_grid = _create_grid(tp=sc.llm_tp, dp=sc.llm_dp, pp=sc.llm_pp)
    _create_embedding_groups([enc_grid, llm_grid])

    enc_pg = _get_pg(enc_grid)
    llm_pg = _get_pg(llm_grid, is_lm=True)

    tp_rank = dist.get_rank(llm_pg.tp) if llm_pg.tp else 0
    model_parallel_cuda_manual_seed(
        42, tp_rank=tp_rank, ep_rank=0, etp_rank=0, force_reset_rng=True
    )
    torch.manual_seed(42)

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
        attention_backend=AttnBackend.unfused,
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
        attention_backend=AttnBackend.unfused,
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
            memory_config = dict(memory_config)
            memory_config[ENCODER_NAME] = _dc.replace(enc_mcfg, recompute_num_layers=sc.enc_layers)

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

    @contextmanager
    def no_sync():
        yield

    model.config.no_sync_func = no_sync
    model.config.finalize_model_grads_func = lambda *a, **kw: None
    model.config.grad_scale_func = lambda l: (
        torch.tensor(l, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(l, (int, float))
        else l
    )

    return model, enc_grid, llm_grid, llm_pg


def _make_batch(sc, seed):
    torch.manual_seed(seed)
    ids = torch.randint(0, sc.vocab_size, (sc.micro_batch_size, sc.seq_length), device='cuda')
    ids[:, : sc.image_seq_length] = sc.image_token_id
    return {
        'input_ids': ids,
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


class _Iter:
    def __init__(self, sc, seed=42):
        self.sc, self.seed, self.i = sc, seed, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        return _make_batch(self.sc, self.seed + self.i)


def _fwd_step(data_iterator, model, *a, **kw):
    batch = next(data_iterator)
    out, lm = model(**batch)

    def loss_fn(lm, out):
        if out is None:
            return torch.tensor(0.0, device='cuda', requires_grad=True), {}
        return out.float().sum(), {}

    return out, partial(loss_fn, lm)


def _grads(model):
    g = {}
    for n, p in model.named_parameters():
        grad = getattr(p, 'main_grad', None)
        if grad is None:
            grad = p.grad
        if grad is not None:
            g[n] = grad.float().clone()
    return g


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================================
# Oracle
# ============================================================================


def run_and_validate(sc, memory_config, check_memory=True):
    """Build baseline + optimized, verify torch.equal grads, check memory."""

    def _run(sc, enc_grid, llm_grid, llm_pg, seed):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if sc.is_pp:
            pp_group = llm_grid.get_pg("pp")
            p2p = P2PCommunicator(pp_group=pp_group, config=model.config)
            colocated_forward_backward_with_pp(
                mimo_model=model,
                data_iterator=_Iter(sc, seed),
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
            schedule.forward_backward_no_pipelining(
                forward_step_func=_fwd_step,
                data_iterator=_Iter(sc, seed),
                model=[model],
                num_microbatches=sc.num_microbatches,
                seq_length=sc.seq_length,
                micro_batch_size=sc.micro_batch_size,
                forward_only=False,
                pg_collection=llm_pg,
            )
        return _grads(model)

    # Optimized first (avoids CUDA allocator caching bias)
    model, enc_grid, llm_grid, llm_pg = _build_model(sc, memory_config)
    _run(sc, enc_grid, llm_grid, llm_pg, seed=1000)  # warmup
    _reset()
    torch.cuda.reset_peak_memory_stats()
    grads_opt = _run(sc, enc_grid, llm_grid, llm_pg, seed=42)
    torch.cuda.synchronize()
    peak_opt = torch.cuda.max_memory_allocated()
    del model
    _reset()
    _destroy_all_grids()

    # Baseline
    model, enc_grid, llm_grid, llm_pg = _build_model(sc, None)
    _run(sc, enc_grid, llm_grid, llm_pg, seed=1000)  # warmup
    _reset()
    torch.cuda.reset_peak_memory_stats()
    grads_base = _run(sc, enc_grid, llm_grid, llm_pg, seed=42)
    torch.cuda.synchronize()
    peak_base = torch.cuda.max_memory_allocated()
    del model
    _reset()
    _destroy_all_grids()

    # Gradient correctness — bitwise identical
    assert len(grads_base) > 0 and len(grads_opt) > 0
    matched = 0
    for name in grads_base:
        if name not in grads_opt:
            continue
        assert torch.equal(grads_base[name], grads_opt[name]), (
            f"Gradient mismatch for {name}: "
            f"max_diff={torch.max(torch.abs(grads_base[name] - grads_opt[name])).item():.2e}"
        )
        matched += 1
    assert matched > 0

    # Memory check
    if check_memory:
        saved_mb = (peak_base - peak_opt) / (1024**2)
        if dist.get_rank() == 0:
            print(
                f"  {matched} params matched | saved={saved_mb:.1f}MB "
                f"(base={peak_base / (1024**2):.0f}MB opt={peak_opt / (1024**2):.0f}MB)"
            )
        assert saved_mb > 0, f"No memory saved: base={peak_base}, opt={peak_opt}"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module", autouse=True)
def init_dist():
    Utils.initialize_distributed()
    yield
    _destroy_all_grids()


# ============================================================================
# Tests (4 total)
# ============================================================================


def test_pp1_recompute():
    """PP=1: encoder full recompute + recompute_combined_embeddings."""
    run_and_validate(
        PP1,
        {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=2,
                recompute_combined_embeddings=True,
            )
        },
    )


def test_pp1_offload():
    """PP=1: offload_combined_embeddings (frees projection_output + combined)."""
    run_and_validate(PP1, {ENCODER_NAME: ModuleMemoryConfig(offload_combined_embeddings=True)})


def test_pp2_recompute():
    """PP>1 colocated: encoder recompute + recompute_combined_embeddings."""
    run_and_validate(
        PP2,
        {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=2,
                recompute_combined_embeddings=True,
            )
        },
    )


def test_pp1_mixed():
    """PP=1: encoder recompute + LLM recompute + recompute_combined_embeddings."""
    run_and_validate(
        PP1,
        {
            ENCODER_NAME: ModuleMemoryConfig(
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=2,
                recompute_combined_embeddings=True,
            ),
            MIMO_LANGUAGE_MODULE_KEY: ModuleMemoryConfig(
                recompute_granularity='selective', recompute_modules=['core_attn']
            ),
        },
    )
