# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration test: small MoE GPT model driven by ``forward_backward_no_pipelining``
with process groups built from a HyperCommGrid (alt-factorization for EP overlap),
and NO global ``parallel_state`` initialization.

Verifies the NMFW-464 wiring: HyperCommGrid → ProcessGroupCollection → MoE layers +
DDP + schedule, all without touching ``parallel_state``.
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state, pipeline_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


def _build_grid_and_pg_collection():
    """Build an 8-rank HyperCommGrid with expert overlap and a pg_collection."""
    grid = HyperCommGrid(
        shape=[2, 1, 4, 1],  # tp=2 cp=1 dp=4 pp=1
        dim_names=["tp", "cp", "dp", "pp"],
        backend="nccl",
        alt_factorizations={
            "expert": {
                # ep=4 carved out of the dp=4 axis; etp=1, edp=1.
                # Product: 1 * 4 * 1 == 2 * 1 * 4 == 8.
                "shape": [1, 4, 2],
                "dim_names": ["etp", "ep", "edp"],
                "replaces": ["tp", "cp", "dp"],
            }
        },
    )
    pg = ProcessGroupCollection.from_hyper_comm_grid(grid)
    # forward_backward_no_pipelining reads pg.embd / pg.pos_embd if present;
    # with pp=1 a singleton group per rank is fine. Reuse pp group.
    pg.embd = pg.pp
    pg.pos_embd = pg.pp
    return grid, pg


def _mock_data_iterator(seq_length: int, micro_batch_size: int, vocab_size: int):
    while True:
        input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device="cuda")
        labels = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device="cuda")
        loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
        position_ids = (
            torch.arange(seq_length, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)
        )
        attention_mask = None
        yield {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


def _loss_func(loss_mask, output_tensor):
    """Sum-of-logits surrogate loss, matching test_mimo_1f1b_schedule.

    We avoid VocabParallelCrossEntropy because it calls ``parallel_state``; this test verifies
    that the schedule path itself works without ``parallel_state`` initialization.
    """
    loss = output_tensor.float().sum()
    return loss, {"lm_loss": loss.detach()}


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    # Pass labels=None so the model returns logits and we can apply a no-parallel-state loss.
    output_tensor = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
    )
    return output_tensor, partial(_loss_func, batch["loss_mask"])


class TestMoEWithHCGProcessGroups:
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    def test_moe_gpt_forward_backward_no_pipelining_8gpu(self):
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")

        torch.manual_seed(12345)
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

        grid, pg = _build_grid_and_pg_collection()

        # Tightening assertion (per code review): prove the alt-factorization path was taken
        # by showing ep group ranks differ from dp group ranks. With primary tp=2 cp=1 dp=4
        # and alt etp=1 ep=4 edp=2, dp groups are [[0,2,4,6],[1,3,5,7]] but ep groups are
        # [[0,1,2,3],[4,5,6,7]] — distinct. A silent orthogonal fallback would alias them.
        ep_ranks = sorted(dist.get_process_group_ranks(pg.ep))
        dp_ranks = sorted(dist.get_process_group_ranks(pg.dp))
        assert ep_ranks != dp_ranks, (
            f"ep ranks {ep_ranks} match dp ranks {dp_ranks}; alt-factorization is likely "
            f"silently aliasing dp instead of carving its own slab"
        )

        # Initialize model-parallel CUDA RNG tracker without parallel_state, by passing
        # explicit ranks derived from the HyperCommGrid groups.
        tp_rank = dist.get_rank(group=pg.tp)
        ep_rank = dist.get_rank(group=pg.ep)
        etp_rank = dist.get_rank(group=pg.expt_tp)
        model_parallel_cuda_manual_seed(
            seed=12345, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank
        )

        # Sequence-parallel allgather buffer lives in parallel_state but is a side-effect-free
        # global cache; we initialize it without touching the model-parallel groups.
        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        num_experts = 4
        hidden = 64
        seq_length = 32
        micro_batch_size = 2
        vocab_size = 128

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden,
            num_attention_heads=4,
            ffn_hidden_size=4 * hidden,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=2 * hidden,
            add_bias_linear=False,
            use_cpu_initialization=True,
            tensor_model_parallel_size=2,
            context_parallel_size=1,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            calculate_per_token_loss=False,
            bf16=False,
            params_dtype=torch.float32,
            attention_backend=AttnBackend.unfused,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts, moe_grouped_gemm=False
        )

        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=seq_length,
            pre_process=True,
            post_process=True,
            pg_collection=pg,
        ).cuda()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
            bucket_size=None,
            average_in_collective=False,
        )
        ddp_model = DistributedDataParallel(
            config=config, ddp_config=ddp_config, module=model, pg_collection=pg
        )

        data_iter = _mock_data_iterator(seq_length, micro_batch_size, vocab_size)

        losses = pipeline_parallel.schedules.forward_backward_no_pipelining(
            forward_step_func=_forward_step,
            data_iterator=data_iter,
            model=ddp_model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            pg_collection=pg,
        )

        # Loss should be finite scalar.
        assert isinstance(losses, list) and len(losses) == 1
        loss_dict = losses[0]
        assert "lm_loss" in loss_dict
        loss_val = loss_dict["lm_loss"]
        assert torch.isfinite(loss_val).item(), f"loss not finite: {loss_val}"

        # At least one parameter must have a non-zero gradient.
        any_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
                break
            if hasattr(p, "main_grad") and p.main_grad is not None and p.main_grad.abs().sum() > 0:
                any_grad = True
                break
        assert any_grad, "no parameter received a non-zero gradient"

        # Cleanup: keep things tidy for the next test run inside the same process.
        ddp_model.zero_grad_buffer()
        del ddp_model, model
        torch.cuda.empty_cache()
