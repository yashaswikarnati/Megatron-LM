# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for colocated MIMO training with LLM PP>1.

Uses two-phase execution: encoder pre-compute + 1F1B LLM pipeline.

Run individually (8 GPUs):
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        -m pytest tests/unit_tests/models/test_mimo_colocated_pp.py -v
"""

import logging
from contextlib import ExitStack, contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.colocated_schedule import colocated_forward_backward_with_pp
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
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

_active_grids: list = []
_embedding_pg_cache: dict = {}


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    for dim in ["tp", "cp", "pp", "dp", "ep", "expt_dp"]:
        grid.create_pg([dim])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    grid.create_pg(["tp", "cp", "ep", "pp", "dp"])
    _active_grids.append(grid)
    return grid


def destroy_all_grids():
    for g in _active_grids:
        g.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def create_all_embedding_groups(grids):
    for grid in grids:
        pp_group = grid.get_pg("pp")
        if not pp_group:
            continue
        pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
        key = tuple(pp_ranks)
        if key not in _embedding_pg_cache:
            pos = [pp_ranks[0]]
            embd = [pp_ranks[0]]
            if pp_ranks[-1] != pp_ranks[0]:
                embd.append(pp_ranks[-1])
            _embedding_pg_cache[key] = (dist.new_group(ranks=pos), dist.new_group(ranks=embd))


def get_pg_collection(grid, is_language_model=False):
    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.ep = grid.get_pg("ep")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])
    pg.expt_dp = grid.get_pg("expt_dp")
    pp_ranks = sorted(dist.get_process_group_ranks(pg.pp))
    key = tuple(pp_ranks)
    if key in _embedding_pg_cache:
        pos_pg, embd_pg = _embedding_pg_cache[key]
        pg.pos_embd = pos_pg if is_pp_first_stage(pg.pp) else None
        pg.embd = (
            embd_pg
            if is_language_model and (is_pp_last_stage(pg.pp) or is_pp_first_stage(pg.pp))
            else None
        )
    return pg


def get_language_model_spec(
    num_layers, hidden_size, num_attention_heads, vocab_size, seq_len, pg_collection
):
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    tp_size = pg_collection.tp.size() if pg_collection.tp else 1
    lm_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
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
            "max_sequence_length": seq_len,
            "pre_process": (pp_rank == 0),
            "post_process": (pp_rank == pp_size - 1),
            "pg_collection": pg_collection,
        },
    )


def get_vision_submodules_spec(
    num_layers, hidden_size, num_attention_heads, language_hidden_size, pg_collection
):
    from megatron.core.transformer.transformer_block import TransformerBlock

    tp_size = pg_collection.tp.size() if pg_collection.tp else 1
    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type='alltoall',
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    proj_cfg = TransformerConfig(
        num_layers=1, hidden_size=language_hidden_size, num_attention_heads=1
    )
    proj_cfg.ffn_hidden_size = language_hidden_size
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
                        "input_size": vision_config.hidden_size,
                        "tp_group": pg_collection.tp,
                    },
                )
            ],
        },
    )


class DataIterator:
    def __init__(
        self,
        hidden_size,
        seq_length,
        micro_batch_size,
        vocab_size,
        encoder_name,
        image_token_id=50257,
        image_seq_length=None,
    ):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size
        self.encoder_name = encoder_name
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length or (seq_length // 2)

    def __iter__(self):
        return self

    def __next__(self):
        encoder_hidden_states = torch.randn(
            self.image_seq_length,
            self.micro_batch_size,
            self.hidden_size,
            device='cuda',
            dtype=torch.bfloat16,
        )
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            self.image_token_id,
            dtype=torch.long,
            device='cuda',
        )
        text_tokens = torch.randint(
            1,
            self.vocab_size,
            (self.micro_batch_size, self.seq_length - self.image_seq_length),
            device='cuda',
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)
        labels = input_ids.clone()
        labels[input_ids == self.image_token_id] = -100
        loss_mask = (input_ids != self.image_token_id).float()
        position_ids = (
            torch.arange(self.seq_length, device='cuda')
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone()
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "modality_inputs": {
                self.encoder_name: {
                    "clip_encoder": {'hidden_states': encoder_hidden_states, 'attention_mask': None}
                }
            },
        }


def run_colocated_pp_test(
    encoder_tp,
    encoder_dp,
    llm_tp,
    llm_pp,
    llm_dp,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    micro_batch_size=2,
    num_microbatches=4,
):
    """Run colocated MIMO with encoder PP=1 + LLM PP>1."""
    import os

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_name = "images"

    encoder_grid = create_hypercomm_grid(offset=0, tp=encoder_tp, cp=1, pp=1, dp=encoder_dp)
    llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
    create_all_embedding_groups([encoder_grid, llm_grid])
    torch.manual_seed(12345)

    vision_pg = get_pg_collection(encoder_grid, is_language_model=False)
    language_pg = get_pg_collection(llm_grid, is_language_model=True)

    language_model_spec = get_language_model_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        vocab_size=vocab_size,
        seq_len=seq_length,
        pg_collection=language_pg,
    )
    vision_submodule_spec = get_vision_submodules_spec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        language_hidden_size=hidden_size,
        pg_collection=vision_pg,
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={encoder_name: vision_submodule_spec},
        special_token_ids={encoder_name: 50257},
        module_to_grid_map={encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid},
    )

    mimo_model = MimoModel(mimo_config)
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    mimo_model.model_type = ModelType.encoder_or_decoder

    # Wrap with DDP (per-module process groups)
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False, bucket_size=10000, use_distributed_optimizer=True
    )
    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=language_pg,
        )
    if encoder_name in mimo_model.modality_submodules:
        submodule = mimo_model.modality_submodules[encoder_name]
        if submodule is not None:
            mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                config=submodule.encoders['clip_encoder'].config,
                ddp_config=ddp_config,
                module=submodule,
                pg_collection=vision_pg,
            )

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for sub in mimo_model.modality_submodules.values():
                if sub is not None:
                    stack.enter_context(sub.no_sync())
            yield

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model], num_tokens=None, pg_collection=language_pg
            )
        for sub in mimo_model.modality_submodules.values():
            if sub is not None:
                finalize_model_grads([sub], num_tokens=None, pg_collection=vision_pg)

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    data_iterator = DataIterator(
        hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name
    )
    lm_pp_group = llm_grid.get_pg("pp")

    rank = dist.get_rank()
    num_iterations = 2
    all_losses = []
    optimizer.zero_grad()

    for iteration in range(num_iterations):
        losses = colocated_forward_backward_with_pp(
            mimo_model=mimo_model,
            data_iterator=data_iterator,
            num_microbatches=num_microbatches,
            encoder_grid=encoder_grid,
            llm_grid=llm_grid,
            encoder_name=encoder_name,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            p2p_communicator=P2PCommunicator(pp_group=lm_pp_group, config=mimo_model.config),
            pg_collection=language_pg,
        )

        success, grad_norm, _ = optimizer.step()
        assert success, f"Rank {rank}: Optimizer step failed at iteration {iteration}"
        optimizer.zero_grad()

        all_losses.extend(losses or [])
        logger.info(f"Rank {rank}: iteration {iteration} done, losses={len(losses or [])}")

    # Verify on last PP stage
    if is_pp_last_stage(lm_pp_group):
        assert len(all_losses) > 0, f"Rank {rank}: No losses on last stage"
        for i, loss_dict in enumerate(all_losses):
            loss_val = loss_dict.get('loss_reduced', 0)
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            assert loss_val == loss_val, f"Rank {rank}: NaN loss at mb {i}"
            assert abs(loss_val) != float('inf'), f"Rank {rank}: Inf loss at mb {i}"

    return all_losses


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimoColocatedPP:
    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_fan_in_enc_tp2_dp4_llm_tp2_dp2_pp2(self):
        """Fan-in: encoder TP2/DP4 → LLM TP2/DP2/PP2."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_pp_test(
            encoder_tp=2, encoder_dp=4, llm_tp=2, llm_pp=2, llm_dp=2, num_microbatches=4
        )

    def test_equal_dp_enc_tp4_dp2_llm_tp2_dp2_pp2(self):
        """Equal DP: encoder TP4/DP2 → LLM TP2/DP2/PP2 (enc_dp == llm_dp)."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_pp_test(
            encoder_tp=4, encoder_dp=2, llm_tp=2, llm_pp=2, llm_dp=2, num_microbatches=4
        )

    def test_fan_in_with_grad_acc(self):
        """Fan-in with gradient accumulation (num_microbatches > pp_size)."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_pp_test(
            encoder_tp=2,
            encoder_dp=4,
            llm_tp=2,
            llm_pp=2,
            llm_dp=2,
            num_microbatches=6,  # > pp_size=2, tests grad accumulation
        )

    def test_fan_in_enc_tp1_dp8_llm_tp4_dp1_pp2(self):
        """Fan-in extreme: encoder TP1/DP8 → LLM TP4/DP1/PP2."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        # micro_batch_size must be >= fan-in scale (enc_dp/llm_dp = 8/1 = 8)
        # to avoid zero-sized slices in _slice_for_encoder_dp.
        run_colocated_pp_test(
            encoder_tp=1,
            encoder_dp=8,
            llm_tp=4,
            llm_pp=2,
            llm_dp=1,
            micro_batch_size=8,
            num_microbatches=4,
        )
