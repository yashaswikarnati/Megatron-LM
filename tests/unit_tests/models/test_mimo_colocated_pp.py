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
from megatron.core.models.mimo.colocated_schedule import colocated_forward_backward_with_pp
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
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
    """Run colocated MIMO with encoder PP=1 + LLM PP>1.

    Beyond "loss is finite", this helper verifies:
      * ``optimizer.step`` returns grad_norm > 0 — catches silently-zeroed
        encoder grads (e.g. broadcast never populating detached_full.grad
        on non-first PP stages).
      * Encoder params' data changed after the step — catches the case
        where grads flow but the update is a no-op (wrong PG, wrong
        device, clipping to zero).
      * LLM params' data changed on every PP stage — catches the case
        where the pipeline runs but a PP stage's grads never backprop.
    """
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

    # Snapshot initial params to verify the step actually moves them.
    # A silently-zeroed encoder grad (e.g. PP>1 grad broadcast missing) would
    # leave these unchanged despite grad_norm appearing nonzero.
    encoder_module = (
        mimo_model.modality_submodules[encoder_name].module
        if encoder_name in mimo_model.modality_submodules
        and mimo_model.modality_submodules[encoder_name] is not None
        else None
    )
    llm_module = mimo_model.language_model.module if mimo_model.language_model is not None else None
    initial_encoder_params = (
        {n: p.detach().clone() for n, p in encoder_module.named_parameters()}
        if encoder_module is not None
        else {}
    )
    initial_llm_params = (
        {n: p.detach().clone() for n, p in llm_module.named_parameters()}
        if llm_module is not None
        else {}
    )

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
        # grad_norm must be strictly positive: zero means every tracked param
        # had zero grad, which indicates the schedule never wired a usable
        # gradient into the param.grad buffers.
        assert grad_norm is not None and grad_norm > 0, (
            f"Rank {rank}: grad_norm={grad_norm} at iter {iteration} — encoder or "
            f"LLM grads were silently zeroed (did Phase 3 broadcast/backward run?)"
        )
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

    # Oracle: at least one param of each module's shard must have changed.
    # Under correct three-phase execution, every encoder rank accumulates a
    # nonzero DP=1 gradient (via Phase 3 backward from the broadcast grad),
    # and every LLM PP stage accumulates nonzero grads from the 1F1B pass.
    # A blanket "all shards unchanged" outcome means the optimizer step was
    # effectively a no-op for that module on this rank.
    if encoder_module is not None:
        changed = any(
            not torch.equal(p.detach(), initial_encoder_params[n])
            for n, p in encoder_module.named_parameters()
            if n in initial_encoder_params
        )
        assert changed, (
            f"Rank {rank}: no encoder params changed after {num_iterations} steps — "
            f"Phase 3 encoder backward likely did not populate grads on this rank"
        )
    if llm_module is not None:
        changed = any(
            not torch.equal(p.detach(), initial_llm_params[n])
            for n, p in llm_module.named_parameters()
            if n in initial_llm_params
        )
        assert changed, (
            f"Rank {rank}: no LLM params changed after {num_iterations} steps — "
            f"PP stage {dist.get_rank(lm_pp_group)} may have received no gradient"
        )

    return all_losses


# ---------------------------------------------------------------------------
# Weight-oracle helpers: dist (PP>1, heterogeneous) vs ref (PP=1, equal-DP).
# ---------------------------------------------------------------------------


def _build_pp_oracle_model(
    encoder_tp,
    encoder_dp,
    llm_tp,
    llm_pp,
    llm_dp,
    hidden_size,
    num_layers,
    vocab_size,
    seq_length,
    ddp_config,
    encoder_name="images",
):
    """Build a MimoModel + DDP wrap for the weight-oracle test. Returns the
    model plus its encoder_grid/llm_grid and pg_collections. Mirrors the
    setup in ``run_colocated_pp_test`` but accepts an explicit ``ddp_config``
    so both dist and ref can share ``gradient_reduce_div_factor=1``.
    """
    encoder_grid = create_hypercomm_grid(offset=0, tp=encoder_tp, cp=1, pp=1, dp=encoder_dp)
    llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=llm_pp, dp=llm_dp)
    create_all_embedding_groups([encoder_grid, llm_grid])

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

    return mimo_model, encoder_grid, llm_grid, language_pg, vision_pg


def _generate_shared_global_batches(
    num_batches,
    global_batch_size,
    seq_length,
    hidden_size,
    vocab_size,
    encoder_name,
    image_token_id=50257,
):
    """Generate global batches on rank 0 and broadcast so every rank sees
    identical data. Encoder input shape is [seq, batch, hidden] (sbh),
    matching ``DataIterator`` above.
    """
    rank = dist.get_rank()
    image_seq_length = seq_length // 2
    batches = []
    for _ in range(num_batches):
        if rank == 0:
            encoder_hidden_states = torch.randn(
                image_seq_length,
                global_batch_size,
                hidden_size,
                device='cuda',
                dtype=torch.bfloat16,
            )
            image_tokens = torch.full(
                (global_batch_size, image_seq_length),
                image_token_id,
                dtype=torch.long,
                device='cuda',
            )
            text_tokens = torch.randint(
                1,
                vocab_size,
                (global_batch_size, seq_length - image_seq_length),
                device='cuda',
            )
            input_ids = torch.cat([image_tokens, text_tokens], dim=1)
        else:
            encoder_hidden_states = torch.empty(
                image_seq_length,
                global_batch_size,
                hidden_size,
                device='cuda',
                dtype=torch.bfloat16,
            )
            input_ids = torch.empty(
                global_batch_size, seq_length, dtype=torch.long, device='cuda'
            )
        dist.broadcast(encoder_hidden_states, src=0)
        dist.broadcast(input_ids, src=0)

        labels = input_ids.clone()
        labels[input_ids == image_token_id] = -100
        loss_mask = (input_ids != image_token_id).float()
        position_ids = (
            torch.arange(seq_length, device='cuda')
            .unsqueeze(0)
            .expand(global_batch_size, -1)
            .clone()
        )
        batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "modality_inputs": {
                    encoder_name: {
                        "clip_encoder": {
                            'hidden_states': encoder_hidden_states,
                            'attention_mask': None,
                        }
                    }
                },
            }
        )
    return batches


def _slice_batch_along_dim0(batch, split, idx):
    """Return ``idx``-th of ``split`` equal slices along the batch dim."""
    b = batch['input_ids'].shape[0]
    size = b // split
    s, e = idx * size, (idx + 1) * size
    out = {k: batch[k][s:e].contiguous() for k in ['input_ids', 'labels', 'loss_mask', 'position_ids']}
    mod_new = {}
    for m, md in batch['modality_inputs'].items():
        mod_new[m] = {}
        for enc, ed in md.items():
            mod_new[m][enc] = {}
            for k, t in ed.items():
                if isinstance(t, torch.Tensor):
                    # modality hidden_states shape [seq, batch, hidden] — dim 1
                    mod_new[m][enc][k] = t[:, s:e, :].contiguous()
                else:
                    mod_new[m][enc][k] = t
    out['modality_inputs'] = mod_new
    return out


def _copy_encoder_params(ref_module, dist_module):
    """Copy encoder params ref → dist. Encoder layouts match by construction
    (same enc_tp and enc_dp in both models), so shards line up 1:1.
    """
    ref_params = dict(ref_module.named_parameters())
    with torch.no_grad():
        for name, dist_param in dist_module.named_parameters():
            assert name in ref_params, f"Encoder param '{name}' missing in ref"
            ref_param = ref_params[name]
            assert ref_param.shape == dist_param.shape, (
                f"Encoder param '{name}': ref.shape={tuple(ref_param.shape)} != "
                f"dist.shape={tuple(dist_param.shape)} — enc_tp/enc_dp must match "
                f"between ref and dist for shard-wise comparison."
            )
            dist_param.data.copy_(ref_param.data.to(dist_param.dtype))


def _copy_llm_params_pp_aware(
    ref_module, dist_module, pp_rank, pp_size, num_layers, dist_tp_group, ref_tp_group
):
    """Copy LLM params ref (PP=1) → dist (PP>=1) with layer-index remapping.

    In Megatron's ``TransformerBlock``, ``self.layers`` is a ``ModuleList``
    indexed 0..N-1 *locally* per PP stage. The global layer number is
    ``local_idx + pp_rank * layers_per_stage``. For a PP=1 reference, all
    N layers live at local indices 0..N-1 on each rank; for a PP>1 dist
    model, PP stage ``s``'s local layer ``i`` corresponds to ref's global
    layer ``s*layers_per_stage + i``.

    Non-layer params (embedding, final_layernorm, output_layer) are only
    present on stages with the relevant ``pre_process``/``post_process``
    flag, and their names match exactly between ref (which has them all)
    and whichever dist stage owns them.

    If dist_tp != ref_tp the helper falls back to the PR-10 pattern of
    gather-across-ref-tp + slice-for-dist-tp. Same-TP is the normal path
    (this helper is designed for tests where ``dist_llm_tp == ref_llm_tp``,
    so the gather path is a no-op fallback).
    """
    import re

    assert num_layers % pp_size == 0, (
        f"num_layers={num_layers} not divisible by pp_size={pp_size}; "
        f"oracle requires even PP split."
    )
    layers_per_stage = num_layers // pp_size
    layer_rx = re.compile(r'^(.*decoder\.layers\.)(\d+)(\..*)$')

    ref_params = dict(ref_module.named_parameters())
    ref_tp_size = dist.get_world_size(ref_tp_group)
    dist_tp_rank = dist.get_rank(dist_tp_group)
    dist_tp_size = dist.get_world_size(dist_tp_group)

    with torch.no_grad():
        for name, dist_param in dist_module.named_parameters():
            m = layer_rx.match(name)
            if m:
                prefix, local_idx_s, suffix = m.groups()
                global_idx = pp_rank * layers_per_stage + int(local_idx_s)
                ref_name = f"{prefix}{global_idx}{suffix}"
            else:
                ref_name = name

            assert ref_name in ref_params, (
                f"LLM param '{name}' on PP stage {pp_rank} maps to ref name "
                f"'{ref_name}' which does not exist in ref (ref has llm_pp=1)."
            )
            ref_param = ref_params[ref_name]
            partition_dim = getattr(dist_param, 'partition_dim', -1)

            if ref_param.shape == dist_param.shape:
                dist_param.data.copy_(ref_param.data.to(dist_param.dtype))
                continue

            assert partition_dim >= 0, (
                f"LLM param '{name}': shapes differ (ref={tuple(ref_param.shape)}, "
                f"dist={tuple(dist_param.shape)}) but partition_dim<0 — cannot reshape "
                f"a replicated param."
            )
            shards = [torch.empty_like(ref_param.data) for _ in range(ref_tp_size)]
            dist.all_gather(shards, ref_param.data.contiguous(), group=ref_tp_group)
            full = torch.cat(shards, dim=partition_dim)
            sliced = torch.tensor_split(full, dist_tp_size, dim=partition_dim)[dist_tp_rank]
            assert sliced.shape == dist_param.shape
            dist_param.data.copy_(sliced.to(dist_param.dtype))


def _sum_loss_func(loss_mask_unused, output_tensor):
    """Match the ``.sum()`` loss used by ``colocated_schedule._loss_func`` so
    the reference's forward_backward_no_pipelining path produces comparable
    gradient magnitudes.
    """
    if output_tensor is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}
    loss = output_tensor.float().sum()
    return loss, {'loss_reduced': loss.detach().item()}


def _assert_encoder_shards_match(ref_module, dist_module, rtol=1e-2, atol=1e-2):
    """Assert every dist encoder shard matches the ref encoder shard.

    Tolerance accounts for bf16 accumulation-order drift between the ref's
    LLM-flat (pp=1) gradient path and the dist's PP>1 1F1B path. Both paths
    yield the same DP=1 encoder gradient in exact arithmetic; bf16 rounding
    bounds the drift within the tolerance below.
    """
    ref_params = dict(ref_module.named_parameters())
    mismatches = []
    for name, dist_param in dist_module.named_parameters():
        ref_param = ref_params[name]
        assert ref_param.shape == dist_param.shape
        try:
            torch.testing.assert_close(dist_param.data, ref_param.data, rtol=rtol, atol=atol)
        except AssertionError as e:
            mismatches.append((name, str(e)))
    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  {n}: {msg}" for n, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: {len(mismatches)} encoder param(s) diverged between "
            f"PP>1 dist and equal-DP PP=1 ref:\n{details}"
        )


def _run_pp_weight_oracle(
    dist_enc_tp,
    dist_enc_dp,
    dist_llm_tp,
    dist_llm_pp,
    dist_llm_dp,
    num_microbatches,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    micro_batch_size_llm=2,
):
    """Drive the dist-vs-ref weight oracle described in
    ``test_pp_matches_pp1_equal_dp_reference``.
    """
    import os

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    encoder_name = "images"

    # Equal-DP reference: enc_tp=dist_enc_tp, enc_dp=dist_enc_dp,
    # llm_tp=dist_enc_tp (→ same encoder & LLM TP layout), llm_dp=dist_enc_dp,
    # llm_pp=1 (identity bridge, only PP value compatible with equal-DP on
    # a fixed rank count).
    ref_enc_tp, ref_enc_dp = dist_enc_tp, dist_enc_dp
    ref_llm_tp, ref_llm_pp, ref_llm_dp = dist_enc_tp, 1, dist_enc_dp

    # Global batch size spans the larger DP side. Dist's LLM DP is smaller
    # (fan-in), so each LLM rank holds micro_batch_size_llm samples.
    global_batch_size = micro_batch_size_llm * dist_llm_dp
    # For ref (equal-DP, llm_dp == enc_dp): per-rank batch = global_batch / enc_dp.
    ref_per_rank_mbs = global_batch_size // ref_llm_dp

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=False,
        bucket_size=10000,
        use_distributed_optimizer=True,
        gradient_reduce_div_factor=1,
    )

    # Build dist first (heterogeneous TP/DP + PP>1).
    torch.manual_seed(12345)
    dist_model, dist_enc_grid, dist_llm_grid, dist_lang_pg, dist_vis_pg = _build_pp_oracle_model(
        encoder_tp=dist_enc_tp,
        encoder_dp=dist_enc_dp,
        llm_tp=dist_llm_tp,
        llm_pp=dist_llm_pp,
        llm_dp=dist_llm_dp,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_length=seq_length,
        ddp_config=ddp_config,
    )
    # Build ref (equal-DP, PP=1).
    torch.manual_seed(12345)
    ref_model, ref_enc_grid, ref_llm_grid, ref_lang_pg, ref_vis_pg = _build_pp_oracle_model(
        encoder_tp=ref_enc_tp,
        encoder_dp=ref_enc_dp,
        llm_tp=ref_llm_tp,
        llm_pp=ref_llm_pp,
        llm_dp=ref_llm_dp,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_length=seq_length,
        ddp_config=ddp_config,
    )

    # Force identical initial state. Encoder: same TP/DP → shard-wise copy.
    # LLM: ref has pp=1 (all layers), dist has pp>=1 (layers split); remap.
    _copy_encoder_params(
        ref_model.modality_submodules[encoder_name].module,
        dist_model.modality_submodules[encoder_name].module,
    )
    dist_pp_rank = dist_llm_grid.get_pg("pp").rank()
    _copy_llm_params_pp_aware(
        ref_model.language_model.module,
        dist_model.language_model.module,
        pp_rank=dist_pp_rank,
        pp_size=dist_llm_pp,
        num_layers=num_layers,
        dist_tp_group=dist_llm_grid.get_pg("tp"),
        ref_tp_group=ref_llm_grid.get_pg("tp"),
    )

    # Build optimizers AFTER weight copy (distributed optimizer snapshots
    # fp32 master weights at __init__).
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    dist_optimizer = get_mimo_optimizer(dist_model, opt_config)
    ref_optimizer = get_mimo_optimizer(ref_model, opt_config)

    # Deterministic shared global data. Both models consume the same global
    # batches but slice differently:
    #   - Dist's data_iterator yields per-LLM-rank micro_batch_size samples
    #     (schedule then fan-in-slices on the encoder side).
    #   - Ref's data_iterator yields per-rank ref_per_rank_mbs samples.
    torch.manual_seed(99999)
    global_batches = _generate_shared_global_batches(
        num_batches=num_microbatches,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        encoder_name=encoder_name,
    )
    dist_llm_dp_pg = dist_llm_grid.get_pg("dp")
    ref_enc_dp_pg = ref_enc_grid.get_pg("dp")
    dist_per_rank_batches = [
        _slice_batch_along_dim0(b, dist_llm_dp, dist_llm_dp_pg.rank())
        for b in global_batches
    ]
    ref_per_rank_batches = [
        _slice_batch_along_dim0(b, ref_enc_dp, ref_enc_dp_pg.rank())
        for b in global_batches
    ]

    # ── Dist forward/backward: three-phase colocated schedule ────────────
    class _ListIter:
        def __init__(self, items):
            self._items = items
            self._i = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._i >= len(self._items):
                raise StopIteration
            v = self._items[self._i]
            self._i += 1
            return v

    dist_optimizer.zero_grad()
    colocated_forward_backward_with_pp(
        mimo_model=dist_model,
        data_iterator=_ListIter(dist_per_rank_batches),
        num_microbatches=num_microbatches,
        encoder_grid=dist_enc_grid,
        llm_grid=dist_llm_grid,
        encoder_name=encoder_name,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size_llm,
        p2p_communicator=P2PCommunicator(
            pp_group=dist_llm_grid.get_pg("pp"), config=dist_model.config
        ),
        pg_collection=dist_lang_pg,
    )
    dist_ok, dist_gn, _ = dist_optimizer.step()
    assert dist_ok, "Dist optimizer step failed"
    assert dist_gn is not None and dist_gn > 0, (
        f"Dist grad_norm={dist_gn} — three-phase schedule produced zero encoder/LLM grads."
    )

    # ── Ref forward/backward: plain no-pipelining schedule ───────────────
    def _ref_forward_step(data_iterator, model, *args):
        batch = next(data_iterator)
        output_tensor, loss_mask = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            loss_mask=batch['loss_mask'],
            position_ids=batch['position_ids'],
            modality_inputs=batch['modality_inputs'],
        )
        return output_tensor, partial(_sum_loss_func, loss_mask)

    ref_optimizer.zero_grad()
    schedule.forward_backward_no_pipelining(
        forward_step_func=_ref_forward_step,
        data_iterator=_ListIter(ref_per_rank_batches),
        model=[ref_model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=ref_per_rank_mbs,
        forward_only=False,
        pg_collection=ref_lang_pg,
    )
    ref_ok, ref_gn, _ = ref_optimizer.step()
    assert ref_ok, "Ref optimizer step failed"
    assert ref_gn is not None and ref_gn > 0, f"Ref grad_norm={ref_gn}"

    # Main oracle: post-step encoder shards match 1:1 (same enc_tp, enc_dp).
    _assert_encoder_shards_match(
        ref_model.modality_submodules[encoder_name].module,
        dist_model.modality_submodules[encoder_name].module,
        rtol=1e-2,
        atol=1e-2,
    )


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

    @pytest.mark.parametrize(
        "num_microbatches",
        [2, 4],
        ids=["num_mb_eq_pp", "num_mb_gt_pp_grad_acc"],
    )
    def test_pp_matches_pp1_equal_dp_reference(self, num_microbatches):
        """Post-step encoder weights under PP>1 match equal-DP PP=1 reference.

        This is the real correctness oracle for PR-9's three-phase schedule.
        Parallels the PR-10 oracle (``test_mimo_colocated_correctness.py``),
        extended with PP-aware LLM weight reshaping so the reference has
        ``llm_pp=1`` (the only config compatible with equal-DP on a fixed
        rank count: ``enc_tp * enc_dp == llm_tp * llm_pp * llm_dp`` and
        ``enc_dp == llm_dp`` force ``llm_pp = enc_tp / llm_tp``; with
        ``enc_tp == llm_tp`` this means ``llm_pp = 1``).

        * Dist: fan-in + PP>1 (the config under test). Runs through
          ``colocated_forward_backward_with_pp`` (three-phase schedule).
        * Ref: ``enc_tp=dist_enc_tp``, ``enc_dp=dist_enc_dp``,
          ``llm_tp=dist_enc_tp``, ``llm_dp=dist_enc_dp``, ``llm_pp=1``.
          Identity bridge (``BridgeDirection.EQUAL``); runs through
          ``forward_backward_no_pipelining``.

        Both use ``gradient_reduce_div_factor=1`` with an identical
        ``.sum()`` loss (matching the loss in ``colocated_schedule.py``),
        so the DDP reduction yields the DP=1 aggregate gradient on every
        encoder shard regardless of LLM layout. Encoder TP matches across
        the two models, so shards line up 1:1. LLM TP matches too; LLM
        weights differ only in PP partitioning, which the copy helper
        below reshapes. Under correct PP>1 encoder grad accumulation +
        broadcast, one Adam step yields shard-wise equal post-step encoder
        weights modulo bf16 accumulation drift.

        If the three-phase schedule mishandles any of: encoder grad
        accumulation across microbatches, PP-stage-0→stage-N broadcast,
        or the detach/reattach boundary, encoder shards diverge and this
        test fails.

        Two parametrized cases:
          * ``num_mb_eq_pp`` (num_microbatches=2, pp=2): the minimal
            pipeline with one microbatch per PP stage. No grad
            accumulation across microbatches.
          * ``num_mb_gt_pp_grad_acc`` (num_microbatches=4, pp=2): 1F1B
            pipeline runs with 2 microbatches per stage, so encoder
            embedding views for 4 microbatches all accumulate into the
            same ``detached_full.grad`` via PyTorch view-gradient
            semantics. If the microbatch slicing in
            ``_build_lm_microbatches`` does not produce proper views of
            ``detached_full`` (e.g., accidentally cloning), grad
            accumulation across microbatches is silently dropped and the
            encoder shards diverge.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        _run_pp_weight_oracle(
            dist_enc_tp=2,
            dist_enc_dp=4,
            dist_llm_tp=2,
            dist_llm_pp=2,
            dist_llm_dp=2,
            num_microbatches=num_microbatches,
        )
