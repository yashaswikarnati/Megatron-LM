# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for colocated MIMO training with LLM PP>1.

Run individually (8 GPUs):
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        -m pytest tests/unit_tests/models/test_mimo_colocated_pp.py -v
"""

import re
from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.colocated_schedule import colocated_forward_backward_with_pp
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.transformer.enums import ModelType
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    build_no_sync_func,
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.test_mimo_colocated_correctness import (
    _assert_encoder_weights_match,
    _BatchIterator,
    _copy_ref_params_to_dist,
    _generate_and_broadcast_global_batches,
    _slice_batch,
    _wire_training_hooks,
)
from tests.unit_tests.test_utilities import Utils


def _wire_pp_training_hooks(mimo_model, language_pg, vision_pg, llm_grid):
    """PP-aware variant of ``_wire_training_hooks`` for LLM PP>1.

    ``calculate_per_token_loss=True`` on both sub-model configs pins DDP's
    gradient_scaling_factor to 1.0 (pure SUM across DP). But with LLM PP>1,
    the inner schedule only populates ``num_tokens`` on the last PP stage;
    non-last stages get 0. Before the DP all-reduce, this helper broadcasts
    ``num_tokens`` from the last LLM PP rank to earlier ones so every rank
    arrives at the same ``N_global`` and the per-token divisor lands
    uniformly on encoder + LLM grads.
    """

    no_sync_func = build_no_sync_func(mimo_model)
    pp_group = llm_grid.get_pg("pp")

    def finalize_grads_func(model_list, num_tokens, force_all_reduce=False, **kwargs):
        assert num_tokens is not None, (
            "finalize_grads_func expects calculate_per_token_loss=True on the "
            "TransformerConfig so the schedule forwards total_num_tokens; got None."
        )

        if pp_group.size() > 1:
            last_rank = dist.get_global_rank(pp_group, pp_group.size() - 1)
            dist.broadcast(num_tokens, src=last_rank, group=pp_group)

        llm_dp_pg = language_pg.dp_cp if language_pg.dp_cp is not None else language_pg.dp
        dist.all_reduce(num_tokens, group=llm_dp_pg, op=dist.ReduceOp.SUM)
        n_global = num_tokens.item()

        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=None,
                pg_collection=language_pg,
                force_all_reduce=force_all_reduce,
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=vision_pg,
                    force_all_reduce=force_all_reduce,
                )

        if n_global > 0:
            inv = 1.0 / n_global
            if mimo_model.language_model is not None:
                mimo_model.language_model.scale_gradients(inv)
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    submodule.scale_gradients(inv)

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )


def _assert_llm_weights_match_pp_aware(
    ref_module, dist_module, pp_rank, pp_size, num_layers, rtol=1e-2, atol=1e-2
):
    """Assert dist LLM shards match ref (PP=1) via the same layer-index remap
    ``_copy_llm_params_pp_aware`` uses. Non-layer params (embedding,
    final_layernorm, output_layer) only exist on stages that own them.
    """
    layers_per_stage = num_layers // pp_size
    layer_rx = re.compile(r'^(.*decoder\.layers\.)(\d+)(\..*)$')
    ref_params = dict(ref_module.named_parameters())

    mismatches = []
    for name, dist_param in dist_module.named_parameters():
        m = layer_rx.match(name)
        if m:
            prefix, local_idx_s, suffix = m.groups()
            global_idx = pp_rank * layers_per_stage + int(local_idx_s)
            ref_name = f"{prefix}{global_idx}{suffix}"
        else:
            ref_name = name
        assert ref_name in ref_params, (
            f"LLM param '{name}' maps to ref '{ref_name}' which does not exist "
            f"(ref has llm_pp=1)."
        )
        ref_param = ref_params[ref_name]
        assert ref_param.shape == dist_param.shape, (
            f"LLM param '{name}': ref.shape={tuple(ref_param.shape)} != "
            f"dist.shape={tuple(dist_param.shape)}."
        )
        try:
            torch.testing.assert_close(
                dist_param.data, ref_param.data, rtol=rtol, atol=atol
            )
        except AssertionError as e:
            mismatches.append((name, ref_name, str(e)))

    if mismatches:
        rank = dist.get_rank()
        details = "\n".join(f"  {n} -> {rn}: {msg}" for n, rn, msg in mismatches)
        raise AssertionError(
            f"Rank {rank}: {len(mismatches)} LLM param(s) diverged between "
            f"PP>1 dist model and PP=1 reference:\n{details}"
        )


def _copy_llm_params_pp_aware(ref_module, dist_module, pp_rank, pp_size, num_layers):
    """Copy LLM params ref (PP=1) → dist (PP>=1) with layer-index remapping.

    Dist's ``decoder.layers.{local_idx}`` on PP stage ``s`` corresponds to
    ref's global layer ``s * layers_per_stage + local_idx``. Non-layer
    params (embedding, final_layernorm, output_layer) are only present on
    stages that own them and their names match exactly between ref and
    dist. Assumes ``dist_llm_tp == ref_llm_tp`` so shards line up 1:1.
    """
    assert num_layers % pp_size == 0, (
        f"num_layers={num_layers} not divisible by pp_size={pp_size}; "
        f"oracle requires even PP split."
    )
    layers_per_stage = num_layers // pp_size
    layer_rx = re.compile(r'^(.*decoder\.layers\.)(\d+)(\..*)$')
    ref_params = dict(ref_module.named_parameters())

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
            assert ref_param.shape == dist_param.shape, (
                f"LLM param '{name}': ref.shape={tuple(ref_param.shape)} != "
                f"dist.shape={tuple(dist_param.shape)} — oracle requires "
                f"dist_llm_tp == ref_llm_tp."
            )
            dist_param.data.copy_(ref_param.data.to(dist_param.dtype))


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
    """Drive the dist (PP>1) vs ref (PP=1, equal-DP) weight oracle."""
    import os

    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    encoder_name = "images"

    # Equal-DP reference: same encoder TP/DP; LLM matches encoder TP/DP and
    # uses PP=1 (the only PP value compatible with equal-DP on a fixed rank
    # count when enc_tp == llm_tp).
    ref_enc_tp, ref_enc_dp = dist_enc_tp, dist_enc_dp
    ref_llm_tp, ref_llm_pp, ref_llm_dp = dist_enc_tp, 1, dist_enc_dp

    global_batch_size = micro_batch_size_llm * dist_llm_dp
    ref_per_rank_mbs = global_batch_size // ref_llm_dp

    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        bucket_size=10000,
        use_distributed_optimizer=True,
    )

    dist_enc_grid = create_hypercomm_grid(
        offset=0, tp=dist_enc_tp, cp=1, pp=1, dp=dist_enc_dp
    )
    dist_llm_grid = create_hypercomm_grid(
        offset=0, tp=dist_llm_tp, cp=1, pp=dist_llm_pp, dp=dist_llm_dp
    )
    ref_enc_grid = create_hypercomm_grid(
        offset=0, tp=ref_enc_tp, cp=1, pp=1, dp=ref_enc_dp
    )
    ref_llm_grid = create_hypercomm_grid(
        offset=0, tp=ref_llm_tp, cp=1, pp=ref_llm_pp, dp=ref_llm_dp
    )
    create_all_embedding_groups([dist_enc_grid, dist_llm_grid, ref_enc_grid, ref_llm_grid])

    torch.manual_seed(12345)
    dist_model, _, _, dist_lang_pg, dist_vis_pg = get_mimo_model(
        encoder_name=encoder_name,
        encoder_grid=dist_enc_grid,
        llm_grid=dist_llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
        ddp_config=ddp_config,
        bf16=False,
        bias=False,
        dropout=False,
        per_token_loss=True,
    )
    dist_model.model_type = ModelType.encoder_or_decoder

    torch.manual_seed(12345)
    ref_model, _, _, ref_lang_pg, ref_vis_pg = get_mimo_model(
        encoder_name=encoder_name,
        encoder_grid=ref_enc_grid,
        llm_grid=ref_llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
        ddp_config=ddp_config,
        bf16=False,
        bias=False,
        dropout=False,
        per_token_loss=True,
    )
    ref_model.model_type = ModelType.encoder_or_decoder

    _copy_ref_params_to_dist(
        ref_model.modality_submodules[encoder_name].module,
        dist_model.modality_submodules[encoder_name].module,
        ref_enc_grid.get_pg("tp"),
        dist_enc_grid.get_pg("tp"),
    )
    _copy_llm_params_pp_aware(
        ref_model.language_model.module,
        dist_model.language_model.module,
        pp_rank=dist_llm_grid.get_pg("pp").rank(),
        pp_size=dist_llm_pp,
        num_layers=num_layers,
    )

    _wire_pp_training_hooks(dist_model, dist_lang_pg, dist_vis_pg, dist_llm_grid)
    _wire_training_hooks(ref_model, ref_lang_pg, ref_vis_pg)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=False,
        use_distributed_optimizer=True,
    )
    dist_optimizer = get_mimo_optimizer(dist_model, opt_config)
    ref_optimizer = get_mimo_optimizer(ref_model, opt_config)

    torch.manual_seed(99999)
    global_batches = _generate_and_broadcast_global_batches(
        global_mbs=global_batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        encoder_name=encoder_name,
        num_batches=num_microbatches,
    )
    dist_batches = [
        _slice_batch(b, dist_llm_dp, dist_llm_grid.get_pg("dp").rank())
        for b in global_batches
    ]
    ref_batches = [
        _slice_batch(b, ref_enc_dp, ref_enc_grid.get_pg("dp").rank())
        for b in global_batches
    ]

    dist_optimizer.zero_grad()
    colocated_forward_backward_with_pp(
        mimo_model=dist_model,
        data_iterator=_BatchIterator(dist_batches),
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
        f"Dist grad_norm={dist_gn} — three-phase schedule produced zero grads."
    )

    def _sum_loss(loss_mask, output_tensor):
        """Per-token-loss 3-tuple matching ``_wire_training_hooks`` contract."""
        if output_tensor is None:
            zero_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
            zero_count = torch.tensor(0, device='cuda', dtype=torch.int)
            return zero_loss, zero_count, {'loss_reduced': 0.0}
        masked = output_tensor.float() * loss_mask.float()
        local_sum = masked.sum()
        local_num_tokens = loss_mask.float().sum().to(torch.int)
        return local_sum, local_num_tokens, {'loss_reduced': local_sum.detach().item()}

    def _ref_forward_step(data_iterator, model, *args):
        batch = next(data_iterator)
        output_tensor, loss_mask = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            loss_mask=batch['loss_mask'],
            position_ids=batch['position_ids'],
            modality_inputs=batch['modality_inputs'],
        )
        return output_tensor, partial(_sum_loss, loss_mask)

    ref_optimizer.zero_grad()
    schedule.forward_backward_no_pipelining(
        forward_step_func=_ref_forward_step,
        data_iterator=_BatchIterator(ref_batches),
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

    # LLM forward differs between 1F1B (dist, PP>1) and no-pipelining (ref),
    # and TP shards may accumulate in a different order; keep tolerances
    # loose enough to absorb that drift even in fp32.
    _assert_encoder_weights_match(
        ref_model.modality_submodules[encoder_name].module,
        dist_model.modality_submodules[encoder_name].module,
        rtol=1e-2,
        atol=1e-2,
    )
    _assert_llm_weights_match_pp_aware(
        ref_model.language_model.module,
        dist_model.language_model.module,
        pp_rank=dist_llm_grid.get_pg("pp").rank(),
        pp_size=dist_llm_pp,
        num_layers=num_layers,
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

    @pytest.mark.parametrize(
        "num_microbatches",
        [2, 4],
        ids=["num_mb_eq_pp", "num_mb_gt_pp_grad_acc"],
    )
    def test_pp_matches_pp1_equal_dp_reference(self, num_microbatches):
        """Post-step encoder weights under PP>1 match equal-DP PP=1 reference.

        Dist runs ``colocated_forward_backward_with_pp`` (three-phase
        schedule with PP=2 on the LLM); ref runs
        ``forward_backward_no_pipelining`` with matching encoder TP/DP and
        LLM PP=1. Under correct PP>1 encoder grad accumulation + broadcast,
        one Adam step yields shard-wise equal post-step encoder weights
        (modulo bf16 drift).

        The ``num_mb_gt_pp_grad_acc`` case runs more microbatches than PP
        stages so encoder embedding views for every microbatch must
        accumulate into the same ``detached_full.grad`` via PyTorch
        view-gradient semantics — a regression there surfaces here.
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
