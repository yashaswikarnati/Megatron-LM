# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Gradient-scaling correctness for colocated MimoModel with dest CP>1.

Extends the equal-DP-reference oracle from
``test_mimo_colocated_correctness.py`` to a heterogeneous-DP dist model
whose LLM has ``cp > 1``. The reference is the same equal-DP CP=1
config used in the PR-10 oracle (bridge is identity passthrough,
encoder TP layout matches dist's encoder shard-for-shard); the dist
side adds CP>1 on the LLM, so this test specifically exercises:

* ``loss_func`` reducing ``(num, den)`` over ``dp * cp`` so the per-token
  grad factor stays ``1 / global_den`` instead of being scaled by
  ``cp_size``. Reduce over plain DP and the encoder grad would shrink
  by ``cp_size``.
* ``ColocatedBridgeCommunicator`` backward's intra-CP ``all_reduce``,
  which reconstructs the full-sequence gradient from the zero-padded
  per-CP-rank grad produced by ``PartitionAdapter.shard``'s
  ``index_select`` adjoint. Without it the encoder receives only the
  current CP rank's sequence chunk, dropping the rest.
* For fan-out: the per-CP-level gather groups built by
  ``_build_fan_out_gather_groups`` (a single pooled group per
  ``(src_dp, dest_tp)`` would orphan ``cp>0`` ranks).

If any of those is wrong the encoder gradient gets a non-unit factor of
``cp_size`` (or worse, drops sequence content), and one Adam step is
enough to make the encoder shards diverge from the CP=1 reference.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_colocated_correctness_cp.py -v -s
"""

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.test_mimo_colocated_correctness import (
    _assert_encoder_weights_match,
    _copy_ref_params_to_dist,
    _generate_and_broadcast_global_batches,
    _run_forward_backward,
    _set_deterministic_env,
    _slice_global_batch_by_dp,
    _slice_global_batch_for_dist,
    _wire_training_hooks,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.3.0"),
    reason="Requires PyTorch 2.3+",
)
class TestColocatedCPCorrectness:
    """Equal-DP CP=1 reference oracle for heterogeneous-DP dist with LLM CP>1."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        destroy_all_grids()

    @pytest.mark.parametrize(
        "enc_tp,enc_dp,llm_tp,llm_dp,llm_cp",
        [(2, 4, 2, 2, 2), (4, 2, 1, 4, 2)],
        ids=["fan_in_cp2", "fan_out_cp2"],
    )
    def test_cp_dist_matches_cp1_reference_post_step_weights(
        self, enc_tp, enc_dp, llm_tp, llm_dp, llm_cp
    ):
        """Hetero-DP+CP>1 dist post-step encoder weights match equal-DP CP=1 ref.

        Both sides use ``gradient_reduce_div_factor=1`` and the num+den
        global-mean CE so the DDP reduction is a pure SUM and the
        aggregate grad on every encoder shard equals the DP=1 gradient.
        Encoder TP and per-rank batch are matched between dist and ref so
        encoder shards line up 1:1 for direct comparison.

        On the dist side the LLM additionally runs CP>1, so the encoder-
        bound gradient must survive three transforms unchanged:
          * loss reduction over ``dp*cp`` (not just dp),
          * bridge backward intra-CP all_reduce(SUM),
          * (fan-out only) per-CP-level fan-out gather groups.

        Any of those mis-scoped scales the encoder grad by ``cp_size``
        (or drops sequence content), and one Adam step makes shards
        diverge from the ref beyond bf16 rounding.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        # seq_length must be divisible by 2*llm_cp — PartitionAdapter's
        # causal load-balancing splits each sequence into 2*cp chunks.
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        micro_batch_size = 2
        num_microbatches = 1

        global_batch_size = micro_batch_size * max(enc_dp, llm_dp)

        # Dist: heterogeneous TP/DP, llm_cp>1. Ref: equal-DP uniform with the
        # SAME encoder TP/DP as dist so the bridge is identity and encoder
        # shards align 1:1 for direct comparison. Ref keeps cp=1; CP>1 lives
        # only on the dist side because that is the path under audit.
        dist_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        dist_llm_grid = create_hypercomm_grid(
            offset=0, tp=llm_tp, cp=llm_cp, pp=1, dp=llm_dp
        )
        ref_enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        ref_llm_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        create_all_embedding_groups(
            [dist_enc_grid, dist_llm_grid, ref_enc_grid, ref_llm_grid]
        )

        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            bucket_size=10000,
            use_distributed_optimizer=True,
            gradient_reduce_div_factor=1,
        )

        torch.manual_seed(12345)
        dist_mimo, _, _, dist_language_pg, dist_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=dist_enc_grid,
            llm_grid=dist_llm_grid,
            hidden_size=hidden_size,
            num_layers=2,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
        )
        dist_mimo.model_type = ModelType.encoder_or_decoder

        torch.manual_seed(12345)
        ref_mimo, _, _, ref_language_pg, ref_vision_pg = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=ref_enc_grid,
            llm_grid=ref_llm_grid,
            hidden_size=hidden_size,
            num_layers=2,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=ddp_config,
        )
        ref_mimo.model_type = ModelType.encoder_or_decoder

        # Encoder TP layouts match between dist and ref → shard-to-shard
        # copy. LLM TP differs (and dist additionally has CP, but CP does
        # not reshape weights), so the helper all-gathers ref's shards
        # across ref's LLM TP group and re-slices for dist's LLM TP group.
        _copy_ref_params_to_dist(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            ref_enc_grid.get_pg("tp"),
            dist_enc_grid.get_pg("tp"),
        )
        _copy_ref_params_to_dist(
            ref_mimo.language_model.module,
            dist_mimo.language_model.module,
            ref_llm_grid.get_pg("tp"),
            dist_llm_grid.get_pg("tp"),
        )

        _wire_training_hooks(dist_mimo, dist_language_pg, dist_vision_pg)
        _wire_training_hooks(ref_mimo, ref_language_pg, ref_vision_pg)

        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            bf16=True,
            use_distributed_optimizer=True,
        )
        dist_optimizer = get_mimo_optimizer(dist_mimo, opt_config)
        ref_optimizer = get_mimo_optimizer(ref_mimo, opt_config)

        torch.manual_seed(99999)
        global_batches = _generate_and_broadcast_global_batches(
            global_mbs=global_batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_name=encoder_name,
            num_batches=num_microbatches,
        )
        # Dist: pre-slice along the larger DP side; forward_step further
        # slices the encoder/LLM side as needed. CP does not affect the
        # batch dim so the helper is reused unchanged.
        dist_batches = [
            _slice_global_batch_for_dist(b, dist_enc_grid, dist_llm_grid)
            for b in global_batches
        ]
        # Ref is equal-DP (enc_dp == llm_dp) so the dist helper would
        # return the full batch; slice explicitly so each rank sees the
        # same per-rank encoder batch as dist's encoder.
        ref_batches = [
            _slice_global_batch_by_dp(b, ref_enc_grid.get_pg("dp"))
            for b in global_batches
        ]
        ref_per_rank_batch_size = global_batch_size // enc_dp

        dist_optimizer.zero_grad()
        _run_forward_backward(
            mimo_model=dist_mimo,
            batches=dist_batches,
            enc_grid=dist_enc_grid,
            llm_grid=dist_llm_grid,
            encoder_name=encoder_name,
            language_pg=dist_language_pg,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            num_microbatches=num_microbatches,
        )
        dist_success, dist_grad_norm, _ = dist_optimizer.step()
        assert dist_success, "Dist optimizer step failed"
        assert dist_grad_norm is not None and dist_grad_norm > 0, (
            f"Dist grad_norm={dist_grad_norm} — encoder grads may have been "
            "silently zeroed by wrong CP scaling"
        )

        ref_optimizer.zero_grad()
        _run_forward_backward(
            mimo_model=ref_mimo,
            batches=ref_batches,
            enc_grid=ref_enc_grid,
            llm_grid=ref_llm_grid,
            encoder_name=encoder_name,
            language_pg=ref_language_pg,
            micro_batch_size=ref_per_rank_batch_size,
            seq_length=seq_length,
            num_microbatches=num_microbatches,
        )
        ref_success, ref_grad_norm, _ = ref_optimizer.step()
        assert ref_success, "Ref optimizer step failed"
        assert ref_grad_norm is not None and ref_grad_norm > 0, (
            f"Ref grad_norm={ref_grad_norm}"
        )

        # Loose-ish tolerance because dist and ref differ on the LLM side
        # (different llm_tp, additionally cp>1 on dist) — bf16 accumulation
        # noise from the LLM forward propagates into each model's encoder
        # gradient. Mirrors the tolerances in the CP=1 correctness test.
        _assert_encoder_weights_match(
            ref_mimo.modality_submodules[encoder_name].module,
            dist_mimo.modality_submodules[encoder_name].module,
            rtol=1e-3,
            atol=1e-3,
        )
