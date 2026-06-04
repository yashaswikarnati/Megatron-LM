# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression tests for the cross-entropy loss tensor-parallel process-group contract.

Background
----------
There are two cross-entropy implementations behind ``--cross-entropy-loss-fusion``:

* **fused** (``fused_vocab_parallel_cross_entropy`` / ``te_parallel_cross_entropy``)
  derives BOTH the per-rank vocab partition range (``tp_group.rank()`` /
  ``tp_group.size()``) AND the vocab all-reduce group *solely* from the
  ``tp_group`` argument it is handed.
* **non-fused** (``vocab_parallel_cross_entropy``) ignores any passed group and
  re-queries ``parallel_state`` directly.

``LanguageModule.compute_language_model_loss`` feeds the fused path
``self.tp_group`` -- the canonicalized group (``ColumnParallelLinear`` applies
the same ``get_tensor_model_parallel_group_if_none`` canonicalization when
sharding the logits), so the fused vocab reduction stays aligned with the vocab
sharding and honors the None fallback. If a wrong group ever reached the fused
CE, the loss would be silently wrong while the non-fused loss (which re-queries
``parallel_state``) is unaffected -- i.e. toggling fusion changes the loss.
These tests pin that contract.

Run (1 node, 8 GPUs):
    uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
        tests/unit_tests/tensor_parallel/test_fused_cross_entropy_pg.py
"""
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils

SEQ = 8
BATCH = 2
VOCAB_PER_PARTITION = 16


def _make_shard_and_target(tp_group, seed=1234):
    """Build a vocab-parallel logits shard + targets that are consistent across
    the ``tp_group``, so the partitioned CE has a well-defined full-vocab value.

    Returns ``(logits_shard [s, b, v/tp], target [s, b])``. Using a fixed seed
    on every rank makes the pre-slice full logits identical across ranks; each
    rank then keeps its own complementary vocab slice.
    """
    torch.manual_seed(seed)
    world = tp_group.size()
    rank = tp_group.rank()
    full_vocab = VOCAB_PER_PARTITION * world

    full_logits = torch.randn(SEQ, BATCH, full_vocab, device="cuda")
    target = torch.randint(0, full_vocab, (SEQ, BATCH), device="cuda")

    lo = rank * VOCAB_PER_PARTITION
    hi = lo + VOCAB_PER_PARTITION
    shard = full_logits[..., lo:hi].clone().contiguous()
    return shard, target.contiguous()


@pytest.mark.internal
def test_fused_matches_nonfused_with_correct_group():
    """With the correct TP group, fused CE == non-fused CE (the reference)."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    logits, target = _make_shard_and_target(tp_group)

    fused = fused_vocab_parallel_cross_entropy(logits.clone(), target, tp_group)
    nonfused = vocab_parallel_cross_entropy(logits.clone(), target)

    torch.testing.assert_close(fused, nonfused, rtol=1e-3, atol=1e-3)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_fused_with_wrong_group_silently_diverges():
    """A wrong-but-valid TP group changes the fused loss with no error.

    Here the WORLD group (size 8) stands in for "wrong group": fused CE reduces
    vocab statistics over all ranks instead of the 2-rank TP group. This is the
    exact silent failure mode the hybrid/mamba CE-loss bug hypothesis is about.
    """
    if Utils.world_size <= 2:
        pytest.skip("Needs WORLD size > TP size to exhibit a wrong-group reduction.")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    logits, target = _make_shard_and_target(tp_group)

    correct = fused_vocab_parallel_cross_entropy(logits.clone(), target, tp_group)
    wrong = fused_vocab_parallel_cross_entropy(
        logits.clone(), target, torch.distributed.group.WORLD
    )

    # No crash, but a materially different loss -> silent corruption.
    assert (correct - wrong).abs().max().item() > 1e-2, (
        "Expected the wrong TP group to change the fused loss; if this passes, "
        "the fused CE is not actually honoring its tp_group argument."
    )
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_compute_language_model_loss_uses_canonical_tp_group():
    """`LanguageModule.compute_language_model_loss` routes the *canonical*
    `self.tp_group` (not the raw `pg_collection.tp`) to the fused CE.

    `self.tp_group` is set in `LanguageModule.__init__` via
    `get_tensor_model_parallel_group_if_none(pg_collection.tp)`, the same
    canonicalization the output layer (`ColumnParallelLinear`) applies. Reading
    that — rather than the raw `pg_collection.tp` — keeps the fused vocab
    reduction aligned with the vocab sharding and honors the None fallback.

    Exercised by calling the production method as an unbound function with a
    minimal stand-in ``self`` (the fused-native path reads ``config`` and
    ``tp_group``; the diagnostic also reads ``pg_collection.tp``).
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    logits, target_sb = _make_shard_and_target(tp_group)  # logits [s, b, v/tp]

    config = SimpleNamespace(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')
    fake_self = SimpleNamespace(
        config=config, pg_collection=SimpleNamespace(tp=tp_group), tp_group=tp_group
    )

    # compute_language_model_loss takes labels as [b, s] and returns loss [b, s].
    labels_bs = target_sb.transpose(0, 1).contiguous()
    loss = LanguageModule.compute_language_model_loss(fake_self, labels_bs, logits.clone())

    reference = vocab_parallel_cross_entropy(logits.clone(), target_sb)
    reference = reference.transpose(0, 1).contiguous()  # [s, b] -> [b, s]
    torch.testing.assert_close(loss, reference, rtol=1e-3, atol=1e-3)

    if Utils.world_size > 2:
        # Prove the method uses self.tp_group: a wrong group there changes the loss.
        fake_self.tp_group = torch.distributed.group.WORLD
        wrong = LanguageModule.compute_language_model_loss(fake_self, labels_bs, logits.clone())
        assert (
            loss - wrong
        ).abs().max().item() > 1e-2, (
            "compute_language_model_loss did not actually use self.tp_group."
        )

        # Prove the method NO LONGER reads pg_collection.tp: corrupting it while
        # tp_group stays correct must leave the loss unchanged (the fix's guard).
        fake_self.tp_group = tp_group
        fake_self.pg_collection.tp = torch.distributed.group.WORLD
        unaffected = LanguageModule.compute_language_model_loss(
            fake_self, labels_bs, logits.clone()
        )
        torch.testing.assert_close(unaffected, reference, rtol=1e-3, atol=1e-3)
    Utils.destroy_model_parallel()
