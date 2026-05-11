# E2E Training Parity Plan

This note tracks the plan for checking end-to-end training parity between the
previous `examples/mimo/train.py` flow from `feat/nemotron-moe-vlm-mimo` and the
new heterogenous `examples/mimo/train_hetero.py` flow.

## Goal

Verify that the new heterogenous MIMO training loop matches the previous
Megatron `pretrain()`-based flow for the Nemotron 20L VLM workflow. The strongest
parity signal is matching behavior on a frozen batch stream before comparing live
Energon training runs.

## Plan

1. Compare resolved training configuration.
   - Dump the final args used by old `train.py`.
   - Dump the final args used by new `train_hetero.py`.
   - Compare behavior-relevant fields: model config, vision config, MoE config,
     TP/PP/EP/ETP/EDP, batch sizes, optimizer, scheduler, seeds, loss scaling,
     per-token loss, and dataloader settings.

2. Start both runs from the same initial weights.
   - Prefer a canonical initialized checkpoint or state dict over relying only on
     seed-based initialization.
   - Compare parameter hashes by logical module: vision encoder, LLM backbone,
     MoE experts, router parameters, and projector/MIMO bridge.

3. Validate data parity before training.
   - First use a recorded frozen batch stream, not live Energon.
   - Dump exact batch tensors and metadata from the old path: tokens, labels,
     loss mask, position ids, modality inputs, packed sequence params, and sample
     signatures if available.
   - Feed the same frozen batches to the new heterogenous loop and compare batch
     hashes before forward.

4. Run forward-only parity.
   - Use the same initialized weights and same frozen batch.
   - Disable optimizer updates.
   - Compare logits checksums where practical, unreduced loss numerator, token
     denominator, normalized loss, and auxiliary/router losses.

5. Run single-step training parity.
   - Use the same frozen batch.
   - Run forward, backward, optimizer step, and LR scheduler step.
   - Compare loss before step, grad norm, skipped/nan flags, LR, selected
     parameter deltas, and post-step parameter hashes.

6. Run short frozen-stream loss-curve parity.
   - Use a fixed stream of 10 to 20 frozen batches.
   - Compare per-iteration loss, grad norm, LR, loss scale, skipped/nan counts,
     consumed samples, and token counts.

7. Run actual Energon parity.
   - Run the old `train.py` flow and the new `train_hetero.py` flow against the
     real Nemotron 20L Energon setup.
   - Log sample signatures per global step in both paths.
   - First verify that both paths consume the same samples in the same order.
   - Compare loss curves only after sample order parity is established.

## Expected Limits

Bitwise parity may not be realistic between the old colocated Megatron
`pretrain()` path and the new non-colocated heterogenous grids because collective
ordering, parameter partitioning, and optimizer sharding can differ. The first
strict gates should therefore be configuration parity, initial-weight parity,
frozen-batch forward parity, token-count parity, LR schedule parity, and a short
frozen-batch training curve within a tight tolerance.

The known parity gap is the old `--use-loss-scaling` path. The new heterogenous
loop uses per-token global loss normalization, but it does not yet implement the
old optional sqrt-weighted scaled loss behavior.
