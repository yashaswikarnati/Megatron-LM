# E2E Training Parity Plan

This note tracks the validation plan for proving that the new heterogeneous MIMO
training loop preserves the behavior of the existing Megatron-style homogeneous
path before we use it for larger Nemotron VLM runs.

## Goal

Validate the new hetero training loop by comparing it against a homogeneous
baseline built in the same branch. The homogeneous runner initializes Megatron
`parallel_state`, uses the same current model provider, starts from the same
explicit checkpoint, and consumes the same deterministic sample stream. This
keeps the old branch as reference material only; the executable comparator lives
in the current branch so model/provider changes are not another variable.

## Harness

The validation entry points are:

- `examples/mimo/validation/training_parity.py`
- `examples/mimo/validation/compare_training_parity.py`
- `examples/mimo/scripts/run_mimo_training_parity.sh`

The runner has three modes:

- `init`: build a colocated MIMO model under `parallel_state` and save the
  initial state.
- `homo`: run the homogeneous baseline with standard Megatron process groups,
  MCore DDP, distributed optimizer, optimizer step, LR scheduler step, and loss
  logging.
- `hetero`: run the new non-colocated hetero runtime, bridge communicator,
  MIMO optimizer, grad finalization, optimizer step, LR scheduler step, and loss
  logging.

The deterministic iterator assigns global sample ids by LLM DP lane. In the
`encoder_dp < llm_dp` case, the encoder iterator consumes the union of the LLM
samples for the microbatch and uses bridge split metadata to route the matching
image embeddings back to each LLM DP rank.

## Success Criteria

1. Short strict parity, equal DP:
   - same initial checkpoint
   - deterministic data stream
   - same consumed sample ids
   - compare losses, params, grads, and optimizer states after each step
   - state tolerance: `atol=2e-4`, `rtol=2e-4`
   - loss tolerance: `1e-5`

2. Long loss-curve parity, equal DP:
   - 250 iterations
   - same consumed sample ids
   - compare loss curves only
   - loss tolerance: `2e-4`

3. Fanout parity, `encoder_dp < llm_dp`:
   - `llm_dp=2`, `encoder_dp=1`
   - same initial checkpoint
   - deterministic data stream
   - same consumed sample ids
   - compare losses, params, grads, and optimizer states after each step
   - state tolerance: `atol=1e-3`, `rtol=1e-3`
   - loss tolerance: `1e-4`

The fanout state tolerance is intentionally looser than the equal-DP strict
gate. The baseline computes each encoder sample on separate DP replicas, while
the hetero path computes the same samples on one encoder rank with a larger
local batch and routes activations to the LLM ranks. That changes local batch
shape and floating-point accumulation order, so the fanout gate checks close
numerical agreement plus exact sample routing rather than treating it as a
bitwise-equivalent same-layout comparison.

## CW Results

All jobs below ran through Cog on CW with `--skip-uv-sync` to reuse the prepared
cluster uv environment.

| Case | Job | Result | Max loss diff | Max state abs diff |
| --- | --- | --- | --- | --- |
| short | `11710390` | pass | `2.5431315098245477e-06` | `5.97536563873291e-05` |
| curve | `11710442` | pass | `0.00011698404947946273` | n/a |
| fanout | `11710246` | pass | `6.230672200580045e-05` | `0.0008821713272482157` |

Artifact roots:

- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/ykarnati/cog-scratch/runs/nmfw-464-validation-parity/mimo_training_parity_short_11710390`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/ykarnati/cog-scratch/runs/nmfw-464-validation-parity/mimo_training_parity_curve_11710442`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/ykarnati/cog-scratch/runs/nmfw-464-validation-parity/mimo_training_parity_fanout_11710246`

## Commands

```bash
cog submit --repo . --run-name nmfw-464-validation-parity --cluster-name cw-dfw \
  --partition batch --gpus 2 --time 00:20:00 --job-name parity-short --skip-uv-sync \
  --command 'PARITY_CASE=short examples/mimo/scripts/run_mimo_training_parity.sh'

cog submit --repo . --run-name nmfw-464-validation-parity --cluster-name cw-dfw \
  --partition batch --gpus 2 --time 00:35:00 --job-name parity-curve --skip-uv-sync \
  --command 'PARITY_CASE=curve examples/mimo/scripts/run_mimo_training_parity.sh'

cog submit --repo . --run-name nmfw-464-validation-parity --cluster-name cw-dfw \
  --partition batch --gpus 3 --time 00:25:00 --job-name parity-fanout --skip-uv-sync \
  --command 'PARITY_CASE=fanout examples/mimo/scripts/run_mimo_training_parity.sh'
```

## Remaining Extensions

- Add a real Energon frozen-stream variant once we want data-loader parity in
  this same harness.
- Add larger Nemotron 20L smoke runs after the mock parity gates stay stable.
- Consider a dedicated fanout reference that runs the encoder with the same
  local batch shape as the hetero path if we need a stricter fanout state gate.
