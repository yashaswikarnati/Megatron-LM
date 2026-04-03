# Campaign Strategy — Hill-Climbing Protocol

Load this when entering `campaign` mode. This drives how campaigns intelligently maximize throughput and showcase hetero benefits.

---

## Starting a Campaign

1. **Ask the user:**
   - What's the goal? (e.g., "show hetero scales from 1N→2N", "find where 12B encoder wins")
   - Which models from the NMFW-58 catalog?
   - Which data configs (vision fraction, seq lengths)?
   - Weak scaling or strong scaling?
   - What does success look like? (quantified: "+X% on Y configs")

2. **Create campaign directory** with plan.md, empty timeline/leaderboard/learnings

3. **Register in CAMPAIGNS.md:**
   ```
   | nmfw57_phase1_2node | logs/mimo_campaigns/nmfw57_phase1_2node | .worktrees/nmfw-57-multinode-benchmarking | ykarnati/nmfw-57-... | active | 2026-04-01 | Scale 1N hero to 2N |
   ```

4. **Write plan.md** with phases, priority order, and success criteria

5. **Show the first batch of configs to the user** before submitting

## Resuming a Campaign

1. Read `${CAMPAIGN_LOGS}/CAMPAIGNS.md` to find the campaign
2. `cd` to the worktree listed in the registry
3. Read from the campaign directory:
   - `plan.md` — what we're trying to achieve
   - `leaderboard.md` — current best results
   - `learnings.md` — what's confirmed, disproven, and untested
   - `timeline.md` — last experiment run (for context)
4. Pick up from the top hypothesis in learnings.md
5. Continue the hill-climbing loop

---

## The Hill-Climbing Loop

```
1. Read leaderboard.md + learnings.md
2. Pick the highest-value experiment from hypotheses in learnings.md
3. State HYPOTHESIS before running:
   "I expect [config] to [outcome] because [reasoning]. Expected: [quantified]."
4. Generate config YAML, write to campaign configs/
5. Spawn runner agent, wait for results
6. LOG IMMEDIATELY (blocking — before anything else):
   a. Append to timeline.md
   b. Update leaderboard.md if top-10 result
   c. Update learnings.md:
      - Hypothesis confirmed? → move to Confirmed with evidence
      - Hypothesis wrong? → move to Disproven with explanation
      - New insight? → add to Confirmed or new Hypothesis
7. REFLECT:
   - Did it beat the leaderboard? Why or why not?
   - Does this change any existing learnings?
   - What's the next highest-value experiment?
8. Repeat from 1
```

---

## learnings.md Structure

This is the brain of the campaign. Read it before every experiment.

```markdown
# Learnings: <campaign_name>

## Confirmed (evidence from experiments)
- [L1] <pattern> — evidence: exp003 vs exp001 showed X
- [L2] <pattern> — evidence: exp005, exp007, exp009 all confirm

## Disproven (tried and failed)
- [D1] ~~<hypothesis>~~ — exp006 showed Y instead because Z

## Hypotheses (untested, priority-ordered)
- [H1] <what to try> — expected impact: +X% — reasoning: <why>
- [H2] <what to try> — expected impact: +X% — reasoning: <why>

## Operational Notes
- <OOM patterns, config gotchas, timing observations>
```

**Rules for maintaining learnings:**
- Every experiment must update at least one entry (confirm, disprove, or add)
- Re-sort hypotheses by expected impact after each experiment
- If a confirmed learning is contradicted by new data, demote it with explanation
- Never delete disproven entries — they prevent re-running dead ends

---

## The NMFW-53 Playbook (What Worked at 1 Node)

These are proven strategies from ~200 experiments across 5 models. Use them as the starting playbook for new campaigns and adapt as you learn.

### Strategy 1: Vision density is the #1 lever

Higher vision token fraction → more encoder compute → larger hetero advantage.

| Lever | Effect (6B+7B) | How |
| -- | -- | -- |
| enc_seq 576→2048 | -1.7% → +6.3% | Longer per-image sequences |
| 1→2→3 images | +6.3% → +10.7% → +14.6% | More images per sample |
| Both combined | up to +17% | 4 img × 2048 at 100% vision |

**Always start campaigns at high vision density (75-100%) where hetero advantage is largest.** Then sweep down to find the crossover point.

### Strategy 2: Encoder TP reduction is the mechanism

Hetero wins by giving the encoder lower TP than homo forces on it.

| TP reduction | Per-mb speedup | Why |
| -- | -- | -- |
| TP8→TP4 | -13.6% | H/GPU doubles: 400→800 |
| TP4→TP2 | -4.3% | H/GPU doubles: 1024→2048 |
| TP2→TP1 | ~-10% | Zero TP communication |

**Push encoder to TP1 whenever memory allows.** At multi-node, more memory headroom makes this feasible for larger encoders.

### Strategy 3: Homo baseline must be strong

The homo baseline is the **lowest TP that fits** — not the obvious TP. NMFW-53 learned:
- 6B+7B homo at TP4/DP2 was strong (TP2 OOM'd). Hetero beat it by +17%.
- 3B+7B homo at TP4/DP2 was strong. But TP8/DP1 was tried as an alternative baseline — hetero beat that by +7.1%.
- Always try 2-3 homo configs to find the real strongest baseline.

### Strategy 4: GBS has minimal impact on the delta

The per-microbatch fwd_bwd speedup is constant across GBS=16/32/64. Higher GBS slightly inflates the TFLOPs delta (optimizer amortization) but the core advantage is in the forward/backward.

**Use GBS=32 or 64 as the standard. Don't waste experiments sweeping GBS — the delta is stable.**

### Strategy 5: Memory is the binding constraint

At 1 node, many configs OOM. Multi-node unlocks:
- Encoder at TP1 for 6B encoder (needs ~24GB params+grads)
- LLM seq > 8192 (activation memory)
- Higher mbs for some configs
- Larger models entirely (14B+ LLM)

**When a config OOMs:** report which module, which phase, peak memory. The supervisor decides whether to increase TP, reduce mbs, or skip.

---

## Self-Critique Protocol

After every 3 experiments, ask yourself:

1. **Am I hill-climbing?** Is the current experiment designed to beat the leaderboard, or am I exploring sideways? Exploration is OK occasionally but must be deliberate and justified.

2. **Are my learnings still correct?** Does new data contradict anything in the Confirmed section? If so, update immediately.

3. **Am I stuck?** If the last 3 experiments didn't improve, change strategy — try a different dimension (model size, seq length, vision density, parallelism) rather than micro-optimizing the same config.

4. **Am I cheating on GBS?** `GBS = mbs × llm_dp × nmb`. When DP or nmb changes, GBS changes. Every comparison must have matching GBS.

5. **Am I repeating a dead end?** Check learnings.md Disproven section before proposing any experiment.

---

## Campaign Phases (typical structure)

### Phase 1: Establish baselines
- For each model in scope, run the strongest homo config at the primary data config (75% vision, 8K seq)
- Record in leaderboard as the target to beat

### Phase 2: Hetero comparison at primary data config
- For each model, run hetero with encoder at min TP
- Compare against homo baseline. Log delta and reasoning.
- This gives the headline numbers.

### Phase 3: Vision density sweep
- For the models with positive hetero delta, sweep vision fraction: 25/50/75/100%
- Map the curve: at what vision % does hetero cross over to positive?
- Confirms NMFW-53 finding that vision density is the #1 predictor

### Phase 4: Sequence length sweep
- Test 8K, 16K, 32K LLM seq for configs that fit
- Does longer context amplify or dilute the hetero advantage?

### Phase 5: Optimizer tuning (multi-node)
- For configs with high encoder DP (DP≥8), test num_distributed_optimizer_instances
- For 1B encoder configs, test non-distributed optimizer
- Goal: reduce optimizer step overhead without affecting fwd/bwd

### Phase 6: Scaling study (if multi-node)
- Repeat Phase 1-3 at next node count
- Compare per-microbatch fwd_bwd time across node counts
- Does hetero advantage grow or shrink with scale?

---

## leaderboard.md Structure

```markdown
# Leaderboard: <campaign_name>

## GBS=64 (primary)

| Rank | Model | Type | Enc TP/DP | LLM TP/DP | mbs | nmb | TFLOPs | fwd_bwd/mb | Mem | Exp |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 1 | 6B+7B | hetero | 1/16 | 4/4 | 2 | 8 | 504 | 484ms | 69GB | exp012 |
| 2 | 6B+7B | homo | 4/4 | 4/4 | 2 | 8 | 430 | 560ms | 71GB | exp011 |
| ... |

## Comparison Summary

| Model | Best Homo | Best Hetero | Delta | Best Vision Config | Exp IDs |
| -- | -- | -- | -- | -- | -- |
| 6B+7B | 430 TFLOPs | 504 TFLOPs | +17.0% | 100% vis, 4×2048 | exp011/exp012 |
```

---

## timeline.md Structure

```markdown
# Timeline: <campaign_name>

## Exp 001: 6b7b_homo_tp4dp4_75vis_8k
- **Job:** 12345 | **Status:** COMPLETED
- **Config:** configs/exp001_6b7b_homo_tp4dp4_75vis_8k.yaml
- **Result:** 430 TFLOPs | 560ms/mb | 71GB
- **GBS:** 64 (mbs=2, dp=4, nmb=8)
- **Hypothesis:** Establish homo baseline at TP4/DP4
- **Outcome:** Baseline set. TP4 is min viable for 6B+7B at 2 nodes.
- **Learning update:** Added [L1] homo baseline for 6B+7B = 430 TFLOPs

## Exp 002: 6b7b_hetero_enc_tp1dp16_llm_tp4dp4_75vis_8k
- **Job:** 12346 | **Status:** COMPLETED
- **Config:** configs/exp002_....yaml
- **Result:** 504 TFLOPs | 484ms/mb | 69GB
- **GBS:** 64 (mbs=2, dp=4, nmb=8)
- **Hypothesis:** Enc TP1 eliminates all TP comm → expect +10-15% vs homo
- **Outcome:** +17% — bigger than expected. DP16 helps more than predicted.
- **Learning update:** Confirmed [L2], updated [H3] to test 50% vision next
```

---

## When to Stop

- **Success criteria met** — the goal from plan.md is achieved
- **Plateau detected** — last 5 experiments all within 1% of leaderboard best
- **All hypotheses tested** — nothing left in the hypothesis queue
- **User says stop**

Before closing: write a final summary in analysis/, update learnings.md as the campaign's lasting contribution, and ask user to review.
