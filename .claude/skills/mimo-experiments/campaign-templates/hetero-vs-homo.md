# Campaign Template: Hetero vs Homo Throughput Optimization

Push heterogeneous and homogeneous parallelism to their limits, then compare fairly.

---

## Parameters (from user)

| Param | Description | Default |
| -- | -- | -- |
| **category** | Model category from NMFW-58: cat1/cat2/cat3/cat4 or specific model | required |
| **nodes** | Node count: 1, 2, 4, 8, 16 | required |
| **vision** | Vision fractions to sweep | 25, 50, 75, 100% |
| **seq** | LLM sequence lengths | 8K, 16K |
| **gbs** | Global batch size | weak-scaled from 32 at 1N |
| **scaling** | Weak or strong | weak |

---

## Scope

- **Parallelism:** TP/DP only. No PP, CP, EP.
- **Precision:** BF16 only. No FP8/FP16.
- **Everything else is tunable:** optimizer instances, non-distributed optimizer, SP, tp_comm_overlap, TE fusions, DDP bucket size, recompute (encoder/LLM/projection/combined embeddings), offload, mbs — whatever the systems expert can think of.

---

## Phase 1: Establish Baselines

**For each model in the category, at each vision fraction (25/50/75/100%) at 8K seq:**

1. **Campaign manager → systems expert:** "For [model] at [N] nodes, [X]% vision, what's the strongest homo config and best hetero starting point?"
2. Systems expert reasons from first principles:
   - Homo: lowest TP where both modules fit at target mbs. This is the strongest — minimum comm overhead.
   - Hetero: each module at its own min TP. Encoder as low as memory allows.
   - Specific mbs, nmb for the target GBS.
3. Campaign manager generates configs, sends to runner.
4. Log results. Record: parallelism, all optimizations, data config, TFLOPs, fwd_bwd/mb, memory.

**Do NOT brute-force TP/DP combos.** Systems expert gives 1-2 configs per type. If OOM or unexpected, consult systems expert with the data.

**Output:** Baseline comparison table per vision fraction.

---

## Phase 2: 16K Sequence Length

Repeat Phase 1 at 16K LLM seq. Some configs will OOM — that's expected. Record which ones and why.

**Output:** Comparison table at 16K. Note which configs are memory-limited.

---

## Phase 3: Hill-Climb Hetero

Take the best hetero config from Phase 1/2. Push throughput as high as possible.

**The loop:**
1. Campaign manager sends current best + all data to systems expert
2. Systems expert analyzes bottleneck and suggests ONE knob to turn:
   - Why this knob? (first-principles reasoning)
   - Expected impact? (quantified)
   - What to watch for? (OOM risk, diminishing returns)
3. Campaign manager generates config, runs it
4. Compare vs previous best. Log.
5. Repeat.

**Knobs the systems expert should consider** (not an exhaustive list — think from first principles):

| Category | Examples |
| -- | -- |
| Parallelism | Enc TP/DP split, push enc to TP1 if possible |
| Optimizer | `num_distributed_optimizer_instances`, non-distributed for small enc, param gather size |
| Communication | `tp_comm_overlap`, sequence parallel (LLM), DDP `bucket_size` tuning |
| TE fusions | `bias_dropout_fusion`, `bias_activation_fusion`, `gradient_accumulation_fusion`, `cross_entropy_loss_fusion` — verify all on, test toggling |
| Recompute | Encoder: selective/full, specific layers/modules (attention, mlp), combined_embeddings |
| Recompute | LLM: selective/full, specific layers/modules |
| Offload | Encoder activations, combined_embeddings offload |
| Offload | LLM activations (if memory-bound) |
| Data | mbs tuning (higher = better arithmetic intensity, if memory allows) |
| Data | Different num_images × enc_seq combos for same vision % |

**Stop when:** 3 consecutive experiments with <1% improvement = plateau.

**Output:** Optimized hetero config + full optimization stack documented.

---

## Phase 4: Hill-Climb Homo (Fair Strong Baseline)

**Same treatment for the best homo config.** This is critical — the comparison is only meaningful if homo is equally optimized.

Apply the same knob exploration:
- Optimizer tuning
- Recompute/offload
- Communication overlaps
- mbs tuning
- Everything that helped hetero — try it on homo too

**Stop when:** 3 consecutive experiments with <1% improvement.

**Output:** Optimized homo config + full optimization stack documented.

---

## Phase 5: Final Comparison

Best optimized hetero vs best optimized homo. Per vision fraction, per seq length.

**Report format:**

```markdown
## [Model] @ [N] nodes — Final Results

### Per vision fraction (8K seq)

| Vis % | Homo Config | Homo TFLOPs | Hetero Config | Hetero TFLOPs | Delta | Per-mb speedup |
| -- | -- | -- | -- | -- | -- | -- |
| 25% | TP4/DP4, SP, recomp=sel | 430 | enc TP1/DP16, llm TP4/DP4, SP, inst=2 | 450 | +4.7% | -4.5% |
| 50% | ... | ... | ... | ... | ... | ... |
| 75% | ... | ... | ... | ... | ... | ... |
| 100% | ... | ... | ... | ... | ... | ... |

### 16K seq (if applicable)

| Vis % | Homo | Hetero | Delta |
| ... |

### Optimization stack comparison

| Optimization | Homo | Hetero | Impact |
| -- | -- | -- | -- |
| SP (LLM) | on | on | +1.3% both |
| tp_comm_overlap | on | on | +1.5% both |
| enc recompute | selective | none | homo needed it for memory |
| dist_opt instances | 1 | 2 | hetero +3% (enc DP=16 cross-node) |
| ... | | | |

### Why hetero wins/loses

<first-principles explanation per vision fraction — from systems expert>
```

---

## Campaign Directory Structure

When instantiated, creates:

```
${CAMPAIGN_LOGS}/<campaign_name>/
├── plan.md              # Filled from this template + user params
├── timeline.md          # Experiment log (append-only)
├── leaderboard.md       # Top results by GBS + optimization columns
├── learnings.md         # Confirmed / disproven / hypotheses
├── configs/             # All experiment YAMLs
├── results/             # Result JSONs
├── slurm/               # Raw job logs
├── jobs/manifest.md     # Experiment ↔ job mapping
└── analysis/            # Final comparison tables
```

---

## Team Roles in This Template

| Role | What they do |
| -- | -- |
| **Team lead** | Instantiates campaign, reviews phase transitions, redirects if stuck |
| **Campaign manager** | Runs the loop: consult SE → generate config → send to runner → log → repeat |
| **Systems expert** | Advises parallelism, analyzes results, suggests knobs (does NOT run experiments) |
| **Experiment runner** | Submits sbatch, polls, collects results (does NOT decide what to run) |

**Key communication pattern per iteration:**
```
Campaign Manager → Systems Expert: "Here's the data, what next?"
Systems Expert → Campaign Manager: "Try X because Y, expect Z"
Campaign Manager → Experiment Runner: "Submit this config"
Experiment Runner → Campaign Manager: "Result: ..."
Campaign Manager: logs, updates leaderboard, back to top
```

---

## Success Criteria

Defined by user at campaign start. Examples:
- "Hetero beats homo by ≥5% at 75%+ vision on at least 3 models"
- "Map the crossover point where hetero starts winning per model"
- "Reach X TFLOPs/GPU on model Y"

Record in plan.md and measure every experiment against it.
