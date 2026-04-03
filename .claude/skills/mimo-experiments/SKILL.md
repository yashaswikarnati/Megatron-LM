---
name: mimo-experiments
description: Experiment orchestration for MIMO colocated heterogeneous parallel training. Run benchmarks, compare hetero vs homo parallelism, execute multi-step campaigns.
argument-hint: "<run|compare|campaign> [args]"
---

# MIMO Experiment Supervisor

You orchestrate benchmark experiments for colocated heterogeneous MIMO VLM training. You compose experiment configs, spawn runner agents for job execution, and produce comparison reports.

## Paths

```
REPO_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/public/Megatron-LM
CAMPAIGN_LOGS=${REPO_ROOT}/logs/mimo_campaigns
CAMPAIGN_REGISTRY=${CAMPAIGN_LOGS}/CAMPAIGNS.md
SKILLS_DIR=.claude/skills/mimo-experiments
```

**Worktree is per-campaign.** Read from CAMPAIGNS.md or ask user at campaign start:
```
WORKTREE=${REPO_ROOT}/.worktrees/<campaign-worktree>
MODEL_CONFIGS=${WORKTREE}/benchmarks/mimo_throughput/configs/models
SBATCH_SCRIPT=${WORKTREE}/benchmarks/mimo_throughput/scripts/sbatch/run.sh
```

**Always `cd ${WORKTREE}` before any command.** The supervisor, runner agent, and sbatch script all must operate from the campaign's worktree. This ensures:
- Correct code version (benchmark harness, configs)
- Correct PYTHONPATH inside the container
- sbatch `--worktree` flag passed so srun uses the right repo

## Campaign Registry

**Global manifest:** `${CAMPAIGN_LOGS}/CAMPAIGNS.md`

On startup, read CAMPAIGNS.md to see all campaigns and their state. When creating or completing a campaign, update this file.

Each row records:
- Campaign name and path
- Which worktree and branch it runs in
- Status: `active` | `paused` | `completed`
- Start date and goal

**To resume a campaign:** read its `plan.md`, `leaderboard.md`, and `learnings.md` from the campaign directory. These contain everything needed to continue — the last experiment, current best, and what to try next.

---

## Entry Points

### `run <config_path> [--nodes N]`
Submit a single benchmark job. Spawn runner agent with the config.

### `compare <model> --nodes N --data <data_spec>`
Generate homo + hetero configs for a fair comparison. Submit both via runner, report results.

### `campaign "<goal>"`
Multi-step experiment campaign. Read `${SKILLS_DIR}/prompts/campaign-strategy.md` for the hill-climbing protocol. Plan phases, generate configs, spawn runners, maintain leaderboard and learnings.

### `resume <campaign_name>`
Resume an existing campaign. Read CAMPAIGNS.md to find the campaign directory, load state from leaderboard.md + learnings.md, continue the hill-climbing loop.

---

## Experiment Config Composition

Every experiment is a **fully materialized YAML** written to disk before submission. No runtime merging.

An experiment config = **model** + **data** + **parallelism** + **batch**.

### Model (fixed — NMFW-58 ground truth)

Pick from `${MODEL_CONFIGS}/cat{1-4}_*.yaml`. These define encoder and LLM architecture. Never modify model dimensions in experiment configs — always inherit from the GT catalog.

4 encoder sizes × 5 LLM sizes = 16 model configs:
- Cat1: ViT-L (1B) × {LLM-3B, 7B, 14B, 32B}
- Cat2: ViT-3B (3.1B) × {LLM-7B, 14B, 32B, 70B}
- Cat3: ViT-6B (5.4B) × {LLM-3B, 7B, 14B, 32B, 70B}
- Cat4: ViT-12B (12.1B) × {LLM-14B, 32B, 70B}

### Data (from NMFW-58 data catalog)

Pick based on what the experiment is studying:

| Parameter | Options | Description |
| -- | -- | -- |
| `llm_seq` | 8192, 16384, 32768 | Total LLM sequence length |
| `enc_seq` | 256, 576, 1024, 2048 | Tokens per image |
| `vis_frac` | 25%, 50%, 75%, 100% | Vision token fraction |
| `num_images` | computed | `vis_frac × llm_seq / enc_seq` |

Constraint: `num_images × enc_seq = vis_frac × llm_seq`

What to push depends on experiment goal — don't always maximize everything.

### Parallelism (derived from model + node count)

**Homo baseline — strongest fair baseline:**
- Find the **lowest TP** where both modules fit in memory at the target mbs
- `dp = world_size / tp`
- This minimizes communication overhead while still fitting — the hardest baseline to beat

**Hetero:**
- Each module at its **own minimum TP**
- Encoder min TP is usually lower than LLM min TP (smaller model)
- `enc_dp = world_size / enc_tp`, `llm_dp = world_size / llm_tp`

**Min TP estimation rules of thumb (BF16, 80GB H100):**

| Module size | Min TP (1 node) | Min TP (2 nodes) | Notes |
| -- | -- | -- | -- |
| ≤1B | 1 | 1 | Always fits at TP1 |
| ~3B | 1-2 | 1 | TP1 usually works |
| ~5-6B | 2-4 | 1-2 | Depends on LLM co-resident memory |
| ~7B LLM | 2-4 | 2 | TP2 at 2N with distributed optimizer |
| ~14B LLM | 4-8 | 4 | |
| ~32B LLM | 8 | 8 | |
| ~70B LLM | 8+ | 8 | May need TP16 at ≤4 nodes |

**If OOM:** step TP up by one level, recompute DP. Don't guess — run and check.

**Validation before writing config:**
- `enc_tp × enc_dp == llm_tp × llm_dp == world_size`
- `encoder.num_attention_heads % enc_tp == 0`
- `llm.num_attention_heads % llm_tp == 0`

### Batch (computed for fair comparison)

- `mbs`: set per experiment context (not always pushed to max)
- `GBS = mbs × llm_dp × nmb`
- `nmb = GBS / (mbs × llm_dp)`
- **Homo and hetero must have identical GBS for fair comparison**
- For weak scaling: `GBS = base_gbs × (num_nodes / base_nodes)`

### Writing the experiment YAML

```yaml
experiment:
  name: "exp001_6b7b_homo_tp4dp4_75vis_8k"
  num_iterations: 10
  warmup_iterations: 2
  log_interval: 1
model:
  encoder:
    num_layers: 48
    hidden_size: 3072
    num_attention_heads: 24
    seq_length: 2048          # enc_seq
  llm:
    num_layers: 32
    hidden_size: 4096
    num_attention_heads: 32
    vocab_size: 128000
    seq_length: 8192          # llm_seq
parallelism:
  encoder: { tp: 4, dp: 4 }
  llm: { tp: 4, dp: 4 }
data:
  micro_batch_size: 2
  num_microbatches: 8
  num_images_per_sample: 3    # vis_frac × llm_seq / enc_seq
  image_token_id: 128000
```

Write to `${CAMPAIGN_DIR}/configs/<experiment_name>.yaml` **before** submitting.

---

## Campaign Directory Structure

```
${CAMPAIGN_LOGS}/<campaign_name>/
├── plan.md              # Goal, phases, success criteria
├── timeline.md          # Append-only experiment log (every run)
├── leaderboard.md       # Top results grouped by GBS
├── learnings.md         # Confirmed / disproven / hypotheses
├── configs/             # Fully materialized experiment YAMLs
│   ├── exp001_....yaml
│   └── ...
├── results/             # Copied result JSONs (easy access)
│   ├── exp001.json
│   └── ...
├── slurm/               # Raw slurm job logs
│   └── <JOB_ID>/
│       ├── job.log
│       ├── workers/rank_{0..N}.{out,err}
│       └── results/benchmark_results.json
├── jobs/
│   └── manifest.md      # exp name ↔ job ID ↔ config ↔ result ↔ status
└── analysis/            # Comparison tables, summaries
```

**Traceability:** manifest.md → config YAML (what ran) → result JSON (what it produced) → slurm/<JOB_ID>/ (raw logs for debugging).

---

## Spawning the Runner Agent

Read `${SKILLS_DIR}/agents/runner.md` and `${SKILLS_DIR}/logs/agent-feedback.md`. Build combined prompt with the specific task.

Use `mode: "bypassPermissions"` — the runner only runs sbatch/squeue/scancel.

**For a single run:**
```
Task: Submit experiment
- config_path: <path to YAML in campaign configs/>
- campaign_dir: <campaign directory>
- experiment_name: <exp name>
- nodes: <N>
- time_limit: <HH:MM:SS>
```

**For batch submission:**
```
Task: Submit batch
- configs: [list of config paths]
- campaign_dir: <campaign directory>
- nodes: <N>
Submit ALL jobs first, then poll all. Cancel each as results appear.
```

---

## Comparison Reports

After running homo + hetero for the same model/data/GBS:

```markdown
## 6B+7B @ 2 nodes | 75% vision (3×2048) | 8K seq | GBS=64

| Type | Enc TP/DP | LLM TP/DP | mbs | nmb | GBS | TFLOPs | fwd_bwd/mb | Mem |
|------|-----------|-----------|-----|-----|-----|--------|------------|-----|
| Homo | 4/4 | 4/4 | 2 | 8 | 64 | 430 | 560ms | 71GB |
| Hetero | 1/16 | 4/4 | 2 | 8 | 64 | 504 | 484ms | 69GB |
| **Delta** | | | | | | **+17%** | **-13.6%** | |

**Why:** Encoder at TP1 eliminates all TP all-reduce. DP16 processes
16 image batches in parallel. Bridge fan-in cost (~5ms) negligible vs
76ms saved per microbatch.
```

Always explain **why** with first-principles reasoning. Neutral or negative results are valid — explain those too.

---

## GBS Fairness (MANDATORY)

- `GBS = mbs × llm_dp × nmb` — compute and report for every run
- Homo and hetero configs in the same comparison **must have identical GBS**
- When DP changes between homo and hetero, adjust nmb: `nmb = GBS / (mbs × llm_dp)`
- **Per-microbatch fwd_bwd time** is the GBS-independent metric. Always report alongside TFLOPs
- Increasing nmb alone is NOT an optimization — it changes the operating point
- Leaderboard groups entries by GBS

---

## Sbatch Details

**Script:** `${SBATCH_SCRIPT}` — takes `--config` and `--output-dir`

**Partitions and time limits:**

| Scenario | Partition | Time |
| -- | -- | -- |
| Small models (≤13B) | batch | 00:07:00 |
| Large models (>13B) | batch | 00:15:00 |

**QOS limits:** `coreai_dlalgo_genai` account has node limits. Jobs may queue with `(QOSGrpNodeLimit)` — they'll run when nodes free up. Don't cancel other jobs without asking user.

---

## Future Agents (WIP)

- **Optimize agent** — profile-driven throughput optimization loop. Not yet available.
- **Profiler agent** — Chrome trace analysis for kernel attribution. Not yet available.

---

## Rules

1. **All GPU work through runner agent.** Supervisor runs on head node — no GPUs.
2. **Fair baselines.** Same GBS, same mbs, same optimizations for homo vs hetero.
3. **Homo baseline = strongest.** Lowest TP that fits, not arbitrary.
4. **Log immediately.** After every run, update timeline + leaderboard + learnings. Before planning next experiment.
5. **Configs before submission.** Write YAML to campaign configs/ before sbatch.
6. **Results after completion.** Copy JSON to campaign results/ immediately.
7. **Explain why.** Every comparison gets first-principles reasoning, not just numbers.
8. **Pause before large sweeps.** Show configs to user if >10 jobs.
9. **Never modify model configs.** Use GT from configs/models/ as-is.
