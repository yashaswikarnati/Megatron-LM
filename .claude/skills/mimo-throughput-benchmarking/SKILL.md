---
name: mimo-throughput-benchmarking
description: Autonomous performance optimization and benchmarking for colocated heterogeneous MIMO training. Profile-driven optimization loop that improves TFLOPs/GPU and tokens/sec.
argument-hint: "<goal1|goal2|profile|sweep> [config-path]"
---

# MIMO Throughput Benchmarking & Optimization

You are an autonomous performance optimization agent for colocated heterogeneous parallel MIMO training (vision encoder + LLM with different TP/DP on same GPUs).

## Working Directory

All work happens in the nmfw-23 worktree:
```
cd /workspace/Megatron-LM/.worktrees/nmfw-23-benchmarking
```

## Modes

Parse `$ARGUMENTS` to determine the mode:

### Mode: `goal1 [config-path]` — Performance Optimization Loop

**Objective:** For a fixed model config, iteratively profile and optimize e2e throughput. Push TFLOPs/GPU and tokens/sec higher. Focus on the colocated code path and communicator.

**Step 1: Read state**
```
Read: benchmarks/mimo_throughput/experiments/EXPERIMENT_LOG.md
Read: benchmarks/mimo_throughput/experiments/BOTTLENECKS.md
Read: benchmarks/mimo_throughput/experiments/CURRENT_BEST.json (if exists)
```

**Step 2: If no CURRENT_BEST.json, establish baseline**
- Run the config with the harness:
  ```bash
  uv run python -m torch.distributed.run --nproc_per_node=8 \
    -m benchmarks.mimo_throughput.runner --config <config-path>
  ```
- Run with profiler:
  ```bash
  uv run python -m torch.distributed.run --nproc_per_node=8 \
    -m benchmarks.mimo_throughput.runner --config <config-path> \
    --profile --profile-steps 5-7 --results-dir benchmarks/mimo_throughput/experiments
  ```
- Read the profiler output text file from experiments/ directory
- Write CURRENT_BEST.json with baseline metrics
- Write initial BOTTLENECKS.md analysis from profiler data
- Log Experiment 1 in EXPERIMENT_LOG.md

**Step 3: Analyze profiler output**
Read the profiler text file (key_averages table sorted by cuda_time_total). Identify:
- Top 5 CUDA time consumers (kernels, collectives, operators)
- What fraction is compute (GEMM, attention) vs communication (all_reduce, all_gather, reduce_scatter)
- What fraction is in colocated code path (bridge communicator, encoder fwd, LLM fwd)
- Any suspicious overhead (elementwise ops, memory copies, sync points)

**Step 4: Propose optimization**
Based on profiling data, propose ONE targeted change. Prioritize:

**PRIMARY — Colocated code path optimizations (what we control):**
- Overlap text embedding computation with encoder forward (restructure `_forward_all_modules`)
- Async/non-blocking all-gather in bridge communicator (CUDA streams)
- Encoder output recomputation or offload to free memory for larger batch
- Encoder DDP gradient overlap with optimizer step
- Communication-computation overlap in backward

**When the orchestrator determines memory blocks larger batch sizes**, spawn a research agent to study how Megatron enables `recompute_modules` and `fine_grained_activation_offload` in standard training, then design how to wire these into the MIMO encoder path.

**SECONDARY — Config tuning (when code path is not the bottleneck):**
- Batch size scaling (higher mbs → better arithmetic intensity)
- TP/DP rebalancing based on per-module profiling
- Seq length tuning (vision vs text ratio)
- Kernel/TE settings

**Step 5: Implement the change**
Make the code or config modification. Keep changes minimal and reversible.

**Step 6: Re-benchmark**
Run the modified config. Compare TFLOPs/GPU and tokens/sec to CURRENT_BEST.

**Step 7: Keep or revert**
- If TFLOPs improved >0.5%: KEEP. Update CURRENT_BEST.json.
- If TFLOPs regressed >0.5%: REVERT (git checkout the changed files).
- If neutral but saves memory: KEEP (enables future larger batch).

**Step 8: Update experiment log**
Append to EXPERIMENT_LOG.md with this format:
```markdown
### Experiment N: <descriptive name> (<date>)
**Config:** <model> | <parallelism> | <batch>
**Change:** What was different and WHY (based on profiling data)
**Profiling insight:** What the profiler showed that motivated this change
**Result:** TFLOPs/GPU, tokens/sec, memory, fwd_bwd_ms, opt_step_ms
**vs Baseline:** +X% / -X% TFLOPs
**vs Previous Best:** +X% / -X% TFLOPs
**Verdict:** KEEP / REVERT
**Learning:** What we learned about the system's behavior
**Next hypothesis:** What to try next based on this result
```

**Step 9: Update BOTTLENECKS.md**
Revise the bottleneck analysis based on the new profiling data. Update "what to try next".

**Step 10: Report**
Print summary: current best metrics, what was tried, what's next. Tell the user to invoke `/mimo-throughput-benchmarking goal1` again to continue the loop.

---

### Mode: `goal2 [configs-dir]` — Ablation Sweep

**Objective:** Sweep across parallelism configs, batch sizes, seq lengths to show where heterogeneous beats homogeneous.

1. Read CURRENT_BEST.json for the optimized code path
2. If configs-dir provided, run all configs and produce comparison CSV
3. If no configs-dir, generate sweep configs from baseline:
   - Homogeneous: TP2/DP4, TP4/DP2, TP8/DP1
   - Heterogeneous: enc_TP1/DP8+llm_TP4/DP2, enc_TP2/DP4+llm_TP4/DP2, enc_TP1/DP8+llm_TP8/DP1
4. Run each config, collect metrics
5. Produce comparison table showing:
   - TFLOPs/GPU for each config
   - % improvement of best heterogeneous over best homogeneous
   - Memory usage comparison
6. Identify sweet spots: model/seq/batch regimes where heterogeneous wins

---

### Mode: `profile [config-path]` — Single Profiling Run

1. Run config with phase timers (always on)
2. Run with torch.profiler for iterations 5-7
3. Read profiler output, produce detailed analysis
4. Update BOTTLENECKS.md with findings

---

### Mode: `sweep [configs-dir]` — Run All Configs

Run all YAML configs in the directory, produce aggregated CSV.
```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m benchmarks.mimo_throughput.runner \
  --configs-dir <configs-dir> --results-dir benchmarks/mimo_throughput/experiments
```

---

## Key Files

**Harness:**
- `benchmarks/mimo_throughput/training.py` — model creation, forward_step, run_benchmark
- `benchmarks/mimo_throughput/runner.py` — CLI with --profile support
- `benchmarks/mimo_throughput/metrics.py` — TFLOPs computation, phase timing
- `benchmarks/mimo_throughput/configs/1node_1b_7b/` — 1B ViT + 7B LLM configs

**Colocated code path (primary optimization targets):**
- `megatron/core/models/mimo/model/base.py` — `_forward_all_modules()`, `_forward_colocated()`
- `megatron/core/models/mimo/comm/colocated_communicator.py` — bridge communicator
- `megatron/core/models/mimo/colocated_schedule.py` — PP>1 three-phase schedule

**State:**
- `benchmarks/mimo_throughput/experiments/EXPERIMENT_LOG.md`
- `benchmarks/mimo_throughput/experiments/BOTTLENECKS.md`
- `benchmarks/mimo_throughput/experiments/CURRENT_BEST.json`

## Run Commands

```bash
# Benchmark
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m benchmarks.mimo_throughput.runner --config <yaml>

# Benchmark with profiler
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m benchmarks.mimo_throughput.runner --config <yaml> \
  --profile --profile-steps 5-7 --results-dir <dir>

# Sweep
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m benchmarks.mimo_throughput.runner --configs-dir <dir>
```

## Important Notes

- Always use `uv run python -m torch.distributed.run` (not torchrun)
- Capture per-rank logs for debugging: add `--redirects 3 --log-dir /tmp/bench_logs`
- Use 60s timeout for single distributed tests
- After code changes, always re-run the benchmark to verify no regression
- Document EVERY experiment in the log, including failed approaches (prevents repeating dead ends)
