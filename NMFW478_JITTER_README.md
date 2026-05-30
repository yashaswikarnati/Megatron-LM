# NMFW-478 Iter-Jitter Experiments — Run Guide

## Use the pre-synced workspace on cw-dfw (no clone needed)

```
ssh cw-dfw-cs-001-vscode-01
cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/ykarnati/cog-scratch/workspaces/megatron_lm/c7a6bc8d421a1a19/repo
```

This is a synced snapshot of branch `ykarnati/nmfw-464-boundary-mb-experiments`
(commit `aa3e6c12f7`) at https://github.com/yashaswikarnati/Megatron-LM.
Everything is in place — sbatch directly from here.

## Launch scripts (cw-dfw)

| scale | script |
|---|---|
| 9n  | `examples/mimo/scripts/cw-dfw/sbatch_hetero_parity_gbs192_9n_cw.sh` |
| 34n | `examples/mimo/scripts/cw-dfw/sbatch_hetero_jitter_gbs768_34n_cw.sh` |

Both go to `coreai_dlalgo_llm` partition (use `-A coreai_dlalgo_llm` for better fair-share).

## Baseline launch — best-known config

From the synced repo path:

```
NUM_WORKERS=4 \
HETERO_DISABLE_AUTO_GC=1 \
HETERO_GC_INTERVAL=30 \
DDP_NUM_BUCKETS=4 \
NCCL_PROTO=LL128 \
sbatch -A coreai_dlalgo_llm \
  --export=ALL,NUM_WORKERS=4,HETERO_DISABLE_AUTO_GC=1,HETERO_GC_INTERVAL=30,DDP_NUM_BUCKETS=4,NCCL_PROTO=LL128 \
  examples/mimo/scripts/cw-dfw/sbatch_hetero_jitter_gbs768_34n_cw.sh
```

Replace the script path with the 9n one for 9-node runs. 9n defaults to `LLM_TP=2, LLM_DP=32, LLM_EP=8` matching 34n's MoE topology.

## Diagnostic-probe launch (per-mb timing breakdown)

Add three env vars on top of baseline:

```
MIMO_FWD_SYNC_PROBE=1     # cuda.synchronize() before every mb fwd
MIMO_BWD_SYNC_PROBE=1     # cuda.synchronize() around every mb's autograd.backward
MIMO_TIMELINE_CUDA_EVENTS=1  # record cuda_ms (GPU stream time) on timeline events
```

These add per-mb sync events to the timeline. Adds ~5% overhead but enables full per-mb fwd/bwd attribution. **Use for analysis, off for prod.**

## Env knobs cheat-sheet

| env | default | what it does |
|---|---|---|
| `HETERO_DISABLE_AUTO_GC` | 0 | =1 disables Python generational GC during train loop |
| `HETERO_GC_INTERVAL` | 0 | =N explicit `gc.collect()` every N iters |
| `NUM_WORKERS` | 2 | energon data-loader workers per rank |
| `PACKING_BUFFER_SIZE` | 4 | energon packing-buffer size |
| `DDP_NUM_BUCKETS` | 8 (34n), 8 (9n) | DDP grad bucket count |
| `OVERLAP_PARAM_GATHER` | 1 | =0 disables overlap-param-gather |
| `OVERLAP_GRAD_REDUCE` | 1 | =0 disables overlap-grad-reduce |
| `NUM_DIST_OPT_INSTANCES` | 1 | distributed-optimizer instance count |
| `NCCL_PROTO` | LL128 (34n) / LL128 (9n) | NCCL protocol |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 1 | kernel-launch queue depth |
| `DATASET_PROVIDER` | energon_multimodal | =mock skips data loader |
| `MIMO_VISION_INPUT_MODE` | (auto) | mock-data only: pixels vs hidden_states |
| `MIMO_FWD_SYNC_PROBE` | 0 | =1 adds `mb_pre_sync.mbK` events |
| `MIMO_BWD_SYNC_PROBE` | 0 | =1 adds `autograd.bwd_pre_sync/post_sync.mbK` events |
| `MIMO_TIMELINE_CUDA_EVENTS` | 0 | =1 records GPU stream time on cuda-tagged events |
| `LLM_EP` (9n only) | 8 | LLM expert parallel size |
| `TRITON_CACHE_DIR_BASE` | per-job (`$RUN_DIR/triton-cache`) | set to a persistent path to share Triton cache across runs |
| `TRITON_PRINT_AUTOTUNING` | unset | =1 prints autotune events to stdout |

RUN_NAME embeds all knob values for self-identification (e.g. `mimo-jitter-gbs768-34n-cw-PG1-GR1-NDOI1-B4-NAN1-LL128-NVLS0-CDMC1-GC1-GCI30-DSmultimodal-W4-PB4`).

## In-code sync / barrier instrumentation

In `megatron/core/pipeline_parallel/schedules.py:forward_backward_pipelining_without_interleaving`:

- **Per-mb pre-fwd**: gated by `MIMO_FWD_SYNC_PROBE=1`
  - `mb_pre_barrier.mbK` → `dist.barrier(group=pg_collection.tp_dp_cp)` — **currently no-ops** for hetero (pg_collection is `MultiModuleProcessGroupCollection`, doesn't expose `tp_dp_cp` directly; needs a fix to access via inner module's collection)
  - `mb_pre_sync.mbK` → `torch.cuda.synchronize()` — **fires correctly**

- **Per-mb backward (in `backward_step_multimodule`)**: gated by `MIMO_BWD_SYNC_PROBE=1`
  - `autograd.bwd_pre_sync.mbK` → `torch.cuda.synchronize()` before `torch.autograd.backward()`
  - `autograd.bwd_post_sync.mbK` → `torch.cuda.synchronize()` after `torch.autograd.backward()`

Default is OFF so production iter time is unaffected. To disable: unset the env vars.

## Timeline output

Per-rank JSONL at:
```
${RUN_DIR}/timeline/rank<00000..00271>.jsonl
```
where `RUN_DIR = /lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/<RUN_NAME>/<SLURM_JOB_ID>`.

Each line is one event:
```json
{"event":"schedule.forward","iteration":42,"rank":16,"microbatch":2,
 "start_time_ns":1780...,"duration_us":167.5,"cuda_ms":167.4,"module":"llm",...}
```

Standard log lives at:
```
/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-{jitter,boundary}-*-<JOB>.{out,err}
```

stdout includes (since the param-count commit): `[rank N] local_params total=… trainable=…` and `GLOBAL params: total_sum_across_ranks=…` for sanity-checking sharding.

## Analysis hooks (existing scripts)

- `experiments/analyze.py` — single-rank per-mb fwd/bwd distribution + 8-slowest stall-budget attribution
- `iter_jitter_attribution.html` — pre-built per-iter waterfall (from W=4 baseline)
- `skew_viz.html` — per-rank Gantt across 256 LLM ranks for slow iters

Latest representative baseline run for comparison:
- **W=4 baseline at 34n**: job `12295701`, p50=3934 ms, p99=4864 ms, stdev=348 ms
- All probes on: `MIMO_FWD_SYNC_PROBE=1 MIMO_BWD_SYNC_PROBE=1 MIMO_TIMELINE_CUDA_EVENTS=1`
