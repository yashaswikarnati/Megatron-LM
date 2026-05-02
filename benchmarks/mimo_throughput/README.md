# MIMO Throughput Benchmark Harness

This harness benchmarks proxy multi-image, multi-modal training throughput with
separate vision encoder and LLM modules. It supports three placement modes:

- `colocated`: encoder and LLM use the same global rank set.
- `non_colocated`: encoder and LLM use disjoint rank ranges.
- `homo`: homogeneous baseline where the encoder shares the LLM TP/DP/CP layout
  and lives on LLM pipeline stage 0.

Use this harness to compare placement strategy, module parallelism, memory
settings, and image-token pressure while keeping model shape and global batch
constant.

## Entry Points

Single node:

```bash
NPROC=8 benchmarks/mimo_throughput/scripts/run.sh \
    --config benchmarks/mimo_throughput/configs/campaigns/cat1_1b3b_noncolocated_enc_tp2dp2_llm_tp2pp2dp1_8g_mbs1_gbs4_4img_1N_sp_smoke.yaml \
    --results-dir /tmp/mimo_results
```

Direct `torchrun` equivalent:

```bash
uv run python -m torch.distributed.run \
    --nproc_per_node=8 \
    -m benchmarks.mimo_throughput.runner \
    --config benchmarks/mimo_throughput/configs/campaigns/cat1_1b3b_noncolocated_enc_tp2dp2_llm_tp2pp2dp1_8g_mbs1_gbs4_4img_1N_sp_smoke.yaml \
    --results-dir /tmp/mimo_results
```

Multi-node Slurm:

```bash
sbatch --nodes=9 --time=01:00:00 \
    benchmarks/mimo_throughput/scripts/sbatch/run.sh \
    --worktree /path/to/Megatron-LM \
    --config benchmarks/mimo_throughput/configs/campaigns/cat3_6b70b_noncolocated_enc_tp1dp8_8g_llm_tp4pp4dp4_64g_mbs1_gbs288_4img_9N_f14.yaml \
    --output-dir /path/to/mimo_runs
```

Run all YAMLs in a directory:

```bash
NPROC=8 benchmarks/mimo_throughput/scripts/run.sh \
    --configs-dir benchmarks/mimo_throughput/configs/campaigns \
    --results-dir /tmp/mimo_sweep
```

The runner writes one JSON file per experiment. Directory sweeps also write
`summary.csv`.

## Config Anatomy

Each config can be fully materialized or can inherit defaults from a
`baseline.yaml` file in the same directory. The examples below show the full
shape:

```yaml
experiment:
  name: example_name
  num_iterations: 10
  warmup_iterations: 2
  log_interval: 1

model:
  encoder:
    num_layers: 48
    hidden_size: 2304
    num_attention_heads: 32
    seq_length: 1024
  llm:
    num_layers: 80
    hidden_size: 8192
    num_attention_heads: 64
    vocab_size: 128000
    seq_length: 8192

parallelism:
  encoder:
    tp: 1
    dp: 8
    pp: 1
    offset: 0
  llm:
    tp: 4
    dp: 4
    pp: 4
    offset: 8

data:
  micro_batch_size: 1
  num_microbatches: 72
  num_images_per_sample: 4
  image_token_id: 128000

memory:
  encoder:
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 48

pp_mode: non_colocated
sequence_parallel: true
```

Important derived values:

- `encoder.world_size = encoder.tp * encoder.dp * encoder.pp * encoder.cp`
- `llm.world_size = llm.tp * llm.dp * llm.pp * llm.cp`
- `global_batch_size = micro_batch_size * llm.dp * num_microbatches`
- `num_images_per_sample * encoder.seq_length` must fit inside `llm.seq_length`

The encoder must always use `pp: 1`. `cp` defaults to `1` when omitted.
`pp_mode` defaults to `colocated` when omitted.

## Placement Modes

### Colocated

Use `pp_mode: colocated` when encoder and LLM share the same global rank set.
This is useful for comparing heterogeneous module TP/DP on the same GPUs.

Rules:

- Encoder and LLM world sizes must match.
- Rank offsets should be omitted or set to `0`.
- Encoder CP must be `1`.
- `encoder_offload: true` is only valid for colocated mode with LLM `pp > 1`.

Tracked example:

```bash
benchmarks/mimo_throughput/configs/3b70b_8n/colocated_with_offload.yaml
```

Shape:

```yaml
pp_mode: colocated
encoder_offload: true
parallelism:
  encoder: { tp: 8, dp: 8, pp: 1 }
  llm:     { tp: 8, dp: 2, pp: 4 }
```

Both modules consume 64 ranks in that example.

### Non-Colocated

Use `pp_mode: non_colocated` when the encoder and LLM should run on separate
GPU islands. This is the mode for testing whether a small or memory-sensitive
encoder benefits from lower TP and higher DP while the LLM uses a different
parallel layout.

Rules:

- Encoder and LLM rank ranges must be disjoint.
- `encoder.offset + encoder.world_size` and `llm.offset + llm.world_size` must
  fit inside distributed `WORLD_SIZE`.
- `encoder.world_size + llm.world_size` must equal distributed `WORLD_SIZE`.
- Encoder and LLM DP must be divisible in one direction for bridge fan-in or
  fan-out.
- Encoder CP and LLM CP must both be `1`.
- With `sequence_parallel: true` and LLM `tp > 1`, the LLM offset must be
  divisible by LLM TP.

Single-node smoke example:

```bash
benchmarks/mimo_throughput/configs/campaigns/cat1_1b3b_noncolocated_enc_tp2dp2_llm_tp2pp2dp1_8g_mbs1_gbs4_4img_1N_sp_smoke.yaml
```

Larger 9-node example with uneven LLM PP:

```bash
benchmarks/mimo_throughput/configs/campaigns/cat3_6b70b_noncolocated_enc_tp1dp8_8g_llm_tp4pp4dp4_64g_mbs1_gbs288_4img_9N_f14.yaml
```

Shape:

```yaml
pp_mode: non_colocated
sequence_parallel: true
parallelism:
  encoder: { tp: 1, dp: 8, pp: 1, offset: 0 }
  llm:     { tp: 4, dp: 4, pp: 4, offset: 8 }
llm_num_layers_in_first_pipeline_stage: 14
```

That uses ranks `[0, 8)` for the encoder and `[8, 72)` for the LLM.

### Homo

Use `pp_mode: homo` for the baseline where encoder parallelism matches LLM
TP/DP/CP. The encoder lives on LLM PP stage 0, so this mode is useful as a
standard homogeneous comparison against colocated and non-colocated layouts.

Rules:

- Encoder TP must equal LLM TP.
- Encoder DP must equal LLM DP.
- Encoder CP must equal LLM CP.
- Rank offsets should be omitted or set to `0`.
- LLM world size must equal distributed `WORLD_SIZE`.

Example shape:

```yaml
pp_mode: homo
sequence_parallel: true
parallelism:
  encoder: { tp: 4, dp: 6, pp: 1 }
  llm:     { tp: 4, dp: 6, pp: 3 }
data:
  micro_batch_size: 1
  num_microbatches: 48
  num_images_per_sample: 4
```

This consumes `4 * 6 * 3 = 72` ranks. Because the encoder also lives on LLM PP
stage 0, uneven LLM PP can be useful:

```yaml
llm_num_layers_in_first_pipeline_stage: 24
```

## Uneven LLM Pipeline Splits

The harness supports Megatron LLM pipeline split controls:

```yaml
llm_num_layers_in_first_pipeline_stage: 14
llm_num_layers_in_last_pipeline_stage: 22
```

or:

```yaml
llm_pipeline_model_parallel_layout: ...
```

Do not combine `llm_pipeline_model_parallel_layout` with the first/last stage
layer-count fields. If no custom split is provided, `llm.num_layers` must be
evenly divisible by LLM `pp`.

## Fair Comparisons

For colocated vs non-colocated vs homo comparisons:

- Keep `model` identical.
- Keep `data` identical, especially `micro_batch_size`, `num_microbatches`,
  and `num_images_per_sample`.
- Keep memory settings comparable unless the experiment is explicitly testing
  memory features.
- Compare at the same total rank count, unless the point is scaling.
- Check `global_batch_size` in the JSON result before comparing throughput.

Useful result fields:

- `summary.median_tflops_per_gpu`
- `summary.median_tokens_per_sec`
- `summary.median_samples_per_sec`
- `summary.median_elapsed_sec`
- `summary.max_memory_gb`
- `config.placement`
- `config.llm_pipeline_split`

## Profiling Options

Torch profiler:

```bash
benchmarks/mimo_throughput/scripts/run.sh \
    --config <config.yaml> \
    --profile \
    --profile-steps 5-7 \
    --results-dir /tmp/mimo_profile
```

Memory snapshot:

```bash
benchmarks/mimo_throughput/scripts/run.sh \
    --config <config.yaml> \
    --profile-memory \
    --results-dir /tmp/mimo_memory
```

Pipeline timers:

```bash
benchmarks/mimo_throughput/scripts/run.sh \
    --config <config.yaml> \
    --pipeline-timers \
    --results-dir /tmp/mimo_timers
```

Pipeline timers add synchronization and should be used for profiling, not for
final throughput numbers.
