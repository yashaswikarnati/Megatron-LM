# Debug Forward Pass POC

This example runs one synthetic GPT forward pass on a 2-node, 16-GPU Slurm allocation and captures the hidden activation after every transformer layer.

Default topology:

- Model: GPT-style decoder, 12 layers, hidden size 1024, 8 attention heads
- Parallelism: tensor parallel size 8, pipeline parallel size 2, data parallel size 1
- Placement: each pipeline stage owns 6 transformer layers; tensor parallel ranks are assumed to be within each stage
- Forward inputs: deterministic token IDs and position IDs follow the GPT unit-test pattern (`range(seq_length)` repeated across the micro-batch). By default the script passes `attention_mask=None`, matching the pipeline-layout GPT forward tests where the GPT layer spec supplies causal masking.
- Capture policy: TP rank 0 and DP rank 0 on each pipeline stage saves layer outputs. Sequence parallelism is disabled, so these tensors are full hidden states with shape `[sequence, batch, hidden]`.

The main script uses Megatron-Core directly: `TransformerConfig`, `GPTModel`, `get_gpt_layer_local_spec`, `parallel_state.initialize_model_parallel`, and the core non-interleaved pipeline schedule.

## Files

- `debug_forward_pass.py`: builds the model, initializes distributed and model-parallel groups, registers layer hooks, runs one forward pass, gathers activations, prints stats, and saves a `.pt` file on global rank 0.
- `slurm_launch.sh`: self-submitting Slurm launcher for 2 nodes with 8 GPUs per node.
- `inspect_activations.py`: offline inspection, histogram plotting, and two-file comparison.
- `README.md`: this guide.

## Prerequisites

- Megatron-LM dependencies installed in the environment used by Slurm.
- A Slurm partition with 2 nodes and 8 CUDA GPUs per node.
- Working multi-node NCCL setup.
- Run commands from the Megatron-LM repository root.

No dataset or tokenizer is needed. The forward pass uses deterministic dummy token IDs.

## Run With Slurm

Submit through the self-submitting wrapper so partition, account, and time can be normal environment variables:

```bash
PARTITION=batch \
ACCOUNT=my_account \
TIME=00:30:00 \
SAVE_PATH=/path/to/debug_forward_pass_activations.pt \
./examples/debug_forward_pass/slurm_launch.sh
```

Useful overrides:

```bash
TP=8
PP=2
SEQ_LENGTH=64
MICRO_BATCH_SIZE=1
MASTER_PORT=29501
NCCL_SOCKET_IFNAME=ib0
```

Extra arguments are passed to `debug_forward_pass.py`:

```bash
./examples/debug_forward_pass/slurm_launch.sh --print-rank-map --seed 7
```

The `--attention-mask-mode` flag controls how closely the script mirrors the forward tests:

```bash
--attention-mask-mode none       # default; matches pipeline-layout GPT forward tests
--attention-mask-mode test-ones  # matches tests/unit_tests/models/test_gpt_model.py
--attention-mask-mode causal     # explicit upper-triangular causal mask
```

The script launches one `torchrun` per node through `srun`, with `--nproc_per_node=8`, `--nnodes=2`, and `--node_rank=$SLURM_NODEID`.

## Run With Cog On cw-dfw

The local `cog` binary was found at `/opt/hermes/cog`, and the default registered cluster is `cw-dfw`. Cog handles remote workspace sync, container selection, environment setup, and job lifecycle, so this is the preferred cw-dfw entrypoint when available.

Check the resolved profile and cluster:

```bash
PATH=/opt/hermes:$PATH cog profile --repo . --cluster-name cw-dfw --base-image-flavor dev
```

For the exact 2-node, 16-GPU POC, use `cog submit`:

```bash
PATH=/opt/hermes:$PATH cog submit \
  --repo . \
  --cluster-name cw-dfw \
  --run-name debug-forward-pass-2n \
  --nodes 2 \
  --gpus 8 \
  --ntasks-per-node 1 \
  --time 00:30:00 \
  --partition interactive \
  --base-image-flavor dev \
  --command "bash -lc '
    set -euo pipefail
    export MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n 1)
    export MASTER_PORT=\${MASTER_PORT:-29500}
    uv run python -m torch.distributed.run \
      --nnodes=\"\$SLURM_NNODES\" \
      --nproc_per_node=8 \
      --node_rank=\"\$SLURM_NODEID\" \
      --master_addr=\"\$MASTER_ADDR\" \
      --master_port=\"\$MASTER_PORT\" \
      examples/debug_forward_pass/debug_forward_pass.py \
        --tp 8 --pp 2 \
        --save debug_forward_pass_activations.pt \
        --print-rank-map
  '"
```

For iterative debugging, start a persistent session and reuse the same session handle. The installed `cog session start` help exposes `--gpus` but not `--nodes`, so persistent sessions on this build are one-node sessions. That is still useful for fast hook/debug iterations with a one-node topology:

```bash
PATH=/opt/hermes:$PATH cog session start \
  --repo . \
  --cluster-name cw-dfw \
  --session-handle debug-forward-pass-dev \
  --gpus 8 \
  --ntasks-per-node 1 \
  --time 02:00:00 \
  --partition interactive \
  --base-image-flavor dev

PATH=/opt/hermes:$PATH cog session status \
  --cluster-name cw-dfw \
  --session-handle debug-forward-pass-dev

PATH=/opt/hermes:$PATH cog session exec \
  --cluster-name cw-dfw \
  --session-handle debug-forward-pass-dev \
  --command 'uv run python -m torch.distributed.run --nproc_per_node=8 --log-dir "$TORCHRUN_LOG_DIR" examples/debug_forward_pass/debug_forward_pass.py --tp 4 --pp 2 --save debug_forward_pass_1node.pt --interactive' \
  --wait-timeout 3600
```

If your Cog deployment does not attach stdin to `session exec`, omit `--interactive` and inspect the saved `.pt` file with `inspect_activations.py`.

Stop the session when finished:

```bash
PATH=/opt/hermes:$PATH cog session stop \
  --cluster-name cw-dfw \
  --session-handle debug-forward-pass-dev
```

If your `cog session start --help` includes a `--nodes` option, use the same session workflow with `--nodes 2 --gpus 8` and run the debug script with `--tp 8 --pp 2`.

## Run Inside An Existing Allocation

If you already have an allocation, disable self-submission:

```bash
SUBMIT_SELF=0 SAVE_PATH=/path/to/activations.pt ./examples/debug_forward_pass/slurm_launch.sh
```

You can also run the underlying command directly:

```bash
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --gpu-bind=none bash -lc '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="${MASTER_PORT:-29500}" \
    examples/debug_forward_pass/debug_forward_pass.py --tp 8 --pp 2 --save activations.pt
'
```

## Interactive Mode

For `pdb`, use an interactive allocation so rank 0 has a terminal:

```bash
salloc -N 2 --gpus-per-node=8 --ntasks-per-node=1 --time=00:30:00
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun --pty --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --gpu-bind=none bash -lc '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    examples/debug_forward_pass/debug_forward_pass.py \
      --tp 8 --pp 2 --save activations.pt --interactive
'
```

Only `--interactive-rank` enters `pdb` and the other ranks wait at a barrier. The default is global rank 0. Useful debugger locals are `local_activations`, `local_stats`, and `gathered_payload`.

## Output

Rank 0 writes a `.pt` file with this structure:

```python
{
    "metadata": {...},
    "activations": {
        "layer_001": torch.Tensor,  # [sequence, batch, hidden]
        ...
        "layer_012": torch.Tensor,
    },
    "stats": {
        "layer_001": {"shape": [32, 1, 1024], "mean": ..., "std": ..., "min": ..., "max": ...},
        ...
    },
    "sources": {
        "layer_001": {"rank": 0, "pp_rank": 0, ...},
        ...
    },
}
```

Example console output:

```text
Saved activations to debug_forward_pass_activations.pt
Layer activation statistics:
  layer_001 pp=0 rank=0 shape=32x1x1024 dtype=torch.float32 mean=-0.000231 std=0.029811 min=-0.119734 max=0.123802
  ...
  layer_012 pp=1 rank=8 shape=32x1x1024 dtype=torch.float32 mean=0.001194 std=0.087522 min=-0.352901 max=0.341665
```

## Inspect Saved Activations

Print stats:

```bash
python examples/debug_forward_pass/inspect_activations.py debug_forward_pass_activations.pt
```

Plot histograms:

```bash
python examples/debug_forward_pass/inspect_activations.py \
  debug_forward_pass_activations.pt \
  --plot-dir activation_histograms
```

Compare two runs:

```bash
python examples/debug_forward_pass/inspect_activations.py \
  run_a.pt \
  --compare run_b.pt
```

Restrict to specific layers:

```bash
python examples/debug_forward_pass/inspect_activations.py run_a.pt --layer layer_006 --layer layer_012
```

## Modify The POC

Model shape is controlled by `debug_forward_pass.py` arguments:

```bash
--num-layers 24 --hidden-size 2048 --num-attention-heads 16 --seq-length 128
```

Keep `num_attention_heads` divisible by TP. If you change PP, `num_layers` should divide cleanly enough for the Megatron-Core pipeline partitioning you want.

Hooks are registered in `register_activation_hooks()`. To capture attention or MLP internals instead of whole-layer outputs, attach hooks to modules such as `layer.self_attention` or `layer.mlp` and update the saved key names. Be careful with tensor parallel internals: unlike the whole layer output in this POC, internal projections may be TP-sharded.

Sequence parallelism is intentionally disabled. If you enable it, hidden states are partitioned across TP ranks along sequence dimension and the capture logic must gather those shards before saving rank-0 tensors.
