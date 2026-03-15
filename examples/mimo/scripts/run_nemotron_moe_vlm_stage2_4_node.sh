#!/bin/bash
#
# MIMO Nemotron6-MoE VLM – Stage 2 training (64 nodes, 512x H100).
# Frozen ViT, trainable projection + LLM.
# Loads from Stage 1 adapter-only checkpoint.
#
# Matches Kevin's stage 2 settings from NMFW-25:
#   lr=5e-4, GBS=256, WSD schedule (500 warmup, 4000 stable, 500 decay)
#   TP=4, EP=64, PP=1, 64 nodes
#
# Usage:
#   sbatch examples/mimo/scripts/run_nemotron_moe_vlm_stage2_64node.sh
#
# Environment variables (all have defaults):
#   STAGE1_CKPT     – path to Stage 1 adapter-only checkpoint
#   DATA_PATH       – energon MetadatasetV2 blend YAML (OmniCorpus)
#   TOKENIZER_MODEL
#   CONTAINER_IMAGE
#   CONTAINER_MOUNTS

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH --job-name=mimo_vlm_stage2
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --dependency=singleton

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TRITON_ALWAYS_BENCH=0
export TORCHINDUCTOR_FX_GRAPH_CACHE=0

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_42663997_energon_v2.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1}"

SRUN_CONTAINER_ARGS=()
if [ -n "${CONTAINER_IMAGE}" ]; then
    SRUN_CONTAINER_ARGS+=(--container-image "${CONTAINER_IMAGE}")
    SRUN_CONTAINER_ARGS+=(--container-mounts "${CONTAINER_MOUNTS}")
fi

# ── Paths ────────────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]; then
    MEGATRON_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MEGATRON_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${MEGATRON_ROOT}:${MEGATRON_ROOT}/examples/mimo:${PYTHONPATH:-}"

# ── Cluster / parallelism (TP=4, EP=64, PP=1) ───────────────────────────────
GPUS_PER_NODE=8
NNODES="${SLURM_NNODES:-64}"
NGPUS=$((NNODES * GPUS_PER_NODE))

TP=2
EP=32
PP=1
GBS=1024

# ── Output directories ──────────────────────────────────────────────────────
MODEL_NAME="mimo_vlm_stage2"
BASE_DIR="${BASE_DIR:-${MEGATRON_ROOT}/logs/${MODEL_NAME}}"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
TENSORBOARD_DIR="${BASE_DIR}/tensorboard"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"

export TRITON_CACHE_DIR="/tmp/triton_cache_${MODEL_NAME}"
export TRITON_HOME="${TRITON_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"

# ── Checkpoint (Stage 1 adapter-only output) ─────────────────────────────────
STAGE1_CKPT="${STAGE1_CKPT:-${MEGATRON_ROOT}/logs/mimo_vlm_adapter_only_stage1/checkpoints}"
STAGE1_ITER="${STAGE1_ITER:-3000}"

# ── Restart-aware checkpoint loading ─────────────────────────────────────────
# If we already have a stage 2 checkpoint, resume from it.
# Otherwise, load from stage 1 adapter-only checkpoint (specific iteration).
if [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo ">>> Resuming from existing stage 2 checkpoint in ${CHECKPOINT_DIR}"
    CKPT_ARGS=(--load "${CHECKPOINT_DIR}")
else
    echo ">>> First run: loading from stage 1 checkpoint ${STAGE1_CKPT} (iter ${STAGE1_ITER})"
    CKPT_ARGS=(--load "${STAGE1_CKPT}" --ckpt-step "${STAGE1_ITER}" --finetune --no-load-optim --no-load-rng)
fi

# ── Data (OmniCorpus: 50% text + 50% multimodal) ────────────────────────────
DATA_PATH="${DATA_PATH:-${MEGATRON_ROOT}/examples/mimo/data/text_omnicorpus_blend.yaml}"

# ── Tokenizer ────────────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

# ── Model parallelism ────────────────────────────────────────────────────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size 1
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --ddp-num-buckets 8
    --ddp-pad-buckets-for-high-nccl-busbw
)

# ── Language model architecture (full 52-layer Nemotron6-MoE) ────────────────
GPT_MODEL_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --num-layers 52
    --hidden-size 2688
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --ffn-hidden-size 1856
    --max-position-embeddings 8192
    --encoder-seq-length 8192
    --position-embedding-type none
    --normalization RMSNorm
    --disable-bias-linear
    --squared-relu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.0173
    --no-create-attention-mask-in-dataloader
    --tiktoken-pattern v2
)

# ── Recompute (full granularity for 64-node memory) ──────────────────────────
RECOMPUTE_ARGS=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# ── MoE / Mamba hybrid (full 52-layer pattern) ──────────────────────────────
HYBRID_ARGS=(
    --num-experts 128
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --moe-aux-loss-coeff 0.0001
    --moe-router-topk-scaling-factor 2.5
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32
    --moe-router-load-balancing-type seq_aux_loss
    --moe-shared-expert-intermediate-size 3712
    --moe-shared-expert-overlap
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --is-hybrid-model
    --hybrid-override-pattern "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    --mamba-num-heads 64
    --mamba-head-dim 64
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
    --attention-backend flash
)

# ── Vision (RADIO) — freeze ViT only, LLM trainable ─────────────────────────
VISION_ARGS=(
    --img-h 512
    --img-w 512
    --patch-dim 16
    --pixel-shuffle
    --disable-vision-class-token
    --vision-model-type radio
    --class-token-len 8
    --max-num-tiles 12
    --use-tiling
    --use-thumbnail
    --freeze-vit
)

# ── Training (stage 2: WSD schedule, matching Kevin's settings) ──────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size ${GBS}
    --train-iters 10000
    --lr 5e-5
    --min-lr 0
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
    --lr-warmup-iters 500
    --lr-decay-iters 10000
    --lr-wsd-decay-iters 500
    --model-provider nemotron_moe_vlm
    --dataset-provider energon_multimodal
    --bf16
)

# ── Data / Tokenizer ────────────────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-type external
    --data-path "${DATA_PATH}"
    --total-seq-length 8192
    --num-workers 2
    --packing-buffer-size 128
    --use-loss-scaling
)

TOKENIZER_ARGS=(
    --tokenizer-type MultimodalTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --tokenizer-prompt-format nemotron6-moe
    --special-tokens "<image>"
)

# ── Checkpointing / logging ─────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-vlm-adapter-training}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-mimo-stage2-omnicorpus}"

EVAL_AND_LOGGING_ARGS=(
    --save "${CHECKPOINT_DIR}"
    --save-interval 1000
    --eval-interval 99999999
    --eval-iters 0
    --log-interval 1
    --log-progress
    --log-throughput
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-exp-name "${WANDB_EXP_NAME}"
    --ckpt-format torch_dist
    --ckpt-fully-parallel-save
    --ckpt-fully-parallel-load
    --ckpt-assume-constant-structure
    --manual-gc
)

# ── Performance ──────────────────────────────────────────────────────────────
PERF_ARGS=(
    --distributed-timeout-minutes 60
    --use-fused-weighted-squared-relu
    --enable-experimental
    --no-check-for-nan-in-loss-and-grad
    --rerun-mode disabled
)

# ── Launch ───────────────────────────────────────────────────────────────────
echo "=== MIMO VLM Stage 2 Training (Projection + LLM) ==="
echo "Nodes:  ${NNODES} | GPUs: ${NGPUS} | TP=${TP} EP=${EP} PP=${PP}"
echo "Layers: 52 | Experts: 128 | Hidden: 2688"
echo "Pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
echo "GBS:    ${GBS} | MBS: 1 | total-seq-length: 8192 | Iters: 10000"
echo "LR:     5e-5 → 0 (WSD minus_sqrt, warmup=500, decay=500)"
echo "Freeze: ViT only (projection + LLM trainable)"
echo "Data:   ${DATA_PATH}"
echo "Ckpt:   ${CKPT_ARGS[*]}"
echo "Output: ${BASE_DIR}"
echo "Container: ${CONTAINER_IMAGE}"
echo "======================================================"

DATETIME=$(date +'%y%m%d_%H%M%S')

srun "${SRUN_CONTAINER_ARGS[@]}" -l \
    --output="${LOG_DIR}/%x_%j_${DATETIME}.log" \
    uv run python -u "${MEGATRON_ROOT}/examples/mimo/train.py" \
        "${GPT_MODEL_ARGS[@]}" \
        "${RECOMPUTE_ARGS[@]}" \
        "${HYBRID_ARGS[@]}" \
        "${VISION_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${PERF_ARGS[@]}"
