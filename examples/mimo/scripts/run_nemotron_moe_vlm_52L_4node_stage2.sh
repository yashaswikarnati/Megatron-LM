#!/bin/bash
#
# MIMO Nemotron6-MoE VLM – Stage 2 pre-training (4 nodes, 32x H100).
# Frozen ViT, trainable projection + LLM.
# Loads from non-MIMO Nemotron VLM checkpoint via --nemotron-checkpoint.
# Restart-aware: resumes from CHECKPOINT_DIR if checkpoint exists.
#

#
# Usage:
#   sbatch examples/mimo/scripts/run_nemotron_moe_vlm_52L_4node_stage2.sh
#   # or interactively:
#   srun --nodes=4 ... bash examples/mimo/scripts/run_nemotron_moe_vlm_52L_4node_stage2.sh
#
# Environment variables (all have defaults):
#   NEMOTRON_CKPT  – path to non-MIMO Nemotron VLM checkpoint
#   DATA_PATH      – energon MetadatasetV2 blend YAML (OmniCorpus)
#   TOKENIZER_MODEL
#   TRAINING_STAGE – stage1 | stage2 (default) | stage3
#   CONTAINER_IMAGE – sqsh image used for all srun steps
#   CONTAINER_MOUNTS – comma-separated mounts passed to srun container runtime

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH --job-name=mimo_nemotron_moe_vlm_stage2
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_genai
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

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_42663997_energon.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1}"

SRUN_CONTAINER_ARGS=()
if [ -n "${CONTAINER_IMAGE}" ]; then
    SRUN_CONTAINER_ARGS+=(--container-image "${CONTAINER_IMAGE}")
    SRUN_CONTAINER_ARGS+=(--container-mounts "${CONTAINER_MOUNTS}")
fi

# ── Paths ────────────────────────────────────────────────────────────────────
# Under sbatch, BASH_SOURCE may point to a staged Slurm copy.
# Prefer the submit directory when it looks like the repo root.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]; then
    MEGATRON_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MEGATRON_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${MEGATRON_ROOT}:${MEGATRON_ROOT}/examples/mimo:${PYTHONPATH:-}"

# ── Training stage (freeze control) ─────────────────────────────────────────
TRAINING_STAGE="${TRAINING_STAGE:-stage2}"

case "$TRAINING_STAGE" in
    stage1)
        FREEZE_ARGS=(--freeze-vit --freeze-lm)
        STAGE_DESC="projection only (ViT frozen, LLM frozen)"
        ;;
    stage2)
        FREEZE_ARGS=(--freeze-vit)
        STAGE_DESC="projection + LLM (ViT frozen)"
        ;;
    stage3)
        FREEZE_ARGS=()
        STAGE_DESC="full fine-tune (nothing frozen)"
        ;;
    *)
        echo "ERROR: Unknown TRAINING_STAGE='$TRAINING_STAGE'. Use stage1, stage2, or stage3."
        exit 1
        ;;
esac

# ── Cluster / parallelism ───────────────────────────────────────────────────
GPUS_PER_NODE=8
NNODES="${SLURM_NNODES:-4}"
NGPUS=$((NNODES * GPUS_PER_NODE))

TP=8
EP=8
PP=1
GBS=128

# ── Output directories ──────────────────────────────────────────────────────
MODEL_NAME="mimo_nemotron_moe_vlm_52L_${TRAINING_STAGE}"
# Keep outputs under repo-local ./logs by default; allow override via BASE_DIR.
BASE_DIR="${BASE_DIR:-${MEGATRON_ROOT}/logs/${MODEL_NAME}}"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
TENSORBOARD_DIR="${BASE_DIR}/tensorboard"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}"

# Triton cache in tmpfs to avoid Lustre lock contention.
export TRITON_CACHE_DIR="/tmp/triton_cache_${MODEL_NAME}"
export TRITON_HOME="${TRITON_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"

# ── Checkpoint (non-MIMO Nemotron VLM) ───────────────────────────────────────
NEMOTRON_CKPT="${NEMOTRON_CKPT:-/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/sasatheesh/data/models/v0/3B_moe_vlm_omnicorpus_adapter_only/checkpoints}"

# ── Data (OmniCorpus multimodal blend) ───────────────────────────────────────
DATA_PATH="${DATA_PATH:-${MEGATRON_ROOT}/examples/mimo/data/test_interleaved_blend.yaml}"
# examples/mimo/data/test_interleaved_blend.yaml
# examples/mimo/data/omnicorpus_blend.yaml
# ── Tokenizer ────────────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

# ── Restart-aware checkpoint loading ─────────────────────────────────────────
# If we already have a MIMO checkpoint saved, resume from it.
# Otherwise, initialize from the non-MIMO Nemotron VLM checkpoint.
if [ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo ">>> Resuming from existing MIMO checkpoint in ${CHECKPOINT_DIR}"
    CKPT_ARGS=(--load "${CHECKPOINT_DIR}")
else
    echo ">>> First run: loading from non-MIMO checkpoint ${NEMOTRON_CKPT}"
    CKPT_ARGS=(--nemotron-checkpoint "${NEMOTRON_CKPT}" --finetune)
fi

# ── Model parallelism ────────────────────────────────────────────────────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size 1
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
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
    --encoder-seq-length 4096
    --position-embedding-type none
    --normalization RMSNorm
    --disable-bias-linear
    --squared-relu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.0173
    --recompute-activations
    --no-create-attention-mask-in-dataloader
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

# ── Vision (RADIO) ───────────────────────────────────────────────────────────
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
    "${FREEZE_ARGS[@]}"
)

# ── Training (stage 2: lr, warmup, etc. matching reference) ──────────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size ${GBS}  
    --train-iters 40000
    --lr 2e-4
    --min-lr 2e-6
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
    --lr-warmup-iters 1000
    --lr-decay-iters 40000
    --lr-wsd-decay-iters 4000
    --model-provider nemotron_moe_vlm
    --dataset-provider energon_multimodal
    --bf16
)

# ── Data / Tokenizer ────────────────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-type external
    --data-path "${DATA_PATH}"
    --total-seq-length 4096
    --num-workers 2
)

TOKENIZER_ARGS=(
    --tokenizer-type MultimodalTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --tokenizer-prompt-format nemotron6-moe
    --special-tokens "<image>"
)

# ── Checkpointing / logging ─────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-mimo-nemotron-moe-vlm}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-stage2-multimodal-blend}"

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
    # NOTE: --overlap-grad-reduce and --overlap-param-gather are disabled because
    # MoE expert-parallelism means not all trainable expert params receive gradients
    # on every rank each step, which triggers an assertion in DDP bucket tracking.
    --distributed-timeout-minutes 60
    --use-fused-weighted-squared-relu
    --enable-experimental
)


# ── Launch ───────────────────────────────────────────────────────────────────
echo "=== MIMO Nemotron6-MoE VLM Stage 2 Pre-training ==="
echo "Stage:  ${TRAINING_STAGE} — ${STAGE_DESC}"
echo "Nodes:  ${NNODES} | GPUs: ${NGPUS} | TP=${TP} EP=${EP} PP=${PP}"
echo "Layers: 52 | Experts: 128 | Hidden: 2688"
echo "Pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
echo "GBS:    ${GBS} | MBS: 1 | Seq: 4096 | Iters: 40000"
echo "LR:     2e-4 → 2e-6 (WSD)"
echo "Data:   ${DATA_PATH}"
echo "Ckpt:   ${CKPT_ARGS[*]}"
echo "Output: ${BASE_DIR}"
echo "Container: ${CONTAINER_IMAGE}"
echo "======================================================"

DATETIME=$(date +'%y%m%d_%H%M%S')


# Inside SLURM allocation — use srun.
srun "${SRUN_CONTAINER_ARGS[@]}" -l \
    --output="${LOG_DIR}/%x_%j_${DATETIME}.log" \
    uv run python -u "${MEGATRON_ROOT}/examples/mimo/train.py" \
        "${GPT_MODEL_ARGS[@]}" \
        "${HYBRID_ARGS[@]}" \
        "${VISION_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${PERF_ARGS[@]}"
