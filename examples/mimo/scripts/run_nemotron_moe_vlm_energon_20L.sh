#!/bin/bash
# Replicate sasatheesh/pre-vlm-01 unfrozen-LM training with MIMO backend.
#
# Reference: examples/multimodal/v3/run_unfrozen_lm_local.sh (LLaVA path)
# This script: same architecture + training config, using MIMO model provider.
#
# 20-layer Nemotron6-MoE hybrid (Mamba + MoE + Attention), RADIO ViT,
# TP=2, EP=4, 1 node / 8 GPUs.
#
# Usage (from repo root):
#   ./examples/mimo/scripts/run_nemotron_moe_vlm_energon_20L.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
NUM_NODES=1

LOG_DIR=./logs/nemotron_moe_vlm_energon_20L_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

CHECKPOINT_PATH="$LOG_DIR/checkpoints"
mkdir -p "$CHECKPOINT_PATH"

# ── Data path (OmniCorpus blend — same as reference) ─────────────────
DATA_PATH="${DATA_PATH:-examples/mimo/data/omnicorpus_blend.yaml}"

# ── Tokenizer ─────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"

# ── Distributed launch ─────────────────────────────────────────────────
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --tee 3
    --log-dir "$LOG_DIR"
)

# ── Model parallelism (matches reference: TP=2, EP=4, PP=1) ──────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 4
    --sequence-parallel
    --use-distributed-optimizer
)

# ── Language model architecture (Nemotron6-MoE, 20 layers) ────────────
# Matches reference base_3B_moe_config.sh with --num-layers 20 override.
GPT_MODEL_ARGS=(
    --num-layers 20
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

# ── MoE / Mamba hybrid ─────────────────────────────────────────────────
# 20-layer pattern from reference: 3 attn (*), 9 MoE (E), 8 Mamba (M)
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
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-overlap
    --moe-permute-fusion
    --is-hybrid-model
    --hybrid-override-pattern "MEMEM*EMEMEM*EMEMEM*"
    --mamba-num-heads 64
    --mamba-head-dim 64
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
    --attention-backend flash
)

# ── Vision (RADIO) ─────────────────────────────────────────────────────
VISION_ARGS=(
    --img-h 512
    --img-w 512
    --patch-dim 16
    --pixel-shuffle
    --disable-vision-class-token
    --freeze-vit
    --vision-model-type radio
    --class-token-len 8
    --max-num-tiles 12
    --use-tiling
    --use-thumbnail
)

# ── Training (matches reference LR, optimizer, schedule) ──────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 10000
    --lr 2e-4
    --min-lr 2e-6
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr-decay-style cosine
    --lr-warmup-iters 2
    --lr-decay-iters 10
    --model-provider nemotron_moe_vlm
    --dataset-provider energon_multimodal
    --bf16
)

# ── Data / Tokenizer ──────────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-type external
    --data-path "$DATA_PATH"
    --image-token-id 128256
    --pad-token-id 128004
    --total-seq-length 4096
    --num-workers 2
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

# ── Eval / logging ─────────────────────────────────────────────────────
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 99999
    --eval-iters 0
    --save "$CHECKPOINT_PATH"
    --tensorboard-dir "$LOG_DIR/tensorboard"
)

TP=2; EP=4; DP=$((GPUS_PER_NODE / TP))
echo "=== Nemotron6-MoE VLM 20L Training (MIMO + energon) ==="
echo "GPUs: $GPUS_PER_NODE | TP=$TP EP=$EP DP=$DP"
echo "Layers: 20 | Pattern: MEMEM*EMEMEM*EMEMEM*"
echo "Seq: 4096 | GBS: 8 | Iters: 10"
echo "Data: $DATA_PATH"
echo "Logs: $LOG_DIR"
echo "========================================================="

uv run python -m torch.distributed.run \
    "${DISTRIBUTED_ARGS[@]}" \
    examples/mimo/train.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${HYBRID_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${TOKENIZER_ARGS[@]}"
