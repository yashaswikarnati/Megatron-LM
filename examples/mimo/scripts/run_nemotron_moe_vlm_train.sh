#!/bin/bash
# Train Nemotron6-MoE VLM (RADIO + MambaModel) with mock data on 8 GPUs.
#
# Usage (from repo root):
#   ./examples/mimo/scripts/run_nemotron_moe_vlm_train.sh
#
# Uses real model configs (real RADIO vision encoder, Nemotron6-MoE architecture)
# but with a small number of LLM layers (4) to fit on 8 GPUs.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MIMO_DEBUG=1

GPUS_PER_NODE=8
NUM_NODES=1

LOG_DIR=./logs/nemotron_moe_vlm_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

CHECKPOINT_PATH="$LOG_DIR/checkpoints"
mkdir -p "$CHECKPOINT_PATH"

# ── Distributed launch ─────────────────────────────────────────────────
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --redirects 3
    --log-dir "$LOG_DIR"
)

# ── Model parallelism ──────────────────────────────────────────────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 4
    --sequence-parallel
)

# ── GPT / language model (Nemotron6-MoE, 4 layers for mock testing) ───
GPT_MODEL_ARGS=(
    --num-layers 4
    --hidden-size 2688
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --ffn-hidden-size 1856
    --max-position-embeddings 4096
    --encoder-seq-length 4096
    --position-embedding-type none
    --normalization RMSNorm
    --disable-bias-linear
    --squared-relu
    --untie-embeddings-and-output-weights
)

# ── MoE / Mamba hybrid ─────────────────────────────────────────────────
HYBRID_ARGS=(
    --num-experts 128
    --moe-router-topk 6
    --moe-grouped-gemm
    --is-hybrid-model
    --hybrid-override-pattern "MEME"
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
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
)

# ── Training ────────────────────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 20
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction 0.001
    --lr-decay-iters 10
    --model-provider nemotron_moe_vlm
    --dataset-provider mock
    --bf16
)

# ── Mock data ───────────────────────────────────────────────────────────
MOCK_DATA_ARGS=(
    --image-size 512
    --total-seq-length 512
    --image-seq-length 256
    --image-token-id 32000
)

# ── Checkpoint loading (optional) ─────────────────────────────────────
# Set LM_CKPT to load pretrained language model weights (iter directory).
# Set VIT_CKPT to load pretrained vision encoder weights.
LM_CKPT="${LM_CKPT:-}"
VIT_CKPT="${VIT_CKPT:-}"

CKPT_ARGS=()
if [[ -n "$LM_CKPT" ]]; then
    CKPT_ARGS+=(--language-model-checkpoint "$LM_CKPT")
fi
if [[ -n "$VIT_CKPT" ]]; then
    CKPT_ARGS+=(--vision-encoder-checkpoint "$VIT_CKPT")
fi

# ── Eval / logging ─────────────────────────────────────────────────────
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --save "$CHECKPOINT_PATH"
    --eval-iters 2
    --tensorboard-dir "$LOG_DIR/tensorboard"
)

# ── Tokenizer (placeholder, not used in mock path) ─────────────────────
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'llava-hf/llava-1.5-7b-hf'
)

echo "=== Nemotron6-MoE VLM Training (mock data) ==="
echo "GPUs: $GPUS_PER_NODE | Logs: $LOG_DIR"
echo "================================================"

uv run python -m torch.distributed.run \
    "${DISTRIBUTED_ARGS[@]}" \
    examples/mimo/train.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${HYBRID_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MOCK_DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${TOKENIZER_ARGS[@]}" \
    "${CKPT_ARGS[@]}"
