#!/bin/bash
# Train Nemotron6-MoE VLM (RADIO + MambaModel) with real energon data on 8 GPUs.
#
# Usage (from repo root):
#   ./examples/mimo/scripts/run_nemotron_moe_vlm_energon_train.sh
#
# Uses real model configs (real RADIO vision encoder, Nemotron6-MoE architecture)
# but with a small number of LLM layers (4) to fit on 8 GPUs.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MIMO_DEBUG=1

GPUS_PER_NODE=8
NUM_NODES=1

LOG_DIR=./logs/nemotron_moe_vlm_energon_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

CHECKPOINT_PATH="$LOG_DIR/checkpoints"
mkdir -p "$CHECKPOINT_PATH"

# ── Data path (energon MetadatasetV2 blend YAML) ─────────────────────
DATA_PATH="${DATA_PATH:-examples/mimo/data/test_interleaved_blend.yaml}"

# ── Tokenizer ─────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"

# ── Distributed launch ─────────────────────────────────────────────────
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --tee 3
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

# ── GPT / language model (Nemotron6-MoE architecture, reduced layers) ─
GPT_MODEL_ARGS=(
    --num-layers 4              # reduced from 52 for testing
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
)

# ── MoE / Mamba hybrid ─────────────────────────────────────────────────
HYBRID_ARGS=(
    --num-experts 128
    --moe-router-topk 6
    --moe-grouped-gemm
    --is-hybrid-model
    --hybrid-override-pattern "MEME"   # 4 layers: Mamba-MoE-Mamba-MoE
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
    --max-num-tiles 12
    --use-tiling
    --use-thumbnail
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

# ── Checkpoint loading (optional) ─────────────────────────────────────
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

echo "=== Nemotron6-MoE VLM Training (energon multimodal data) ==="
echo "GPUs: $GPUS_PER_NODE | Logs: $LOG_DIR"
echo "Data: $DATA_PATH"
echo "Tokenizer: $TOKENIZER_MODEL"
echo "============================================================="

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
    "${TOKENIZER_ARGS[@]}" \
    "${CKPT_ARGS[@]}"
