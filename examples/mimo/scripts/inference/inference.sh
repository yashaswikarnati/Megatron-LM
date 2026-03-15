#!/bin/bash
# MIMO model inference script.
#
# Usage:
#   # Text-only (default):
#   bash examples/mimo/scripts/inference/inference.sh <megatron_root>
#
#   # Vision+text:
#   bash examples/mimo/scripts/inference/inference.sh <megatron_root> --mode vision_text
#
#   # Custom prompt:
#   bash examples/mimo/scripts/inference/inference.sh <megatron_root> --prompt "What is gravity?"
#
#   # Custom checkpoint + parallelism:
#   TP=2 EP=4 CKPT=/path/to/ckpt bash examples/mimo/scripts/inference/inference.sh <megatron_root>
#
# Environment overrides:
#   TP          Tensor parallel size (default: 8)
#   EP          Expert parallel size (default: 8)
#   CKPT        Checkpoint path (default: sasatheesh v0)
#   DATA_PATH   Data config for vision_text mode (default: test_interleaved_blend.yaml)
#   TOKENS      Max output tokens (default: 30)

set -euo pipefail

MEGATRON_ROOT="${1:?Usage: $0 <megatron_root> [extra args...]}"
shift  # remaining args passed through

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${MEGATRON_ROOT}:${MEGATRON_ROOT}/examples/mimo"

cd "${MEGATRON_ROOT}"
echo "=== Root: ${MEGATRON_ROOT} ==="
echo "=== Commit: $(git log --oneline -1 2>/dev/null || echo unknown) ==="

# Defaults (overridable via env vars)
TP="${TP:-8}"
EP="${EP:-8}"
CKPT="${CKPT:-/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/sasatheesh/data/models/v0/3B_moe_vlm_omnicorpus_adapter_only/checkpoints}"
CKPT_FORMAT="${CKPT_FORMAT:-nemotron}"  # "nemotron" or "load"
CKPT_STEP="${CKPT_STEP:-}"
TOKENIZER="${TOKENIZER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"
DATA_PATH="${DATA_PATH:-examples/mimo/data/test_interleaved_blend.yaml}"
TOKENS="${TOKENS:-30}"

echo "=== TP=${TP} EP=${EP} TOKENS=${TOKENS} ==="
echo "=== CKPT: ${CKPT} (format: ${CKPT_FORMAT}) ==="

# Build checkpoint args
CKPT_ARGS=""
if [ "${CKPT_FORMAT}" = "load" ]; then
    CKPT_ARGS="--load ${CKPT} --finetune --no-load-optim --no-load-rng"
    if [ -n "${CKPT_STEP}" ]; then
        CKPT_ARGS="${CKPT_ARGS} --ckpt-step ${CKPT_STEP}"
    fi
else
    CKPT_ARGS="--nemotron-checkpoint ${CKPT} --finetune"
fi

uv run python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes 1 --tee 3 \
    "${SCRIPT_DIR}/inference.py" \
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --num-layers 52 \
    --hidden-size 2688 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 128 \
    --ffn-hidden-size 1856 \
    --max-position-embeddings 8192 \
    --encoder-seq-length 4096 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --disable-bias-linear \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.0173 \
    --no-create-attention-mask-in-dataloader \
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-grouped-gemm \
    --moe-router-score-function sigmoid \
    --moe-aux-loss-coeff 0.0001 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size 3712 \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --is-hybrid-model \
    --hybrid-override-pattern "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME" \
    --mamba-num-heads 64 \
    --mamba-head-dim 64 \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --attention-backend flash \
    --img-h 512 \
    --img-w 512 \
    --patch-dim 16 \
    --pixel-shuffle \
    --disable-vision-class-token \
    --vision-model-type radio \
    --class-token-len 8 \
    --max-num-tiles 12 \
    --use-tiling \
    --use-thumbnail \
    --freeze-vit \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size "${EP}" \
    --expert-tensor-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 0 \
    --lr 0 \
    --min-lr 0 \
    --bf16 \
    --dataloader-type external \
    --data-path "${DATA_PATH}" \
    --total-seq-length 4096 \
    --num-workers 2 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model "${TOKENIZER}" \
    --tokenizer-prompt-format nemotron6-moe \
    --special-tokens "<image>" \
    --log-interval 1 \
    --eval-interval 99999 \
    --eval-iters 0 \
    ${CKPT_ARGS} \
    --max-out-tokens "${TOKENS}" \
    --distributed-timeout-minutes 30 \
    "$@"
