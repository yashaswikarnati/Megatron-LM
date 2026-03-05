#!/bin/bash
# Checkpoint sanity check: load a Nemotron VLM checkpoint, compute loss on one
# sample, and autoregressively generate tokens to verify model quality.
#
# Checks:
#   1) Loss on a real sample (should be << ln(vocab_size) ≈ 11.8 for 128k vocab)
#   2) Greedy generation produces coherent text (not garbage)
#
# Usage (from repo root, 8 GPUs with TP=2, EP=4):
#   ./examples/mimo/scripts/test_nemotron_checkpoint_loading.sh
#
# Override:
#   NEMOTRON_CKPT=/path/to/ckpt GENERATE_TOKENS=64 ./examples/mimo/scripts/test_nemotron_checkpoint_loading.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
NUM_NODES=1

# ── Checkpoint path ──────────────────────────────────────────────────
NEMOTRON_CKPT="${NEMOTRON_CKPT:-/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/sasatheesh/data/models/v0/3B_moe_vlm_omnicorpus_adapter_only/checkpoints}"

# ── Data ─────────────────────────────────────────────────────────────
DATA_PATH="${DATA_PATH:-examples/mimo/data/test_interleaved_blend.yaml}"

# ── Tokenizer ────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

# ── Generation length ────────────────────────────────────────────────
GENERATE_TOKENS="${GENERATE_TOKENS:-128}"

# ── Parallelism (single node defaults) ───────────────────────────────
TP="${TP:-8}"
EP="${EP:-8}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --tee 3
)

echo "=== Checkpoint Generation Sanity Check ==="
echo "Checkpoint: $NEMOTRON_CKPT"
echo "Data:       $DATA_PATH"
echo "TP=$TP EP=$EP  Generate=$GENERATE_TOKENS tokens"
echo "============================================"

uv run python -m torch.distributed.run \
    "${DISTRIBUTED_ARGS[@]}" \
    examples/mimo/scripts/test_checkpoint_generate.py \
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
    --moe-permute-fusion \
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
    --model-provider nemotron_moe_vlm \
    --dataset-provider energon_multimodal \
    --bf16 \
    --dataloader-type external \
    --data-path "${DATA_PATH}" \
    --total-seq-length 4096 \
    --num-workers 2 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --tokenizer-prompt-format nemotron6-moe \
    --special-tokens "<image>" \
    --log-interval 1 \
    --eval-interval 99999 \
    --eval-iters 0 \
    --nemotron-checkpoint "${NEMOTRON_CKPT}" \
    --finetune \
    --generate-tokens "${GENERATE_TOKENS}"
