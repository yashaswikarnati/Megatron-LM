#!/bin/bash
# Run non-colocated heterogeneous MIMO Nemotron6-MoE VLM 20L training on Energon data.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAINING_STAGE="${TRAINING_STAGE:-stage2}"
case "${TRAINING_STAGE}" in
  stage1|stage2|stage3)
    ;;
  *)
    echo "ERROR: Unknown TRAINING_STAGE='${TRAINING_STAGE}'. Use stage1, stage2, or stage3." >&2
    exit 1
    ;;
esac

if [[ "${INSTALL_ENERGON:-0}" == "1" ]]; then
  if [[ -n "${ENERGON_PATH:-}" ]]; then
    bash examples/mimo/scripts/install_energon.sh "${ENERGON_PATH}"
  else
    bash examples/mimo/scripts/install_energon.sh
  fi
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
NUM_MICROBATCHES="${NUM_MICROBATCHES:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LLM_DP=2
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP))}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
MAX_SAMPLES_PER_SEQUENCE="${MAX_SAMPLES_PER_SEQUENCE:-100}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/kshih/workspace/blends/eagle_recipe_online_packing/final_recipe/pretrain_base_non_sft_cw_dfw.yaml}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

echo "=== Hetero MIMO Nemotron6-MoE VLM 20L Energon training ==="
echo "stage=${TRAINING_STAGE} train_iters=${TRAIN_ITERS} gbs=${GLOBAL_BATCH_SIZE}"
echo "data=${DATA_PATH}"
echo "tokenizer=${TOKENIZER_MODEL}"
echo "==========================================================="

DATA_LOADER_ARGS=(
  --num-workers "${NUM_WORKERS}"
  --shuffle-buffer-size "${SHUFFLE_BUFFER_SIZE}"
  --max-samples-per-sequence "${MAX_SAMPLES_PER_SEQUENCE}"
)
if [[ "${PACKING_BUFFER_SIZE}" != "0" ]]; then
  DATA_LOADER_ARGS+=(--packing-buffer-size "${PACKING_BUFFER_SIZE}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${GPUS_PER_NODE}" \
  examples/mimo/train_hetero.py \
  --model-provider nemotron-moe-vlm-20l \
  --dataset-provider energon_multimodal \
  --training-stage "${TRAINING_STAGE}" \
  --encoder-tp 2 \
  --encoder-pp 1 \
  --encoder-dp 2 \
  --llm-offset 4 \
  --llm-tp 2 \
  --llm-pp 1 \
  --llm-dp "${LLM_DP}" \
  --llm-ep 4 \
  --llm-expt-tp 1 \
  --llm-expt-dp 1 \
  --vocab-size 131072 \
  --max-num-tiles 12 \
  --data-path "${DATA_PATH}" \
  "${DATA_LOADER_ARGS[@]}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --tokenizer-prompt-format nemotron6-moe \
  --image-token "<image>" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --num-microbatches "${NUM_MICROBATCHES}" \
  --lr 2e-4 \
  --min-lr 2e-6 \
  --lr-decay-style cosine \
  --lr-warmup-iters "${LR_WARMUP_ITERS}" \
  --lr-decay-iters "${LR_DECAY_ITERS}" \
  --weight-decay 0.05 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --no-overlap-grad-reduce \
  --ddp-bucket-size 0 \
  --log-interval 1 \
  --train-iters "${TRAIN_ITERS}" \
  "$@"
