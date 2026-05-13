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

GPUS_PER_NODE="${GPUS_PER_NODE:-}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
NUM_MICROBATCHES="${NUM_MICROBATCHES:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
ENCODER_TP="${ENCODER_TP:-2}"
ENCODER_PP="${ENCODER_PP:-1}"
ENCODER_DP="${ENCODER_DP:-2}"
LLM_TP="${LLM_TP:-2}"
LLM_PP="${LLM_PP:-1}"
LLM_DP="${LLM_DP:-2}"
LLM_EP="${LLM_EP:-4}"
ENABLE_EXPERIMENTAL="${ENABLE_EXPERIMENTAL:-1}"
MOE_ROUTER_FORCE_LOAD_BALANCING="${MOE_ROUTER_FORCE_LOAD_BALANCING:-0}"
ENCODER_SIZE=$((ENCODER_TP * ENCODER_PP * ENCODER_DP))
LLM_SIZE=$((LLM_TP * LLM_PP * LLM_DP))
LLM_OFFSET="${LLM_OFFSET:-${ENCODER_SIZE}}"
EXPECTED_WORLD_SIZE=$((ENCODER_SIZE + LLM_SIZE))
GPUS_PER_NODE="${GPUS_PER_NODE:-${EXPECTED_WORLD_SIZE}}"
if [[ "${GPUS_PER_NODE}" -ne "${EXPECTED_WORLD_SIZE}" ]]; then
  echo "ERROR: GPUS_PER_NODE=${GPUS_PER_NODE} but hetero layout requires ${EXPECTED_WORLD_SIZE}" >&2
  exit 1
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP))}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
MAX_SAMPLES_PER_SEQUENCE="${MAX_SAMPLES_PER_SEQUENCE:-100}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/kshih/workspace/blends/eagle_recipe_online_packing/final_recipe/pretrain_base_non_sft_cw_dfw.yaml}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

if [[ "${VERIFY_ENERGON:-1}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" bash examples/mimo/scripts/verify_energon.sh
fi

echo "=== Hetero MIMO Nemotron6-MoE VLM 20L Energon training ==="
echo "stage=${TRAINING_STAGE} train_iters=${TRAIN_ITERS} gbs=${GLOBAL_BATCH_SIZE}"
echo "layout=encoder(tp=${ENCODER_TP},pp=${ENCODER_PP},dp=${ENCODER_DP}) llm(tp=${LLM_TP},pp=${LLM_PP},dp=${LLM_DP},ep=${LLM_EP}) world=${EXPECTED_WORLD_SIZE}"
echo "enable_experimental=${ENABLE_EXPERIMENTAL}"
echo "moe_router_force_load_balancing=${MOE_ROUTER_FORCE_LOAD_BALANCING}"
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
MODEL_ARGS=()
if [[ "${ENABLE_EXPERIMENTAL}" == "1" || "${ENABLE_EXPERIMENTAL}" == "true" ]]; then
  MODEL_ARGS+=(--enable-experimental)
fi
if [[ "${MOE_ROUTER_FORCE_LOAD_BALANCING}" == "1" || "${MOE_ROUTER_FORCE_LOAD_BALANCING}" == "true" ]]; then
  MODEL_ARGS+=(--moe-router-force-load-balancing)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${GPUS_PER_NODE}" \
  examples/mimo/train_hetero.py \
  --model-provider nemotron-moe-vlm-20l \
  --dataset-provider energon_multimodal \
  --training-stage "${TRAINING_STAGE}" \
  --encoder-tp "${ENCODER_TP}" \
  --encoder-pp "${ENCODER_PP}" \
  --encoder-dp "${ENCODER_DP}" \
  --llm-offset "${LLM_OFFSET}" \
  --llm-tp "${LLM_TP}" \
  --llm-pp "${LLM_PP}" \
  --llm-dp "${LLM_DP}" \
  --llm-ep "${LLM_EP}" \
  --llm-expt-tp 1 \
  --llm-expt-dp 1 \
  "${MODEL_ARGS[@]}" \
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
