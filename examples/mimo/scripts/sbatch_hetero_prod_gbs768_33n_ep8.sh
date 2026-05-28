#!/bin/bash
# Production hetero MIMO Nemotron6-MoE VLM training at 33 nodes (1 enc + 32 LLM).
# Adapted from sbatch_hetero_parity_gbs768_33n_ep8.sh (scaling-study sbatch) by
# pinning Sanjeev's production knobs from
# examples/multimodal/v3/pretrain_3b_nano_vlm_sota_90t_10v.sh:
#   * TRAIN_SAMPLES=122070313
#   * LR_WARMUP_SAMPLES=1024000, LR_DECAY_SAMPLES=121046313,
#     LR_WSD_DECAY_SAMPLES=18310547, LR_WSD_DECAY_STYLE=minus_sqrt
#   * PACKING_BUFFER_SIZE=128
#   * SAVE_INTERVAL=1000 (LOG_INTERVAL=1 for per-iter visibility)
#   * NUM_WORKERS=2
# Deviations from Sanjeev's baseline:
#   * LLM_EP=8 (vs Sanjeev's EP=16) — kept from our scaling study
#   * Hetero topology TP=2 (vs Sanjeev's TP=4)
#   * MOE_ROUTER_FORCE_LOAD_BALANCING=0 (natural seq_aux_loss)
#   * No MTP layers

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 33
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH -J mimo-prod-gbs768-33n-ep8
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

SCRATCH_ROOT=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch
CONTAINER_IMAGE="${HETERO_CONTAINER_IMAGE:-${SCRATCH_ROOT}/images/m_lm_energon_0506.sqsh}"
export HETERO_SKIP_UV="${HETERO_SKIP_UV:-1}"
ENV_ROOT="${SCRATCH_ROOT}/envs/megatron_lm/01f0da7539da4b39"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"
VISION_CKPT="${SCRATCH_ROOT}/encoders/post-c-radio-omni"

NEMOTRON_CKPT="${NEMOTRON_CKPT:-/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/workspace/output/3b_nano_vlm_sota_mtp2_90t10v_post_c_radio_omni_96n_tp2_ep16_selective_300b_20260511/checkpoints/iter_0001000}"

RUN_NAME="mimo-prod-gbs768-33n-ep8"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"

# ---- topology: TP=2 EP=8 LLM (32 nodes, DP=128, expt_dp=32) + TP=1 DP=8 encoder lane (1 node)
ENCODER_TP=1;  ENCODER_CP=1;  ENCODER_PP=1;  ENCODER_DP=8;   ENCODER_EP=1
LLM_TP=2;      LLM_CP=1;      LLM_PP=1;      LLM_DP=128;     LLM_EP=8;    LLM_EXPT_TP=1
LLM_ONLY=0

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=768
NUM_MICROBATCHES=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * LLM_DP) ))   # = 6

# Sanjeev's sample-based WSD schedule from pretrain_3b_nano_vlm_sota_90t_10v.sh.
TRAIN_SAMPLES=122070313
LR_WARMUP_SAMPLES=1024000
LR_DECAY_SAMPLES=$(( TRAIN_SAMPLES - LR_WARMUP_SAMPLES ))   # = 121046313
LR_WSD_DECAY_SAMPLES=18310547
LR_WSD_DECAY_STYLE=minus_sqrt
LR=1.2e-3
MIN_LR=1.2e-5
WEIGHT_DECAY=0.1
LR_DECAY_STYLE=WSD

# --train-samples drives the actual stopping condition; TRAIN_ITERS is a derived
# fallback that validate_args overrides via ceil(TRAIN_SAMPLES / GBS).
TRAIN_ITERS=$(( (TRAIN_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
LOG_INTERVAL=1
SAVE_INTERVAL=1000

TRAINING_STAGE=stage2
MODEL_PROVIDER=nemotron-moe-vlm-54l
ENABLE_EXPERIMENTAL=1
MOE_ROUTER_FORCE_LOAD_BALANCING=0
NUM_WORKERS=2
PACKING_BUFFER_SIZE=128
SHUFFLE_BUFFER_SIZE=100
MAX_SAMPLES_PER_SEQUENCE=100
CHECK_HEL_PATHS=1

WORLD_SIZE=$(( ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP \
             + LLM_TP * LLM_CP * LLM_PP * LLM_DP ))
[[ "${WORLD_SIZE}" -eq 264 ]] || { echo "ERROR: derived world_size=${WORLD_SIZE} (expected 264)" >&2; exit 1; }

mkdir -p "${RUN_DIR}/logs/app" "${RUN_DIR}/logs/torchrun" "${RUN_DIR}/checkpoints" \
         "${RUN_DIR}/tensorboard" "${RUN_DIR}/data_cache" "${RUN_DIR}/tmp"

export REPO_ROOT RUN_DIR SCRATCH_ROOT
export OUTPUT_PATH="${RUN_DIR}" LOG_DIR="${RUN_DIR}/logs/app" APP_LOG_DIR="${RUN_DIR}/logs/app"
export TORCHRUN_LOG_DIR="${RUN_DIR}/logs/torchrun"
export CHECKPOINT_SAVE_PATH="${RUN_DIR}/checkpoints" CHECKPOINT_LOAD_PATH="${NEMOTRON_CKPT}"
export DATALOADER_SAVE_PATH="${DATALOADER_SAVE_PATH:-${CHECKPOINT_SAVE_PATH}/dataloader}"
export DATALOADER_LOAD_PATH="${DATALOADER_LOAD_PATH:-}"
export ENERGON_SAMPLE_TRACE_DIR="${ENERGON_SAMPLE_TRACE_DIR:-}"
export CHECKPOINT_DIR="${RUN_DIR}/checkpoints" TENSORBOARD_PATH="${RUN_DIR}/tensorboard" TB_DIR="${RUN_DIR}/tensorboard"
export DATA_CACHE_DIR="${RUN_DIR}/data_cache"
export TMPDIR="/tmp"

export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache"
export TRITON_CACHE_DIR_BASE="${TRITON_CACHE_DIR_BASE:-${SCRATCH_ROOT}/triton-cache}"
export CUDA_CACHE_PATH="${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache"
export PYTHONPATH="${REPO_ROOT}" PYTHONNOUSERSITE=1 PIP_CONSTRAINT=""
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm" UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16 NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_PROTO=LL128 NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export TRAINING_STAGE MODEL_PROVIDER ENABLE_EXPERIMENTAL MOE_ROUTER_FORCE_LOAD_BALANCING
export TRAIN_ITERS NUM_MICROBATCHES MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE LOG_INTERVAL
export ENCODER_TP ENCODER_CP ENCODER_PP ENCODER_DP ENCODER_EP
export LLM_TP LLM_CP LLM_PP LLM_DP LLM_EP LLM_EXPT_TP LLM_ONLY
export LR MIN_LR WEIGHT_DECAY LR_DECAY_STYLE
export LR_WARMUP_SAMPLES LR_DECAY_SAMPLES LR_WSD_DECAY_SAMPLES LR_WSD_DECAY_STYLE TRAIN_SAMPLES
export NUM_WORKERS PACKING_BUFFER_SIZE SHUFFLE_BUFFER_SIZE MAX_SAMPLES_PER_SEQUENCE CHECK_HEL_PATHS
export TOKENIZER_MODEL VISION_CKPT

RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-}"
LOAD_ARGS=()
if [[ -n "${RESUME_CHECKPOINT_PATH}" ]]; then
  LOAD_ARGS+=(--load "${RESUME_CHECKPOINT_PATH}")
  DATALOADER_LOAD_PATH="${DATALOADER_LOAD_PATH:-${RESUME_CHECKPOINT_PATH}/dataloader}"
else
  LOAD_ARGS+=(--no-load-optim --no-load-rng --load-nemotron-checkpoint "${NEMOTRON_CKPT}")
fi
export DATALOADER_LOAD_PATH

TRAIN_LAUNCH_ARGS=(
  --class-token-len 10
  --image-tag-type internvl
  --max-num-tiles 1
  --overlap-grad-reduce
  --ddp-num-buckets 8 --ddp-pad-buckets-for-high-nccl-busbw
  --correct-encoder-grad-for-partial-participation
  --seed 1234
  --save "${CHECKPOINT_SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
  --dataloader-save "${DATALOADER_SAVE_PATH}"
  "${LOAD_ARGS[@]}"
  --dynamic-resolution
  --tensorboard-dir "${RUN_DIR}/tensorboard"
)
if [[ -n "${DATALOADER_LOAD_PATH}" ]]; then
  TRAIN_LAUNCH_ARGS+=(--dataloader-load "${DATALOADER_LOAD_PATH}")
fi
if [[ -n "${ENERGON_SAMPLE_TRACE_DIR}" ]]; then
  TRAIN_LAUNCH_ARGS+=(--energon-sample-trace-dir "${ENERGON_SAMPLE_TRACE_DIR}")
fi

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/scratch/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
[[ "${REPO_ROOT}" == "${SCRATCH_ROOT}"/* ]] || CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"

echo "=== hetero PROD GBS=768 33n EP=8 (TRAIN_SAMPLES=${TRAIN_SAMPLES}, warmup=${LR_WARMUP_SAMPLES}, wsd-decay=${LR_WSD_DECAY_SAMPLES}, packing=${PACKING_BUFFER_SIZE}) ==="
echo "repo=${REPO_ROOT} run_dir=${RUN_DIR}"
echo "world_size=${WORLD_SIZE} gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES}"
echo "layout: encoder(dp=${ENCODER_DP}) llm(tp=${LLM_TP},dp=${LLM_DP},ep=${LLM_EP})"
echo "ckpt=${NEMOTRON_CKPT}"
echo "========================================================"

srun --kill-on-bad-exit=1 \
  --ntasks="${WORLD_SIZE}" \
  --ntasks-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc 'set -euo pipefail; cd "${REPO_ROOT}";
    if [ -n "${HETERO_SKIP_UV:-}" ]; then export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"; exec bash examples/mimo/scripts/run_hetero_nemotron_54l_hel_train.sh "$@"; else exec uv run --no-sync bash examples/mimo/scripts/run_hetero_nemotron_54l_hel_train.sh "$@"; fi
  ' \
  bash "${TRAIN_LAUNCH_ARGS[@]}"
