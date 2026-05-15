#!/bin/bash
# Submit the 9-node HEL 54L heterogeneous MIMO run.
#
# Intended use from a Cog-synced nb-hel workspace:
#   sbatch examples/mimo/scripts/sbatch_hetero_nemotron_54l_hel_9n.sh

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 9
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH -J mimo54l9n
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  fi
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${SCRATCH_ROOT}/images/e4b4805e816ada20.sqsh}"
ENV_ROOT="${ENV_ROOT:-${SCRATCH_ROOT}/envs/megatron_lm/01f0da7539da4b39}"

TRAIN_ITERS="${TRAIN_ITERS:-30}"
NUM_MICROBATCHES="${NUM_MICROBATCHES:-12}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

RUN_NAME="${RUN_NAME:-mimo54l-hel-9n-gbs${GLOBAL_BATCH_SIZE}}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}}"
TIMELINE_DIR="${TIMELINE_DIR:-${RUN_DIR}/timeline}"

mkdir -p \
  "${RUN_DIR}/logs/app" \
  "${RUN_DIR}/logs/torchrun" \
  "${RUN_DIR}/checkpoints" \
  "${RUN_DIR}/tensorboard" \
  "${RUN_DIR}/data_cache" \
  "${RUN_DIR}/tmp" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/home" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache" \
  "${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache" \
  "${SCRATCH_ROOT}/uv-cache/megatron_lm"

if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
  echo "ERROR: Cannot read CONTAINER_IMAGE=${CONTAINER_IMAGE}" >&2
  exit 1
fi
if [[ ! -d "${ENV_ROOT}/.venv" ]]; then
  echo "ERROR: Cannot find uv environment at ENV_ROOT=${ENV_ROOT}" >&2
  exit 1
fi

export SCRATCH_ROOT
export REPO_ROOT
export RUN_DIR
export OUTPUT_PATH="${RUN_DIR}"
export LOG_DIR="${RUN_DIR}/logs/app"
export APP_LOG_DIR="${RUN_DIR}/logs/app"
export TORCHRUN_LOG_DIR="${RUN_DIR}/logs/torchrun"
export CHECKPOINT_SAVE_PATH="${RUN_DIR}/checkpoints"
export CHECKPOINT_LOAD_PATH="${RUN_DIR}/checkpoints"
export CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
export TENSORBOARD_PATH="${RUN_DIR}/tensorboard"
export TB_DIR="${RUN_DIR}/tensorboard"
export DATA_CACHE_DIR="${RUN_DIR}/data_cache"
export ARTIFACT_MANIFEST="${RUN_DIR}/artifacts.json"
export TMPDIR="${RUN_DIR}/tmp"

export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache}"
export TRITON_CACHE_DIR_BASE="${TRITON_CACHE_DIR_BASE:-${RUN_DIR}/triton-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT=""
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm"
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# Runtime/perf settings carried over from the validated HEL run plus the v3
# multimodal recipe flags that matter for Transformer Engine and NCCL behavior.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE:-2097152}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_PROTO="${NCCL_PROTO:-simple}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-0}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_TRACE_CPP_STACK="${TORCH_NCCL_TRACE_CPP_STACK:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"

export TRAINING_STAGE="${TRAINING_STAGE:-stage2}"
export MODEL_PROVIDER="${MODEL_PROVIDER:-nemotron-moe-vlm-54l}"
export TRAIN_ITERS
export NUM_MICROBATCHES
export MICRO_BATCH_SIZE
export GLOBAL_BATCH_SIZE
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-2}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
export LOG_INTERVAL

export ENCODER_TP="${ENCODER_TP:-1}"
export ENCODER_CP="${ENCODER_CP:-1}"
export ENCODER_PP="${ENCODER_PP:-1}"
export ENCODER_DP="${ENCODER_DP:-8}"
export ENCODER_EP="${ENCODER_EP:-1}"
export LLM_ONLY="${LLM_ONLY:-0}"
export LLM_TP="${LLM_TP:-4}"
export LLM_CP="${LLM_CP:-1}"
export LLM_PP="${LLM_PP:-1}"
export LLM_DP="${LLM_DP:-16}"
export LLM_EP="${LLM_EP:-16}"
export LLM_EXPT_TP="${LLM_EXPT_TP:-1}"

export NUM_WORKERS="${NUM_WORKERS:-0}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
export PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-128}"
export MAX_SAMPLES_PER_SEQUENCE="${MAX_SAMPLES_PER_SEQUENCE:-100}"
export CHECK_HEL_PATHS="${CHECK_HEL_PATHS:-1}"

export ENABLE_EXPERIMENTAL="${ENABLE_EXPERIMENTAL:-1}"
export MOE_ROUTER_FORCE_LOAD_BALANCING="${MOE_ROUTER_FORCE_LOAD_BALANCING:-1}"

if [[ "${LLM_ONLY}" == "1" || "${LLM_ONLY}" == "true" ]]; then
  export LLM_OFFSET="${LLM_OFFSET:-0}"
  WORLD_SIZE=$((LLM_TP * LLM_CP * LLM_PP * LLM_DP))
else
  WORLD_SIZE=$((ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP + LLM_TP * LLM_CP * LLM_PP * LLM_DP))
  if [[ "${WORLD_SIZE}" -ne 72 ]]; then
    echo "ERROR: This 9-node sbatch expects 72 ranks, but layout computed WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 1
  fi
fi
if [[ -n "${SLURM_NTASKS:-}" && "${SLURM_NTASKS}" -ne "${WORLD_SIZE}" ]]; then
  echo "ERROR: SLURM_NTASKS=${SLURM_NTASKS}, expected ${WORLD_SIZE}" >&2
  exit 1
fi

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/scratch/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
if [[ "${REPO_ROOT}" != "${SCRATCH_ROOT}"/* ]]; then
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"
fi
if [[ -n "${CONTAINER_MOUNTS_EXTRA:-}" ]]; then
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${CONTAINER_MOUNTS_EXTRA}"
fi

TRAIN_LAUNCH_ARGS=()
if [[ "${ENABLE_TIMELINE:-1}" == "1" || "${ENABLE_TIMELINE:-1}" == "true" ]]; then
  TRAIN_LAUNCH_ARGS+=(--timeline-profile --timeline-dir "${TIMELINE_DIR}")
  TRAIN_LAUNCH_ARGS+=(--timeline-ranks "${TIMELINE_RANKS:-dp-replica}")
fi
if [[ "${TIMELINE_CUDA_EVENTS:-0}" == "1" || "${TIMELINE_CUDA_EVENTS:-0}" == "true" ]]; then
  TRAIN_LAUNCH_ARGS+=(--timeline-cuda-events)
fi
if [[ "${TIMELINE_NVTX:-0}" == "1" || "${TIMELINE_NVTX:-0}" == "true" ]]; then
  TRAIN_LAUNCH_ARGS+=(--timeline-nvtx)
fi
TRAIN_LAUNCH_ARGS+=("$@")

echo "=== HEL 54L heterogeneous MIMO sbatch ==="
echo "repo=${REPO_ROOT}"
echo "run_dir=${RUN_DIR}"
echo "container_image=${CONTAINER_IMAGE}"
echo "env_root=${ENV_ROOT}"
echo "world_size=${WORLD_SIZE}"
echo "gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES} train_iters=${TRAIN_ITERS}"
echo "llm_only=${LLM_ONLY}"
echo "layout=encoder(tp=${ENCODER_TP},dp=${ENCODER_DP}) llm(tp=${LLM_TP},dp=${LLM_DP},ep=${LLM_EP},etp=${LLM_EXPT_TP})"
echo "timeline=${ENABLE_TIMELINE:-1} timeline_dir=${TIMELINE_DIR}"
echo "================================================"

srun --kill-on-bad-exit=1 \
  --ntasks="${WORLD_SIZE}" \
  --ntasks-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc 'set -euo pipefail; cd "${REPO_ROOT}"; exec uv run --no-sync bash examples/mimo/scripts/run_hetero_nemotron_54l_hel_train.sh "$@"' \
  bash "${TRAIN_LAUNCH_ARGS[@]}"
