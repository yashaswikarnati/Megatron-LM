#!/bin/bash
# Submit a one-node HEL 20L heterogeneous MIMO smoke run on the 90% text / 10% vision blend.
#
# Intended use from a Cog-synced nb-hel workspace:
#   sbatch examples/mimo/scripts/sbatch_hetero_nemotron_20l_hel_1n_text_vision.sh

#SBATCH -A nemotron_n4_pre
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH -J mimo20l1ntv
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

RUN_NAME="${RUN_NAME:-mimo20l-hel-1n-text-vision-10-90}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}}"

mkdir -p \
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
export TMPDIR="${RUN_DIR}/tmp"
export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUN_DIR}/triton-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache}"
export PYTHONPATH="${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT=""
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm"
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/examples/mimo/blend_files/text_omnicorpus_blend_10_90_hel.yaml}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff}"
export TRAIN_ITERS="${TRAIN_ITERS:-30}"
export NUM_MICROBATCHES="${NUM_MICROBATCHES:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
export PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-128}"
export MAX_SAMPLES_PER_SEQUENCE="${MAX_SAMPLES_PER_SEQUENCE:-100}"
export VERIFY_ENERGON="${VERIFY_ENERGON:-1}"
export ENABLE_EXPERIMENTAL="${ENABLE_EXPERIMENTAL:-1}"
export MOE_ROUTER_FORCE_LOAD_BALANCING="${MOE_ROUTER_FORCE_LOAD_BALANCING:-1}"

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/scratch/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
if [[ "${REPO_ROOT}" != "${SCRATCH_ROOT}"/* ]]; then
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"
fi
if [[ -n "${CONTAINER_MOUNTS_EXTRA:-}" ]]; then
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${CONTAINER_MOUNTS_EXTRA}"
fi

echo "=== HEL 20L heterogeneous MIMO sbatch ==="
echo "repo=${REPO_ROOT}"
echo "run_dir=${RUN_DIR}"
echo "container_image=${CONTAINER_IMAGE}"
echo "env_root=${ENV_ROOT}"
echo "data=${DATA_PATH}"
echo "tokenizer=${TOKENIZER_MODEL}"
echo "train_iters=${TRAIN_ITERS} microbatches=${NUM_MICROBATCHES}"
echo "================================================"

srun --kill-on-bad-exit=1 \
  --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc 'set -euo pipefail; cd "${REPO_ROOT}"; exec uv run --no-sync bash examples/mimo/scripts/run_hetero_nemotron_20l_energon_train.sh "$@"' \
  bash "$@"
