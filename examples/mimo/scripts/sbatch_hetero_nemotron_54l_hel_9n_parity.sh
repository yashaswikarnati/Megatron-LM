#!/bin/bash
# Parity reproduction of Sanjeev's pre-vlm-05 VLM training, scaled to 9 HEL
# nodes. Every training value is pinned inline in this file; nothing falls back
# to user-set env vars. GBS is canonical; NUM_MICROBATCHES is derived from
# (GBS / (MBS * LLM_DP)).
#
# Submit from the worktree root:
#   sbatch examples/mimo/scripts/sbatch_hetero_nemotron_54l_hel_9n_parity.sh

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 9
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH -J mimo54l9n-parity
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Repo + cluster paths (pinned)
# -----------------------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

SCRATCH_ROOT=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch
CONTAINER_IMAGE="${SCRATCH_ROOT}/images/megatron-venv-baked-206674.sqsh"
ENV_ROOT="${SCRATCH_ROOT}/envs/megatron_lm/01f0da7539da4b39"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"
VISION_CKPT="${SCRATCH_ROOT}/encoders/post-c-radio-omni"

RUN_NAME="mimo54l-hel-9n-parity"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"

# -----------------------------------------------------------------------------
# Parallelism (pinned — Sanjeev's TP=2 EP=16 with our 9-node shape)
# Encoder grid: 8 ranks. LLM grid: 64 ranks. Total: 72.
# -----------------------------------------------------------------------------
ENCODER_TP=1
ENCODER_CP=1
ENCODER_PP=1
ENCODER_DP=8
ENCODER_EP=1
LLM_TP=2
LLM_CP=1
LLM_PP=1
LLM_DP=32
LLM_EP=16
LLM_EXPT_TP=1
LLM_ONLY=0

# -----------------------------------------------------------------------------
# Batch — GBS canonical, NUM_MICROBATCHES derived
# -----------------------------------------------------------------------------
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=768                                              # Sanjeev's
NUM_MICROBATCHES=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * LLM_DP) ))   # = 24

# -----------------------------------------------------------------------------
# Sanjeev-faithful optimizer + WSD schedule (sample-based)
# -----------------------------------------------------------------------------
LR=1.2e-3
MIN_LR=1.2e-5
WEIGHT_DECAY=0.1
LR_DECAY_STYLE=WSD
LR_WARMUP_SAMPLES=1024000
LR_DECAY_SAMPLES=35597094
LR_WSD_DECAY_SAMPLES=5493164
LR_WSD_DECAY_STYLE=minus_sqrt
TRAIN_SAMPLES=36621094

# validate_args derives args.train_iters = ceil(TRAIN_SAMPLES / GBS) when
# --train-samples is set, so this sentinel is only used pre-derivation.
TRAIN_ITERS=$(( (TRAIN_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))

# -----------------------------------------------------------------------------
# Logging / checkpointing
# -----------------------------------------------------------------------------
LOG_INTERVAL=100
SAVE_INTERVAL=1000

# -----------------------------------------------------------------------------
# Provider / data knobs
# -----------------------------------------------------------------------------
TRAINING_STAGE=stage2
MODEL_PROVIDER=nemotron-moe-vlm-54l
ENABLE_EXPERIMENTAL=1
MOE_ROUTER_FORCE_LOAD_BALANCING=0
NUM_WORKERS=2
PACKING_BUFFER_SIZE=128
SHUFFLE_BUFFER_SIZE=100
MAX_SAMPLES_PER_SEQUENCE=100
CHECK_HEL_PATHS=1

# -----------------------------------------------------------------------------
# Sanity checks (fail fast before srun)
# -----------------------------------------------------------------------------
[[ -r "${CONTAINER_IMAGE}" ]]   || { echo "ERROR: missing ${CONTAINER_IMAGE}" >&2; exit 1; }
[[ -d "${ENV_ROOT}/.venv" ]]    || { echo "ERROR: missing ${ENV_ROOT}/.venv" >&2; exit 1; }
[[ -d "${TOKENIZER_MODEL}" ]]   || { echo "ERROR: missing ${TOKENIZER_MODEL}" >&2; exit 1; }
[[ -d "${VISION_CKPT}" ]]       || { echo "ERROR: missing ${VISION_CKPT}" >&2; exit 1; }

WORLD_SIZE=$(( ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP \
             + LLM_TP * LLM_CP * LLM_PP * LLM_DP ))
[[ "${WORLD_SIZE}" -eq 72 ]] || { echo "ERROR: derived world_size=${WORLD_SIZE} (expected 72 for 9n)" >&2; exit 1; }
if [[ -n "${SLURM_NTASKS:-}" && "${SLURM_NTASKS}" -ne "${WORLD_SIZE}" ]]; then
  echo "ERROR: SLURM_NTASKS=${SLURM_NTASKS}, expected ${WORLD_SIZE}" >&2
  exit 1
fi

# GBS / NUM_MICROBATCHES consistency
if (( MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP != GLOBAL_BATCH_SIZE )); then
  echo "ERROR: GBS=${GLOBAL_BATCH_SIZE} != MBS*NUM_MICROBATCHES*LLM_DP=$((MICRO_BATCH_SIZE*NUM_MICROBATCHES*LLM_DP))" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Exports consumed by run_hetero_nemotron_54l_hel_train.sh
# -----------------------------------------------------------------------------
export REPO_ROOT RUN_DIR SCRATCH_ROOT
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
export TMPDIR="${RUN_DIR}/tmp"

export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache"
export TRITON_CACHE_DIR_BASE="${RUN_DIR}/triton-cache"
export CUDA_CACHE_PATH="${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache"
export TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT=""
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm"
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# Runtime / NCCL / TE env
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TORCH_FR_BUFFER_SIZE=1048576
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Training-knob exports picked up by run_hetero
export TRAINING_STAGE MODEL_PROVIDER ENABLE_EXPERIMENTAL MOE_ROUTER_FORCE_LOAD_BALANCING
export TRAIN_ITERS NUM_MICROBATCHES MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE LOG_INTERVAL
export ENCODER_TP ENCODER_CP ENCODER_PP ENCODER_DP ENCODER_EP
export LLM_TP LLM_CP LLM_PP LLM_DP LLM_EP LLM_EXPT_TP LLM_ONLY
export LR MIN_LR WEIGHT_DECAY LR_DECAY_STYLE
export LR_WARMUP_SAMPLES LR_DECAY_SAMPLES LR_WSD_DECAY_SAMPLES LR_WSD_DECAY_STYLE TRAIN_SAMPLES
export NUM_WORKERS PACKING_BUFFER_SIZE SHUFFLE_BUFFER_SIZE MAX_SAMPLES_PER_SEQUENCE CHECK_HEL_PATHS
export TOKENIZER_MODEL
export VISION_CKPT

# -----------------------------------------------------------------------------
# Extra CLI args appended to train_hetero.py via run_hetero "$@".
# Argparse processes in order; these override the hardcoded values inside
# run_hetero_nemotron_54l_hel_train.sh:CMD[].
# -----------------------------------------------------------------------------
TRAIN_LAUNCH_ARGS=(
  --class-token-len 10            # RADIO vit_huge_patch16_224 has 1 CLS + 9 registers
  --image-tag-type internvl       # Sanjeev wraps <image> with internvl markers
  --max-num-tiles 1               # single-image dynamic-resolution; no LLaVA-NeXT tile grid
  --overlap-grad-reduce           # BooleanOptionalAction; later value wins over --no-overlap-grad-reduce
  --overlap-param-gather          # distributed-optimizer param all-gather overlap
  --ddp-num-buckets 8             # match Sanjeev; mutually exclusive with --ddp-bucket-size>0
  --ddp-pad-buckets-for-high-nccl-busbw   # pad to 2^16 for NCCL busbw at large DP
  --seed 1234                     # Sanjeev's value
  --save "${CHECKPOINT_SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
  # --load-vision-from "${VISION_CKPT}"   # TODO: enable once PR_load_vision_from lands
)

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/scratch/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
if [[ "${REPO_ROOT}" != "${SCRATCH_ROOT}"/* ]]; then
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"
fi

# -----------------------------------------------------------------------------
# Summary banner
# -----------------------------------------------------------------------------
echo "=== HEL 54L hetero MIMO parity sbatch (9n, Sanjeev recipe) ==="
echo "repo=${REPO_ROOT}"
echo "run_dir=${RUN_DIR}"
echo "container_image=${CONTAINER_IMAGE}"
echo "world_size=${WORLD_SIZE}"
echo "mbs=${MICRO_BATCH_SIZE} gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES}  (gbs/(mbs*llm_dp))"
echo "layout=encoder(tp=${ENCODER_TP},dp=${ENCODER_DP}) llm(tp=${LLM_TP},dp=${LLM_DP},ep=${LLM_EP})"
echo "lr=${LR} min_lr=${MIN_LR} wd=${WEIGHT_DECAY} schedule=${LR_DECAY_STYLE}/${LR_WSD_DECAY_STYLE}"
echo "train_samples=${TRAIN_SAMPLES} train_iters=${TRAIN_ITERS}"
echo "log_interval=${LOG_INTERVAL} save_interval=${SAVE_INTERVAL}"
echo "tokenizer=${TOKENIZER_MODEL}"
echo "vision_ckpt=${VISION_CKPT}"
echo "extra_args=${TRAIN_LAUNCH_ARGS[*]}"
echo "==============================================================="

srun --kill-on-bad-exit=1 \
  --ntasks="${WORLD_SIZE}" \
  --ntasks-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc 'set -euo pipefail; cd "${REPO_ROOT}"; exec uv run --no-sync bash examples/mimo/scripts/run_hetero_nemotron_54l_hel_train.sh "$@"' \
  bash "${TRAIN_LAUNCH_ARGS[@]}"
