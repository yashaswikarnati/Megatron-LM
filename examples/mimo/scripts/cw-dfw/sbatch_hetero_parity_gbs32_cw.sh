#!/bin/bash
# cw-dfw variant of sbatch_hetero_parity_gbs32.sh — pipe-flush run on cw-dfw cluster.
# Differences vs the nb-hel parity script:
#   * NEMOTRON_CKPT defaults empty (sasatheesh's iter_0001000 is not on cw-dfw).
#     --load-nemotron-checkpoint is appended only if NEMOTRON_CKPT is non-empty.
#   * VISION_CKPT is dropped — not consumed in run_hetero_nemotron_54l_hel_train.sh,
#     and not staged on cw-dfw.
#   * MOE_ROUTER_FORCE_LOAD_BALANCING defaults to 1 (random LLM init needs forced LB
#     to keep MoE routing stable).
#   * /scratch/fsw/portfolios/llmservice does not exist on cw-dfw host; bind-mounted
#     from /lustre/fsw/portfolios/llmservice so the blend's /scratch/... paths
#     resolve inside the container.
#   * /scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/...
#     /1T-phase1var-moresft.json is overlaid via a nested bind-mount of
#     ${BLEND_SHIM_DIR} at /scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text
#     (more-specific mount overrides the broader llmservice mount).
# Same topology as the original: 3 nodes (1n encoder DP=8 + 2n LLM TP=2 DP=8 EP=16),
# GBS=32, 100 iters.

#SBATCH -A coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH -N 3
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:50:00
#SBATCH -J mimo-parity-gbs32-cw
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/examples/mimo" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi

SCRATCH_ROOT=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch
CONTAINER_IMAGE="${HETERO_CONTAINER_IMAGE:-${SCRATCH_ROOT}/images/m_lm_energon_0506.sqsh}"
export HETERO_SKIP_UV="${HETERO_SKIP_UV:-1}"
ENV_ROOT="${SCRATCH_ROOT}/envs/megatron_lm/01f0da7539da4b39"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"

# Bind-mount shim: holds the unmodified McoreBlend JSON at
# users/rkarimimahab/workspace/blends/1T-phase1var-moresft.json so that the
# /scratch/.../llmservice_fm_text/... path inside the container resolves.
BLEND_SHIM_DIR="${BLEND_SHIM_DIR:-${SCRATCH_ROOT}/blend-shim}"

# Empty by default on cw-dfw (no resume checkpoint available).
NEMOTRON_CKPT="${NEMOTRON_CKPT:-}"

# Use cw-dfw production blend variant (90% 1T text + 10% OmniCorpus). Same
# blend as nb-hel, except OmniCorpus paths point at OmniCorpus-CC-210M-no-links
# (real files; original OmniCorpus-CC-210M tars are symlinks into the nvr
# portfolio which ykarnati can't read).
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/examples/mimo/blend_files/text_omnicorpus_blend_10_90_hel_cw.yaml}"

OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
MOE_ROUTER_FORCE_LOAD_BALANCING=${MOE_ROUTER_FORCE_LOAD_BALANCING:-1}
RUN_NAME="mimo-parity-gbs32-cw-PG${OVERLAP_PARAM_GATHER}-GR${OVERLAP_GRAD_REDUCE}-FLB${MOE_ROUTER_FORCE_LOAD_BALANCING}"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"

# ---- topology: TP=2 EP=16 LLM + TP=1 DP=8 encoder lane ----------------------
ENCODER_TP=1; ENCODER_CP=1; ENCODER_PP=1; ENCODER_DP=8; ENCODER_EP=1
LLM_TP=2;     LLM_CP=1;     LLM_PP=1;     LLM_DP=8;    LLM_EP=16;   LLM_EXPT_TP=1
LLM_ONLY=0

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
NUM_MICROBATCHES=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * LLM_DP) ))   # = 4
TRAIN_ITERS=100
LOG_INTERVAL=1
SAVE_INTERVAL=99999999

LR=1.2e-3
MIN_LR=1.2e-5
WEIGHT_DECAY=0.1
LR_DECAY_STYLE=WSD
LR_WARMUP_SAMPLES=0
LR_DECAY_SAMPLES=121046313
LR_WSD_DECAY_SAMPLES=1
LR_WSD_DECAY_STYLE=minus_sqrt
TRAIN_SAMPLES=$(( TRAIN_ITERS * GLOBAL_BATCH_SIZE ))

TRAINING_STAGE=stage2
MODEL_PROVIDER=nemotron-moe-vlm-54l
ENABLE_EXPERIMENTAL=1
NUM_WORKERS=2
PACKING_BUFFER_SIZE=4
SHUFFLE_BUFFER_SIZE=100
MAX_SAMPLES_PER_SEQUENCE=100
CHECK_HEL_PATHS=1

WORLD_SIZE=$(( ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP \
             + LLM_TP * LLM_CP * LLM_PP * LLM_DP ))
[[ "${WORLD_SIZE}" -eq 24 ]] || { echo "ERROR: derived world_size=${WORLD_SIZE} (expected 24)" >&2; exit 1; }

mkdir -p "${RUN_DIR}/logs/app" "${RUN_DIR}/logs/torchrun" "${RUN_DIR}/checkpoints" \
         "${RUN_DIR}/tensorboard" "${RUN_DIR}/data_cache" "${RUN_DIR}/tmp"

export REPO_ROOT RUN_DIR SCRATCH_ROOT
export OUTPUT_PATH="${RUN_DIR}" LOG_DIR="${RUN_DIR}/logs/app" APP_LOG_DIR="${RUN_DIR}/logs/app"
export TORCHRUN_LOG_DIR="${RUN_DIR}/logs/torchrun"
export CHECKPOINT_SAVE_PATH="${RUN_DIR}/checkpoints" CHECKPOINT_LOAD_PATH="${NEMOTRON_CKPT}"
export CHECKPOINT_DIR="${RUN_DIR}/checkpoints" TENSORBOARD_PATH="${RUN_DIR}/tensorboard" TB_DIR="${RUN_DIR}/tensorboard"
export DATA_CACHE_DIR="${RUN_DIR}/data_cache"
export TMPDIR="/tmp"

export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache"
export TRITON_CACHE_DIR_BASE="${RUN_DIR}/triton-cache"
export CUDA_CACHE_PATH="${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache"
export PYTHONPATH="${REPO_ROOT}" PYTHONNOUSERSITE=1 PIP_CONSTRAINT=""
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm" UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16 NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 NCCL_PROTO=simple NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export TRAINING_STAGE MODEL_PROVIDER ENABLE_EXPERIMENTAL MOE_ROUTER_FORCE_LOAD_BALANCING
export TRAIN_ITERS NUM_MICROBATCHES MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE LOG_INTERVAL
export ENCODER_TP ENCODER_CP ENCODER_PP ENCODER_DP ENCODER_EP
export LLM_TP LLM_CP LLM_PP LLM_DP LLM_EP LLM_EXPT_TP LLM_ONLY
export LR MIN_LR WEIGHT_DECAY LR_DECAY_STYLE
export LR_WARMUP_SAMPLES LR_DECAY_SAMPLES LR_WSD_DECAY_SAMPLES LR_WSD_DECAY_STYLE TRAIN_SAMPLES
export NUM_WORKERS PACKING_BUFFER_SIZE SHUFFLE_BUFFER_SIZE MAX_SAMPLES_PER_SEQUENCE CHECK_HEL_PATHS
export TOKENIZER_MODEL

TRAIN_LAUNCH_ARGS=(
  --class-token-len 10
  --image-tag-type internvl
  --max-num-tiles 1
  --ddp-num-buckets 8 --ddp-pad-buckets-for-high-nccl-busbw
  --correct-encoder-grad-for-partial-participation
  --seed 1234
  --save "${CHECKPOINT_SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
  --no-load-optim --no-load-rng
  --dynamic-resolution
  --tensorboard-dir "${RUN_DIR}/tensorboard"
)
if [[ -n "${NEMOTRON_CKPT}" ]]; then
  TRAIN_LAUNCH_ARGS+=(--load-nemotron-checkpoint "${NEMOTRON_CKPT}")
fi
if [[ "${OVERLAP_PARAM_GATHER}" == "1" ]]; then
  TRAIN_LAUNCH_ARGS+=( --overlap-param-gather )
fi
if [[ "${OVERLAP_GRAD_REDUCE}" == "1" ]]; then
  TRAIN_LAUNCH_ARGS+=( --overlap-grad-reduce )
fi

# Timeline tracing — all-rank by default for small-scale 3-node debug runs.
TIMELINE=${TIMELINE:-1}
TIMELINE_DIR="${RUN_DIR}/timeline"
mkdir -p "${TIMELINE_DIR}"
if [[ "${TIMELINE}" == "1" ]]; then
  TRAIN_LAUNCH_ARGS+=(
    --timeline-profile
    --timeline-dir "${TIMELINE_DIR}"
    --timeline-ranks all
  )
fi

# Container mounts: bind /lustre/.../llmservice at the /scratch/... path the
# blend expects, plus the shim dir for the McoreBlend JSON.
CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT}"
CONTAINER_MOUNTS+=",/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice"
CONTAINER_MOUNTS+=",/lustre/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
CONTAINER_MOUNTS+=",${BLEND_SHIM_DIR}:/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text"
[[ "${REPO_ROOT}" == "${SCRATCH_ROOT}"/* ]] || CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"

echo "=== hetero parity GBS=32 cw-dfw (${TRAIN_ITERS} iters, PG=${OVERLAP_PARAM_GATHER} GR=${OVERLAP_GRAD_REDUCE} FLB=${MOE_ROUTER_FORCE_LOAD_BALANCING}) ==="
echo "repo=${REPO_ROOT} run_dir=${RUN_DIR}"
echo "world_size=${WORLD_SIZE} gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES}"
echo "layout: encoder(dp=${ENCODER_DP}) llm(tp=${LLM_TP},dp=${LLM_DP},ep=${LLM_EP})"
echo "ckpt=${NEMOTRON_CKPT:-<random-init>}"
echo "blend_shim=${BLEND_SHIM_DIR}"
echo "=================================================="

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
