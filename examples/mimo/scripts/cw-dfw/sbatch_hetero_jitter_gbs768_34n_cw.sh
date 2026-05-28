#!/bin/bash
# cw-dfw variant of sbatch_hetero_jitter_gbs768_34n.sh — timeline-only 34n run
# for cross-cluster jitter comparison vs nb-hel 34n nsys (job 297782).
# 34 nodes (2n encoder DP=16 + 32n LLM TP=2 DP=128 EP=8), GBS=768, 100 iters.
# Differences vs the nb-hel jitter script:
#   * Account: coreai_dlalgo_genai (better fairshare on cw-dfw)
#   * NEMOTRON_CKPT defaults empty (sasatheesh iter_0001000 not on cw-dfw).
#     --load-nemotron-checkpoint appended only if NEMOTRON_CKPT non-empty.
#   * VISION_CKPT dropped (dead code in run_hetero_nemotron_54l_hel_train.sh).
#   * DATA_PATH defaults to text_only_1t_hel.yaml — cw-dfw OmniCorpus tars are
#     symlinks into the nvr portfolio that ykarnati can't read.
#   * Container mounts include /lustre/.../llmservice:/scratch/.../llmservice
#     plus a nested ${BLEND_SHIM_DIR} overlay at /scratch/.../llmservice/projects
#     /llmservice_fm_text so the McoreBlend JSON resolves at the canonical path.

#SBATCH -A coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH -N 34
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH -J mimo-jitter-gbs768-34n-cw
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
# users/rkarimimahab/workspace/blends/1T-phase1var-moresft.json
BLEND_SHIM_DIR="${BLEND_SHIM_DIR:-${SCRATCH_ROOT}/blend-shim}"

# Empty by default on cw-dfw (no resume checkpoint available).
NEMOTRON_CKPT="${NEMOTRON_CKPT:-}"

# Use cw-dfw production 90/10 blend variant (same as nb-hel except OmniCorpus
# root swapped to OmniCorpus-CC-210M-no-links; the original symlinks into the
# nvr portfolio aren't readable to ykarnati on cw-dfw).
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/examples/mimo/blend_files/text_omnicorpus_blend_10_90_hel_cw.yaml}"

NUM_DIST_OPT_INSTANCES=${NUM_DIST_OPT_INSTANCES:-1}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
DDP_NUM_BUCKETS=${DDP_NUM_BUCKETS:-8}
# CHECK_FOR_NAN_IN_GRAD=0 passes --no-check-for-nan-in-loss-and-grad. This
# disables per-bucket .norm().item() syncs inside start_grad_sync that fire
# from autograd hooks on mb=last bwd and may inflate mb=last bwd host wall.
CHECK_FOR_NAN_IN_GRAD=${CHECK_FOR_NAN_IN_GRAD:-1}
RUN_NAME="mimo-jitter-gbs768-34n-cw-PG${OVERLAP_PARAM_GATHER}-GR${OVERLAP_GRAD_REDUCE}-NDOI${NUM_DIST_OPT_INSTANCES}-B${DDP_NUM_BUCKETS}-NAN${CHECK_FOR_NAN_IN_GRAD}"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"

# ---- topology: TP=2 EP=8 LLM (32 nodes, DP=128) + TP=1 DP=16 encoder lane (2 nodes)
ENCODER_TP=1;  ENCODER_CP=1;  ENCODER_PP=1;  ENCODER_DP=16;  ENCODER_EP=1
LLM_TP=2;      LLM_CP=1;      LLM_PP=1;      LLM_DP=128;     LLM_EP=8;    LLM_EXPT_TP=1
LLM_ONLY=0

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=768
NUM_MICROBATCHES=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * LLM_DP) ))   # = 6
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
MOE_ROUTER_FORCE_LOAD_BALANCING=1
NUM_WORKERS=2
PACKING_BUFFER_SIZE=4
SHUFFLE_BUFFER_SIZE=100
MAX_SAMPLES_PER_SEQUENCE=100
CHECK_HEL_PATHS=1

WORLD_SIZE=$(( ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP \
             + LLM_TP * LLM_CP * LLM_PP * LLM_DP ))
[[ "${WORLD_SIZE}" -eq 272 ]] || { echo "ERROR: derived world_size=${WORLD_SIZE} (expected 272)" >&2; exit 1; }

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
export HETERO_DIST_TIMEOUT_MIN=25
export TORCH_NCCL_AVOID_RECORD_STREAMS=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export TRAINING_STAGE MODEL_PROVIDER ENABLE_EXPERIMENTAL MOE_ROUTER_FORCE_LOAD_BALANCING
export TRAIN_ITERS NUM_MICROBATCHES MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE LOG_INTERVAL
export ENCODER_TP ENCODER_CP ENCODER_PP ENCODER_DP ENCODER_EP
export LLM_TP LLM_CP LLM_PP LLM_DP LLM_EP LLM_EXPT_TP LLM_ONLY
export LR MIN_LR WEIGHT_DECAY LR_DECAY_STYLE
export LR_WARMUP_SAMPLES LR_DECAY_SAMPLES LR_WSD_DECAY_SAMPLES LR_WSD_DECAY_STYLE TRAIN_SAMPLES
export NUM_WORKERS PACKING_BUFFER_SIZE SHUFFLE_BUFFER_SIZE MAX_SAMPLES_PER_SEQUENCE CHECK_HEL_PATHS
export TOKENIZER_MODEL

TIMELINE=${TIMELINE:-1}  # opt-in JSONL timeline tracing for hetero loop (default ON)
TIMELINE_DIR="${RUN_DIR}/timeline"
mkdir -p "${TIMELINE_DIR}"

TRAIN_LAUNCH_ARGS=(
  --class-token-len 10
  --image-tag-type internvl
  --max-num-tiles 1
  --ddp-num-buckets "${DDP_NUM_BUCKETS}" --ddp-pad-buckets-for-high-nccl-busbw
  --correct-encoder-grad-for-partial-participation
  --seed 1234
  --save "${CHECKPOINT_SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
  --no-load-optim --no-load-rng
  --dynamic-resolution
  --num-distributed-optimizer-instances "${NUM_DIST_OPT_INSTANCES}"
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
if [[ "${CHECK_FOR_NAN_IN_GRAD}" != "1" ]]; then
  TRAIN_LAUNCH_ARGS+=( --no-check-for-nan-in-loss-and-grad )
fi
if [[ "${TIMELINE}" == "1" ]]; then
  TRAIN_LAUNCH_ARGS+=(
    --timeline-profile
    --timeline-dir "${TIMELINE_DIR}"
    --timeline-ranks all
  )
  # --timeline-cuda-events intentionally OFF — recorder syncs per iter and
  # breaks overlap-grad-reduce / overlap-param-gather at production scale.
fi

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT}"
CONTAINER_MOUNTS+=",/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice"
CONTAINER_MOUNTS+=",/lustre/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
CONTAINER_MOUNTS+=",${BLEND_SHIM_DIR}:/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text"
[[ "${REPO_ROOT}" == "${SCRATCH_ROOT}"/* ]] || CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"

echo "=== hetero JITTER GBS=768 34n cw-dfw (${TRAIN_ITERS} iters, PG=${OVERLAP_PARAM_GATHER} GR=${OVERLAP_GRAD_REDUCE} FLB=${MOE_ROUTER_FORCE_LOAD_BALANCING} NDOI=${NUM_DIST_OPT_INSTANCES} BUCKETS=${DDP_NUM_BUCKETS} NAN=${CHECK_FOR_NAN_IN_GRAD}) ==="
echo "repo=${REPO_ROOT} run_dir=${RUN_DIR}"
echo "world_size=${WORLD_SIZE} gbs=${GLOBAL_BATCH_SIZE} microbatches=${NUM_MICROBATCHES}"
echo "layout: encoder(dp=${ENCODER_DP}) llm(tp=${LLM_TP},dp=${LLM_DP},ep=${LLM_EP})"
echo "ckpt=${NEMOTRON_CKPT:-<random-init>}"
echo "overlap_param_gather: ${OVERLAP_PARAM_GATHER}  overlap_grad_reduce: ${OVERLAP_GRAD_REDUCE}"
echo "blend_shim=${BLEND_SHIM_DIR}"
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
