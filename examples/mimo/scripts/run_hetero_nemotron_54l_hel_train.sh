#!/bin/bash
# Run non-colocated heterogeneous MIMO Nemotron6-MoE VLM 54L training on HEL data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_PROTO="${NCCL_PROTO:-simple}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-0}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
export PYTHONNOUSERSITE=1

if [[ -z "${LOCAL_RANK:-}" && -n "${SLURM_LOCALID:-}" ]]; then
  export LOCAL_RANK="${SLURM_LOCALID}"
fi
if [[ -z "${RANK:-}" && -n "${SLURM_PROCID:-}" ]]; then
  export RANK="${SLURM_PROCID}"
fi
if [[ -z "${WORLD_SIZE:-}" && -n "${SLURM_NTASKS:-}" ]]; then
  export WORLD_SIZE="${SLURM_NTASKS}"
fi
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  export MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"

TRAINING_STAGE="${TRAINING_STAGE:-stage2}"
MODEL_PROVIDER="${MODEL_PROVIDER:-nemotron-moe-vlm-54l}"
case "${TRAINING_STAGE}" in
  stage1|stage2|stage3)
    ;;
  *)
    echo "ERROR: Unknown TRAINING_STAGE='${TRAINING_STAGE}'. Use stage1, stage2, or stage3." >&2
    exit 1
    ;;
esac

TRAIN_ITERS="${TRAIN_ITERS:-100}"
NUM_MICROBATCHES="${NUM_MICROBATCHES:-12}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
ENCODER_TP="${ENCODER_TP:-1}"
ENCODER_CP="${ENCODER_CP:-1}"
ENCODER_PP="${ENCODER_PP:-1}"
ENCODER_DP="${ENCODER_DP:-8}"
ENCODER_EP="${ENCODER_EP:-1}"
LLM_TP="${LLM_TP:-4}"
LLM_CP="${LLM_CP:-1}"
LLM_PP="${LLM_PP:-1}"
LLM_DP="${LLM_DP:-64}"
LLM_EP="${LLM_EP:-16}"
LLM_EXPT_TP="${LLM_EXPT_TP:-1}"
LLM_ONLY="${LLM_ONLY:-0}"
ENABLE_EXPERIMENTAL="${ENABLE_EXPERIMENTAL:-1}"
MOE_ROUTER_FORCE_LOAD_BALANCING="${MOE_ROUTER_FORCE_LOAD_BALANCING:-0}"

ENCODER_SIZE=$((ENCODER_TP * ENCODER_CP * ENCODER_PP * ENCODER_DP))
LLM_SIZE=$((LLM_TP * LLM_CP * LLM_PP * LLM_DP))
if [[ "${LLM_ONLY}" == "1" || "${LLM_ONLY}" == "true" ]]; then
  ENCODER_SIZE=0
  LLM_OFFSET="${LLM_OFFSET:-0}"
else
  LLM_OFFSET="${LLM_OFFSET:-${ENCODER_SIZE}}"
fi
EXPECTED_WORLD_SIZE=$((ENCODER_SIZE + LLM_SIZE))
LLM_EXPT_DP="${LLM_EXPT_DP:-$((LLM_SIZE / (LLM_EXPT_TP * LLM_EP * LLM_PP)))}"

if [[ $((LLM_EXPT_TP * LLM_EP * LLM_PP * LLM_EXPT_DP)) -ne "${LLM_SIZE}" ]]; then
  echo "ERROR: LLM expert layout does not cover LLM ranks." >&2
  echo "       llm_size=${LLM_SIZE} etp=${LLM_EXPT_TP} ep=${LLM_EP} pp=${LLM_PP} edp=${LLM_EXPT_DP}" >&2
  exit 1
fi
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -ne "${EXPECTED_WORLD_SIZE}" ]]; then
  echo "ERROR: WORLD_SIZE=${WORLD_SIZE} but hetero layout requires ${EXPECTED_WORLD_SIZE}" >&2
  echo "       Submit with nodes*tasks_per_node=${EXPECTED_WORLD_SIZE}." >&2
  exit 1
fi

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NUM_MICROBATCHES * LLM_DP))}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-10}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-2e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
# Sample-based scheduler knobs (set to enable Sanjeev-style WSD). Empty = unused.
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-}"
LR_WSD_DECAY_SAMPLES="${LR_WSD_DECAY_SAMPLES:-}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-}"
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100}"
MAX_SAMPLES_PER_SEQUENCE="${MAX_SAMPLES_PER_SEQUENCE:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff}"
DATA_TEMPLATE="${DATA_PATH:-${REPO_ROOT}/examples/mimo/blend_files/text_omnicorpus_blend_10_90_hel.yaml}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/runs/mimo_54l_hel/${SLURM_JOB_ID:-local}}"
RESOLVED_CONFIG_DIR="${RESOLVED_CONFIG_DIR:-${RUN_DIR}/resolved_configs}"
DATA_TEMPLATE_BASENAME="$(basename "${DATA_TEMPLATE}")"
DATA_TRAIN="${DATA_TRAIN:-${RESOLVED_CONFIG_DIR}/${DATA_TEMPLATE_BASENAME%.yaml}.train.yaml}"
DATA_READY_FILE="${DATA_TRAIN}.ready"
RANK_ID="${RANK:-${SLURM_PROCID:-0}}"
DATA_READY_TIMEOUT="${DATA_READY_TIMEOUT:-600}"
TMPDIR="${TMPDIR:-${RUN_DIR}/tmp/rank-${RANK_ID}}"
mkdir -p "${TMPDIR}"
export TMPDIR
if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_BASE:-${RUN_DIR}/triton-cache}/rank-${RANK_ID}"
fi
mkdir -p "${TRITON_CACHE_DIR}"

if [[ ! -r "${DATA_TEMPLATE}" ]]; then
  echo "ERROR: Cannot read DATA_PATH template: ${DATA_TEMPLATE}" >&2
  exit 1
fi

if [[ "${RESOLVE_TRAIN_ONLY_CONFIG:-1}" == "1" ]]; then
  if [[ "${RANK_ID}" -eq 0 ]]; then
    mkdir -p "${RESOLVED_CONFIG_DIR}"
    rm -f "${DATA_READY_FILE}"
    DATA_TEMPLATE="${DATA_TEMPLATE}" \
    DATA_TRAIN="${DATA_TRAIN}" \
    REPO_ROOT="${REPO_ROOT}" \
    USER_HOME="${USER_HOME:-/home/${USER:-ykarnati}}" \
    MULTIMODAL_DATA_ROOT="${MULTIMODAL_DATA_ROOT:-/home/${USER:-ykarnati}/data/multimodal_data}" \
      "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["DATA_TEMPLATE"])
dst = Path(os.environ["DATA_TRAIN"])
text = src.read_text()
for key, value in {
    "__MEGATRON_ROOT__": os.environ["REPO_ROOT"],
    "__USER_HOME__": os.environ["USER_HOME"],
    "__MULTIMODAL_DATA_ROOT__": os.environ["MULTIMODAL_DATA_ROOT"],
}.items():
    text = text.replace(key, value)

train_only = []
for line in text.splitlines():
    if line.startswith("  val:") or line.startswith("  test:"):
        break
    train_only.append(line)
text = "\n".join(train_only) + "\n"

dst.parent.mkdir(parents=True, exist_ok=True)
tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
tmp.write_text(text)
tmp.replace(dst)
PY
    touch "${DATA_READY_FILE}"
  else
    waited=0
    until [[ -f "${DATA_READY_FILE}" ]]; do
      sleep 2
      waited=$((waited + 2))
      if [[ "${waited}" -gt "${DATA_READY_TIMEOUT}" ]]; then
        echo "ERROR: Timed out waiting for resolved data config: ${DATA_READY_FILE}" >&2
        exit 1
      fi
    done
  fi
else
  DATA_TRAIN="${DATA_TEMPLATE}"
fi

if [[ ! -r "${DATA_TRAIN}" ]]; then
  echo "ERROR: Cannot read resolved data config: ${DATA_TRAIN}" >&2
  exit 1
fi
if [[ ! -r "${TOKENIZER_MODEL}/tokenizer.json" ]]; then
  echo "ERROR: Cannot read tokenizer.json under TOKENIZER_MODEL=${TOKENIZER_MODEL}" >&2
  exit 1
fi

if [[ "${CHECK_HEL_PATHS:-1}" == "1" ]]; then
  TEXT_MCORE_JSON="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/rkarimimahab/workspace/blends/1T-phase1var-moresft.json"
  OMNICORPUS_SAMPLE="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/multimodal/datasets/OmniCorpus-CC-210M/webdataset/CC-MAIN-2013-20"
  if [[ ! -r "${TEXT_MCORE_JSON}" ]]; then
    echo "ERROR: Cannot read text MCore blend JSON: ${TEXT_MCORE_JSON}" >&2
    exit 1
  fi
  if [[ ! -d "${OMNICORPUS_SAMPLE}" ]]; then
    echo "ERROR: Cannot find OmniCorpus HEL sample directory: ${OMNICORPUS_SAMPLE}" >&2
    exit 1
  fi
fi

if [[ "${RANK_ID}" -eq 0 ]]; then
  echo "=== Hetero MIMO Nemotron6-MoE VLM 54L HEL training ==="
  echo "model_provider=${MODEL_PROVIDER}"
  echo "stage=${TRAINING_STAGE} train_iters=${TRAIN_ITERS} mbs=${MICRO_BATCH_SIZE} microbatches=${NUM_MICROBATCHES} gbs=${GLOBAL_BATCH_SIZE}"
  echo "llm_only=${LLM_ONLY}"
  echo "layout=encoder(tp=${ENCODER_TP},cp=${ENCODER_CP},pp=${ENCODER_PP},dp=${ENCODER_DP},ep=${ENCODER_EP}) llm(tp=${LLM_TP},cp=${LLM_CP},pp=${LLM_PP},dp=${LLM_DP},ep=${LLM_EP},etp=${LLM_EXPT_TP},edp=${LLM_EXPT_DP}) world=${EXPECTED_WORLD_SIZE}"
  echo "enable_experimental=${ENABLE_EXPERIMENTAL}"
  echo "moe_router_force_load_balancing=${MOE_ROUTER_FORCE_LOAD_BALANCING}"
  echo "moe_router_fusion=model-provider-default"
  echo "data=${DATA_TRAIN}"
  echo "tokenizer=${TOKENIZER_MODEL}"
  echo "run_dir=${RUN_DIR}"
  echo "=========================================================="
fi

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
if [[ "${LLM_ONLY}" == "1" || "${LLM_ONLY}" == "true" ]]; then
  MODEL_ARGS+=(--llm-only)
fi

CMD=(
  "${PYTHON_BIN}" -u examples/mimo/train_hetero.py
  --model-provider "${MODEL_PROVIDER}"
  --dataset-provider energon_multimodal
  --training-stage "${TRAINING_STAGE}"
  --encoder-tp "${ENCODER_TP}"
  --encoder-cp "${ENCODER_CP}"
  --encoder-pp "${ENCODER_PP}"
  --encoder-dp "${ENCODER_DP}"
  --encoder-ep "${ENCODER_EP}"
  --llm-offset "${LLM_OFFSET}"
  --llm-tp "${LLM_TP}"
  --llm-cp "${LLM_CP}"
  --llm-pp "${LLM_PP}"
  --llm-dp "${LLM_DP}"
  --llm-ep "${LLM_EP}"
  --llm-expt-tp "${LLM_EXPT_TP}"
  --llm-expt-dp "${LLM_EXPT_DP}"
  "${MODEL_ARGS[@]}"
  --vocab-size 131072
  --max-num-tiles 12
  --data-path "${DATA_TRAIN}"
  "${DATA_LOADER_ARGS[@]}"
  --tokenizer-model "${TOKENIZER_MODEL}"
  --tokenizer-prompt-format nemotron6-moe
  --image-token "<image>"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-microbatches "${NUM_MICROBATCHES}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --lr-warmup-iters "${LR_WARMUP_ITERS}"
  --lr-decay-iters "${LR_DECAY_ITERS}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad 1.0
  --no-overlap-grad-reduce
  --ddp-bucket-size 0
  --log-interval "${LOG_INTERVAL}"
  --train-iters "${TRAIN_ITERS}"
)
if [[ -n "${LR_WARMUP_SAMPLES}" ]]; then
  CMD+=(--lr-warmup-samples "${LR_WARMUP_SAMPLES}")
fi
if [[ -n "${LR_DECAY_SAMPLES}" ]]; then
  CMD+=(--lr-decay-samples "${LR_DECAY_SAMPLES}")
fi
if [[ -n "${LR_WSD_DECAY_SAMPLES}" ]]; then
  CMD+=(--lr-wsd-decay-samples "${LR_WSD_DECAY_SAMPLES}")
fi
if [[ -n "${LR_WSD_DECAY_STYLE}" ]]; then
  CMD+=(--lr-wsd-decay-style "${LR_WSD_DECAY_STYLE}")
fi
if [[ -n "${TRAIN_SAMPLES}" ]]; then
  CMD+=(--train-samples "${TRAIN_SAMPLES}")
fi
CMD+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

# Optional: disable Transparent Huge Pages for this process via libnothp.so
# (calls prctl(PR_SET_THP_DISABLE) in a constructor before main()).
# Gated by DISABLE_THP=1.  Builds the .so on demand in a shared location.
if [[ "${DISABLE_THP:-0}" == "1" ]]; then
  LIBNOTHP_SRC="${REPO_ROOT}/benchmarks/libnothp/disable_thp.c"
  LIBNOTHP_OUT_DIR="${SCRATCH_ROOT}/runtime/megatron_lm/libnothp"
  mkdir -p "${LIBNOTHP_OUT_DIR}"
  LIBNOTHP_SO="${LIBNOTHP_OUT_DIR}/libnothp.so"
  if [[ ! -f "${LIBNOTHP_SO}" || "${LIBNOTHP_SRC}" -nt "${LIBNOTHP_SO}" ]]; then
    if command -v gcc >/dev/null 2>&1; then
      gcc -shared -fPIC -O2 -o "${LIBNOTHP_SO}" "${LIBNOTHP_SRC}" \
        && echo "[disable_thp] built ${LIBNOTHP_SO}" \
        || echo "[disable_thp] WARNING: gcc build failed; THP not disabled"
    else
      echo "[disable_thp] WARNING: gcc not available; THP not disabled"
    fi
  fi
  if [[ -f "${LIBNOTHP_SO}" ]]; then
    export LD_PRELOAD="${LIBNOTHP_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "[disable_thp] rank ${RANK_ID}: LD_PRELOAD=${LD_PRELOAD}"
  fi
fi

# Optional: cuBLASLt logging to a per-rank file so logs from different ranks
# don't interleave.  Gated by CUBLASLT_LOG_LEVEL being set non-zero.
#   CUBLASLT_LOG_LEVEL=2     # 0=off, 1=error, 2=trace, 3=debug
# Output goes to per-rank file at ${RUN_DIR}/cublas/cublaslt-rank<N>.log
if [[ -n "${CUBLASLT_LOG_LEVEL:-}" && "${CUBLASLT_LOG_LEVEL}" != "0" ]]; then
  CUBLAS_LOG_DIR="${RUN_DIR}/cublas"
  mkdir -p "${CUBLAS_LOG_DIR}"
  export CUBLASLT_LOG_FILE="${CUBLAS_LOG_DIR}/cublaslt-rank$(printf '%05d' "${RANK_ID}").log"
  echo "[cublaslt] rank ${RANK_ID}: LOG_LEVEL=${CUBLASLT_LOG_LEVEL} LOG_FILE=${CUBLASLT_LOG_FILE}"
fi

# nsys profile wrapper for selected ranks.
# Set NSYS_RANKS to a comma-separated list of global ranks (e.g. "16,17,18,19,20,21,22,23")
# and NSYS_OUT_DIR to the per-job output dir.  Only those ranks invoke nsys profile.
# The in-code cudaProfilerStart/Stop (in loop.py, gated by --profile-step-start/end)
# bounds the actual capture window.
if [[ -n "${NSYS_RANKS:-}" && -n "${NSYS_OUT_DIR:-}" ]]; then
  if [[ ",${NSYS_RANKS}," == *",${RANK_ID},"* ]]; then
    mkdir -p "${NSYS_OUT_DIR}"
    NSYS_OUT_FILE="${NSYS_OUT_DIR}/rank$(printf '%05d' "${RANK_ID}")"
    echo "[nsys] rank ${RANK_ID}: wrapping with nsys profile -> ${NSYS_OUT_FILE}.nsys-rep"
    # Disable torch inductor background compile workers — nsys's process
    # instrumentation breaks subprocess Python init (sysconfig ImportError).
    export TORCHINDUCTOR_COMPILE_THREADS=1
    # Defaults enable CPU sampling, OS Runtime tracing, Python sampling, and
    # DWARF backtraces in addition to CUDA + NVTX so we capture the host side
    # too.  nsys in this container does not accept '--trace=nccl'; NCCL kernels
    # still show up under 'cuda' tracing.
    #
    # Env overrides:
    #   NSYS_TRACE       — -t value (default "cuda,nvtx,osrt")
    #   NSYS_SAMPLE      — -s value (default "cpu"; "none" disables CPU sampling)
    #   NSYS_EXTRA_ARGS  — appended verbatim
    #     (default: --python-sampling=true --python-sampling-frequency=2000
    #               --backtrace=dwarf)
    NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
    NSYS_SAMPLE="${NSYS_SAMPLE:-cpu}"
    NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:---python-sampling=true --python-sampling-frequency=2000 --backtrace=dwarf}"
    # Intentional word-splitting on NSYS_EXTRA_ARGS so each flag becomes a
    # separate argv entry.
    exec nsys profile \
      -s "${NSYS_SAMPLE}" \
      -t "${NSYS_TRACE}" \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      ${NSYS_EXTRA_ARGS} \
      -o "${NSYS_OUT_FILE}" \
      "${CMD[@]}"
  fi
fi

exec "${CMD[@]}"
