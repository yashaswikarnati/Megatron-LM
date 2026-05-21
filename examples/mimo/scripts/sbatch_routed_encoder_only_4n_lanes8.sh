#!/bin/bash
# Standalone encoder-only run: 4 nodes, 32 GPUs, TP=1 DP=32 encoder.
# Forces virtual llm_dp=256 so lanes_per_encoder=8 — exact encoder-side
# fan-out used in the 68n EP=8 production scaling run. No LLM, no bridge,
# no DDP overlap; isolates encoder data-loading + forward at 32-rank scale.

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH -J routed-enc-only-4n-lanes8
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
ENV_ROOT="${SCRATCH_ROOT}/envs/megatron_lm/01f0da7539da4b39"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"
DATA_TEMPLATE="${REPO_ROOT}/examples/mimo/blend_files/text_omnicorpus_blend_10_90_hel.yaml"

RUN_NAME="routed-enc-only-4n-lanes8"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"

ENCODER_DP=32
LLM_DP=256               # virtual; gives lanes_per_encoder=8 (matches 68n EP=8 production fan-out)
N_STEPS=${N_STEPS:-200}
PARITY=${PARITY:-0}      # off — production behavior, not per-step parity overhead

mkdir -p "${RUN_DIR}" "${RUN_DIR}/tmp" "${RUN_DIR}/resolved_configs"

# Resolve __MEGATRON_ROOT__/__USER_HOME__/__MULTIMODAL_DATA_ROOT__ placeholders
# in the blend template (same substitutions as run_hetero_nemotron_54l_hel_train.sh).
DATA_TRAIN="${RUN_DIR}/resolved_configs/$(basename "${DATA_TEMPLATE}" .yaml).train.yaml"
USER_HOME="${USER_HOME:-/home/${USER:-ykarnati}}"
MULTIMODAL_DATA_ROOT="${MULTIMODAL_DATA_ROOT:-/home/${USER:-ykarnati}/data/multimodal_data}"
DATA_TEMPLATE="${DATA_TEMPLATE}" DATA_TRAIN="${DATA_TRAIN}" REPO_ROOT="${REPO_ROOT}" \
  USER_HOME="${USER_HOME}" MULTIMODAL_DATA_ROOT="${MULTIMODAL_DATA_ROOT}" \
  python - <<'PY'
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
dst.write_text(text)
PY
echo "resolved data template -> ${DATA_TRAIN}"

# Multi-node rendezvous: pick rank-0 node's hostname as MASTER_ADDR (constant
# across all srun tasks). SLURMD_NODENAME is per-task and would point each
# rank at its own node — broken on >1 node.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29501}"

export REPO_ROOT RUN_DIR SCRATCH_ROOT TOKENIZER_MODEL DATA_TRAIN
export TMPDIR="/tmp"
export HOME="${SCRATCH_ROOT}/runtime/megatron_lm/home"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/cache"
export XDG_DATA_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/data"
export XDG_STATE_HOME="${SCRATCH_ROOT}/runtime/megatron_lm/xdg/state"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/runtime/megatron_lm/torchinductor-cache"
export CUDA_CACHE_PATH="${SCRATCH_ROOT}/runtime/megatron_lm/cuda-cache"
export PYTHONPATH="${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT=""
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache/megatron_lm"
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
export NCCL_NVLS_ENABLE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export ENCODER_DP LLM_DP N_STEPS PARITY

CONTAINER_MOUNTS="${SCRATCH_ROOT}:${SCRATCH_ROOT},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/scratch/fsw/portfolios/llmservice:/scratch/fsw/portfolios/llmservice"
[[ "${REPO_ROOT}" == "${SCRATCH_ROOT}"/* ]] || CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${REPO_ROOT}:${REPO_ROOT}"

echo "=== routed-encoder-only 4 nodes 32 GPUs (TP=1 DP=${ENCODER_DP}, virtual llm_dp=${LLM_DP}, lanes=8 — mirrors 68n encoder) ==="
echo "repo=${REPO_ROOT} run_dir=${RUN_DIR}"
echo "master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
echo "data=${DATA_TEMPLATE}"
echo "tokenizer=${TOKENIZER_MODEL}"
echo "==========================================================================="

srun --kill-on-bad-exit=1 \
  --ntasks=32 \
  --ntasks-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --no-container-mount-home \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc '
    set -euo pipefail
    cd "${REPO_ROOT}"
    export RANK="${SLURM_PROCID}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    exec python -u examples/mimo/scripts/test_routed_encoder_only.py \
      --model-provider nemotron-moe-vlm-54l \
      --training-stage stage2 \
      --data-path "${DATA_TRAIN}" \
      --tokenizer-model "${TOKENIZER_MODEL}" \
      --encoder-dp "${ENCODER_DP}" \
      --llm-dp "${LLM_DP}" \
      --max-num-tiles 1 \
      --class-token-len 10 \
      --image-tag-type internvl \
      --freeze-vit \
      --n-steps "${N_STEPS}" \
      $( [[ "${PARITY}" == "1" ]] && echo "--parity" )
  '
