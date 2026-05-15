#!/bin/bash
# Submit the HEL 54L MIMO LLM-only baseline that matches the 9-node VLM run's LLM grid.
#
# Intended use from a Cog-synced nb-hel workspace:
#   sbatch examples/mimo/scripts/sbatch_mimo_nemotron_54l_hel_8n_text_only_llm.sh

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH -J mimo54l8nt
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

export REPO_ROOT
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/examples/mimo/blend_files/text_only_1t_hel.yaml}"
export RUN_NAME="${RUN_NAME:-mimo54l-hel-8n-text-only-llm-tp2-ep8-gbs192}"
export TRAIN_ITERS="${TRAIN_ITERS:-30}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export NUM_MICROBATCHES="${NUM_MICROBATCHES:-6}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"

export LLM_ONLY=1
export LLM_OFFSET=0
export LLM_TP="${LLM_TP:-2}"
export LLM_CP="${LLM_CP:-1}"
export LLM_PP="${LLM_PP:-1}"
export LLM_DP="${LLM_DP:-32}"
export LLM_EP="${LLM_EP:-8}"
export LLM_EXPT_TP="${LLM_EXPT_TP:-1}"

export ENABLE_TIMELINE="${ENABLE_TIMELINE:-0}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

exec bash "${REPO_ROOT}/examples/mimo/scripts/sbatch_hetero_nemotron_54l_hel_9n.sh" "$@"
