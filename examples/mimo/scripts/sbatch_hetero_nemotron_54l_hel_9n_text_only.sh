#!/bin/bash
# Submit the 9-node HEL 54L heterogeneous MIMO run on the text-only blend.
#
# Intended use from a Cog-synced nb-hel workspace:
#   sbatch examples/mimo/scripts/sbatch_hetero_nemotron_54l_hel_9n_text_only.sh

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 9
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH -J mimo54l9nt
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
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
export TRAIN_ITERS="${TRAIN_ITERS:-30}"
export NUM_MICROBATCHES="${NUM_MICROBATCHES:-12}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export RUN_NAME="${RUN_NAME:-mimo54l-hel-9n-text-only-gbs${GLOBAL_BATCH_SIZE}}"

exec bash "${REPO_ROOT}/examples/mimo/scripts/sbatch_hetero_nemotron_54l_hel_9n.sh" "$@"
