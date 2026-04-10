#!/bin/bash
# MIMO throughput benchmark — sbatch script.
#
# Usage:
#   sbatch --nodes=2 --time=00:10:00 scripts/sbatch/run.sh \
#       --worktree /path/to/.worktrees/nmfw-57-multinode-benchmarking \
#       --config benchmarks/mimo_throughput/configs/campaigns/exp001.yaml \
#       --output-dir /path/to/logs/mimo_campaigns/my_campaign/slurm
#
# Required args:
#   --worktree    Absolute path to the git worktree (sets PYTHONPATH + cd)
#   --config      Path to fully materialized experiment YAML
#   --output-dir  Base dir for job outputs (job dir created as <output-dir>/<JOB_ID>/)
#
# Optional env overrides:
#   MIMO_CONTAINER        Container sqsh image path
#   MIMO_CONTAINER_MOUNTS Container bind mounts

#SBATCH --job-name=mimo_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mem=0
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_nemorl

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
WORKTREE=""
CONFIG=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --worktree) WORKTREE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${WORKTREE}" || -z "${CONFIG}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: --worktree, --config, and --output-dir are required"
    exit 1
fi

# ---------------------------------------------------------------------------
# Container config (override via env vars)
# ---------------------------------------------------------------------------
MIMO_CONTAINER="${MIMO_CONTAINER:-/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_main_20260323.sqsh}"
MIMO_CONTAINER_MOUNTS="${MIMO_CONTAINER_MOUNTS:-/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1}"

# ---------------------------------------------------------------------------
# Create structured job directory
# ---------------------------------------------------------------------------
JOB_DIR="${OUTPUT_DIR}/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"/{results,workers}

# Redirect sbatch-level output to job dir
exec > >(tee "${JOB_DIR}/job.log") 2>&1

# ---------------------------------------------------------------------------
# NCCL / CUDA settings
# ---------------------------------------------------------------------------
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Multi-node rendezvous
# ---------------------------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500
NUM_GPUS=$((SLURM_NNODES * 8))

echo "============================================"
echo "  MIMO Throughput Benchmark"
echo "============================================"
echo "JOB_ID      : ${SLURM_JOB_ID}"
echo "JOB_DIR     : ${JOB_DIR}"
echo "NODES       : ${SLURM_NNODES}"
echo "GPUS        : ${NUM_GPUS}"
echo "MASTER      : ${MASTER_ADDR}:${MASTER_PORT}"
echo "WORKTREE    : ${WORKTREE}"
echo "CONFIG      : ${CONFIG}"
echo "CONTAINER   : ${MIMO_CONTAINER}"
echo "============================================"

# ---------------------------------------------------------------------------
# Build per-worker command
# srun launches 8 tasks/node. Each task = one GPU rank.
# SLURM_PROCID = global rank, SLURM_LOCALID = local rank
# ---------------------------------------------------------------------------
run_cmd="cd ${WORKTREE} && \
export MASTER_ADDR=${MASTER_ADDR} && \
export MASTER_PORT=${MASTER_PORT} && \
export WORLD_SIZE=\${SLURM_NTASKS} && \
export RANK=\${SLURM_PROCID} && \
export LOCAL_RANK=\${SLURM_LOCALID} && \
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
if [ \${SLURM_LOCALID} -eq 0 ]; then \
    uv sync 2>&1 && touch /tmp/uv_sync_done_\${SLURM_JOB_ID}; \
else \
    while [ ! -f /tmp/uv_sync_done_\${SLURM_JOB_ID} ]; do sleep 0.5; done; \
fi && \
uv run --no-sync python -u -m benchmarks.mimo_throughput.runner \
    --config ${CONFIG} \
    --results-dir ${JOB_DIR}/results \
    ${EXTRA_ARGS[*]:-}"

# Avoid SLURM cpus-per-task binding conflicts
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

srun -l \
    --no-container-mount-home \
    --container-image "${MIMO_CONTAINER}" \
    --container-mounts "${MIMO_CONTAINER_MOUNTS}" \
    --output="${JOB_DIR}/workers/rank_%t.out" \
    --error="${JOB_DIR}/workers/rank_%t.err" \
    bash -c "${run_cmd}"

EXIT_CODE=$?

echo "============================================"
echo "  Job complete (exit code: ${EXIT_CODE})"
echo "============================================"
echo "Results : ${JOB_DIR}/results/"
echo "Workers : ${JOB_DIR}/workers/"
echo "Job log : ${JOB_DIR}/job.log"

exit ${EXIT_CODE}
