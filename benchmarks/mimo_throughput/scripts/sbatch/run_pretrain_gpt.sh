#!/bin/bash
# Pure LLM-only pretrain_gpt.py benchmark via sbatch.
#
# Usage:
#   sbatch --nodes=8 --time=00:10:00 --partition=batch --account=coreai_dlalgo_genai \
#       run_pretrain_gpt.sh --worktree /path/to/worktree

#SBATCH --job-name=gpt_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mem=0
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_genai

set -euo pipefail

WORKTREE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --worktree) WORKTREE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "${WORKTREE}" ]]; then
    echo "ERROR: --worktree required"
    exit 1
fi

MIMO_CONTAINER="${MIMO_CONTAINER:-/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_main_20260323.sqsh}"
MIMO_CONTAINER_MOUNTS="${MIMO_CONTAINER_MOUNTS:-/lustre/fsw/:/lustre/fsw/,/lustre/fs1:/lustre/fs1}"

JOB_DIR="${WORKTREE}/benchmarks/mimo_throughput/campaigns/gpt_70b_pp8/slurm/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/workers"

exec > >(tee "${JOB_DIR}/job.log") 2>&1

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500
NUM_GPUS=$((SLURM_NNODES * 8))

echo "============================================"
echo "  GPT 70B LLM-only Benchmark"
echo "  JOB_ID: ${SLURM_JOB_ID}"
echo "  NODES: ${SLURM_NNODES}, GPUS: ${NUM_GPUS}"
echo "  MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  WORKTREE: ${WORKTREE}"
echo "============================================"

# 70B config: 80 layers, hidden=8192, heads=64
# TP8/PP8 = 10 layers per stage
# DP1, dist opt, mbs=1, GBS=8
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
uv run --no-sync python -u pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 10 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --mock-data \
    --vocab-size 128000 \
    --log-interval 1 \
    --no-save-optim \
    --no-save-rng \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --transformer-impl transformer_engine"

unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

srun -l \
    --no-container-mount-home \
    --container-image "${MIMO_CONTAINER}" \
    --container-mounts "${MIMO_CONTAINER_MOUNTS}" \
    --output="${JOB_DIR}/workers/rank_%t.out" \
    --error="${JOB_DIR}/workers/rank_%t.err" \
    bash -c "${run_cmd}"

echo "Job complete (exit: $?)"
