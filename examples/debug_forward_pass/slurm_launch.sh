#!/bin/bash
#SBATCH --job-name=debug-forward-pass
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# This script can submit itself so PARTITION, ACCOUNT, and TIME can be regular
# environment variables. Example:
#   PARTITION=batch ACCOUNT=my_account TIME=00:20:00 ./examples/debug_forward_pass/slurm_launch.sh
if [[ -z "${SLURM_JOB_ID:-}" && "${SUBMIT_SELF:-1}" == "1" ]]; then
    mkdir -p "${LOG_DIR:-logs}"

    sbatch_args=(
        --job-name="${JOB_NAME:-debug-forward-pass}"
        --nodes="${NNODES:-2}"
        --ntasks-per-node=1
        --gpus-per-node="${GPUS_PER_NODE:-8}"
        --cpus-per-task="${CPUS_PER_TASK:-16}"
        --time="${TIME:-00:30:00}"
        --output="${LOG_DIR:-logs}/%x-%j.out"
    )

    if [[ -n "${PARTITION:-}" ]]; then
        sbatch_args+=(--partition="${PARTITION}")
    fi
    if [[ -n "${ACCOUNT:-}" ]]; then
        sbatch_args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${QOS:-}" ]]; then
        sbatch_args+=(--qos="${QOS}")
    fi

    exec sbatch "${sbatch_args[@]}" "$0" "$@"
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${SLURM_NNODES:-${NNODES:-2}}"
TP="${TP:-8}"
PP="${PP:-2}"
SEQ_LENGTH="${SEQ_LENGTH:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SAVE_PATH="${SAVE_PATH:-debug_forward_pass_activations.pt}"
MASTER_PORT="${MASTER_PORT:-29500}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_ADDR MASTER_PORT GPUS_PER_NODE

echo "Job ${SLURM_JOB_ID:-manual}: nodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Saving activations to ${SAVE_PATH}"

srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK:-16}" \
    --gpu-bind=none \
    bash -lc '
        set -euo pipefail
        torchrun \
            --nnodes="${SLURM_NNODES}" \
            --nproc_per_node="${GPUS_PER_NODE}" \
            --node_rank="${SLURM_NODEID}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            examples/debug_forward_pass/debug_forward_pass.py \
                --tp "'"${TP}"'" \
                --pp "'"${PP}"'" \
                --seq-length "'"${SEQ_LENGTH}"'" \
                --micro-batch-size "'"${MICRO_BATCH_SIZE}"'" \
                --save "'"${SAVE_PATH}"'" \
                "$@"
    ' bash "$@"
