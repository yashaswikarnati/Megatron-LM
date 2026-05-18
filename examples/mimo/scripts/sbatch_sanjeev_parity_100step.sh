#!/bin/bash
# 150-step train-loss parity run — Sanjeev-side recipe.
# Thin wrapper around the pretrain script with env overrides for parity:
# 2 nodes, GBS=8, --calculate-per-token-loss enabled.
#
# Submit from anywhere; this script cds into the reference repo on the
# cluster and shells out to its pretrain entry point.

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH -J sanj-parity-100step
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

SCRATCH_ROOT=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch
SANJEEV_REPO="${SANJEEV_REPO:-${SCRATCH_ROOT}/sanjeev-repos/megatron-lm-clean}"
CONTAINER_IMAGE="${SANJEEV_CONTAINER_IMAGE:-${SCRATCH_ROOT}/images/m_lm_energon_0506.sqsh}"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"

# Ckpt to resume from. Flip CKPT_RUN_ROOT / CKPT_STEP for a different ckpt.
CKPT_RUN_ROOT="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/workspace/output/3b_nano_vlm_sota_mtp2_90t10v_post_c_radio_omni_96n_tp2_ep16_selective_300b_20260511"
CKPT_STEP="${CKPT_STEP:-1000}"

RUN_NAME="sanj-parity-100step"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/save" "${RUN_DIR}/tb"

# ---- Overrides passed to the pretrain script --------------------------------
# Topology: TP=2 EP=16. EDP/DP fall out from world_size; we constrain TP/EP/MBS/GBS.
export VISION_MODEL_TYPE=radio
export RADIO_ENCODER_DIR=post-c-radio-omni
export TP=2
export EP=16
export NUM_EXPERTS=128
export MOE_ROUTER_TOPK=6
export MBS=1
export GBS=8
export NUM_WORKERS=2
export PACKING_BUFFER_SIZE=4  # Match hetero side. Larger buffers reorder samples differently → per-rank batches diverge.
export SEQ_LEN=8192
export DECODER_SEQ_LEN=8192
# 150 optimizer steps after the resume + the ckpt's iter_1000 offset.
export TRAIN_SAMPLES=$(( 150 * GBS + 1000 * GBS ))
# Override LR_WARMUP_SAMPLES so LR_DECAY_SAMPLES stays positive. With
# --no-load-rng / --no-load-optim we restart the scheduler at iter 0 anyway.
export LR_WARMUP_SAMPLES=0
export LR_WSD_DECAY_SAMPLES=1
export EXIT_MIN=240
export LOG_INTERVAL=1
export EVAL_INTERVAL=99999999999
export EVAL_ITERS=0
export SAVE_INTERVAL=99999999999
export USE_DYNAMIC_RES=0       # parity step 1: fixed-res images on both sides
export SEQUENCE_PARALLEL=1

# Ckpt paths (the pretrain script reads these and routes via --load / --save).
export LOAD_CHECKPOINT_DIR="${CKPT_RUN_ROOT}/checkpoints"
export SAVE_CHECKPOINT_DIR="${RUN_DIR}/save"
export OUTPUT="${RUN_DIR}"
export LOGS_DIR="${RUN_DIR}/logs"
export TENSORBOARD_DIR="${RUN_DIR}/tb"
export WANDB_DIR="${RUN_DIR}/wandb"
export RUN_NAME

# --ckpt-step pins the resume iter. --calculate-per-token-loss aligns gradient
# formula with hetero. --no-load-rng restarts the scheduler from seed=1234.
# Disable MTP and selective recompute for parity (hetero side runs without
# MTP, and --mtp-num-layers 0 also sidesteps an MTP-block autograd bug under
# this resume config).
export HYBRID_LAYER_PATTERN="MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME"
export DISABLE_RECOMPUTE=1
# --pixel-shuffle: gated by USE_DYNAMIC_RES=1 in the recipe but the ckpt was
# trained with pixel-shuffle on (vision_projection input dim 5120 = 4*1280),
# so re-add it explicitly when USE_DYNAMIC_RES=0.
export EXTRA_MEGATRON_ARGS="--ckpt-step ${CKPT_STEP} --calculate-per-token-loss --no-load-rng --no-load-optim --mtp-num-layers 0 --pixel-shuffle"

# Container + repo
export CONTAINER_IMAGE_OVERRIDE="${CONTAINER_IMAGE}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL}"
export MEGATRON_ROOT="${SANJEEV_REPO}"
export SBATCH_NODES=2

# The blend yaml's val: section eagerly post_initializes both splits even with
# --eval-iters 0. /home is NFS-mounted so reading from the cluster-shared home
# makes the val split resolvable cheaply; it's never actually iterated.
export MULTIMODAL_DATA_ROOT=/home/sasatheesh/data/multimodal_data

export DUMP_DATA_ONLY="${DUMP_DATA_ONLY:-0}"
export DUMP_N_STEPS="${DUMP_N_STEPS:-5}"
export DUMP_OUTPUT_DIR="${DUMP_OUTPUT_DIR:-${RUN_DIR}/dumps}"

cd "${SANJEEV_REPO}"
exec bash "${SANJEEV_REPO}/examples/multimodal/v3/pretrain_3b_nano_vlm_sota_90t_10v.sh"
