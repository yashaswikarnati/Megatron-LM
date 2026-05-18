#!/bin/bash
# 5k-step train-loss parity run -- Sanjeev side, GBS=192.
# 8 nodes (TP=2 EP=16, LLM DP=32). GBS=192 via grad accumulation:
# MBS=1, NUM_MICROBATCHES=6.
# Paired with sbatch_hetero_parity_5k.sh.

#SBATCH -A nemotron_n4_pre
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH -J sanj-parity-5k
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/%x-%j.err

set -euo pipefail

SCRATCH_ROOT=/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch
SANJEEV_REPO="${SANJEEV_REPO:-${SCRATCH_ROOT}/sanjeev-repos/megatron-lm-clean}"
CONTAINER_IMAGE="${SANJEEV_CONTAINER_IMAGE:-${SCRATCH_ROOT}/images/m_lm_energon_0506.sqsh}"
TOKENIZER_MODEL="${SCRATCH_ROOT}/tokenizers/sanjeevnv-multimodal-pretraining-26f81d5db838eb6dee2ff8692db83a2fbc76f3ff"

CKPT_RUN_ROOT="/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/workspace/output/3b_nano_vlm_sota_mtp2_90t10v_post_c_radio_omni_96n_tp2_ep16_selective_300b_20260511"
CKPT_STEP="${CKPT_STEP:-1000}"

RUN_NAME="sanj-parity-5k"
RUN_DIR="${SCRATCH_ROOT}/runs/${RUN_NAME}/${SLURM_JOB_ID:-local}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/save" "${RUN_DIR}/tb"

# ---- Overrides passed to the pretrain script --------------------------------
# Topology: TP=2 EP=16. With 8 nodes (64 GPUs) and TP=2 we get LLM_DP=32.
export VISION_MODEL_TYPE=radio
export RADIO_ENCODER_DIR=post-c-radio-omni
export TP=2
export EP=16
export NUM_EXPERTS=128
export MOE_ROUTER_TOPK=6
export MBS=1
export GBS=192
export NUM_WORKERS=2
export PACKING_BUFFER_SIZE=4
export SEQ_LEN=8192
export DECODER_SEQ_LEN=8192
# 5000 optimizer steps after the resume + the ckpt's iter_1000 offset.
export TRAIN_SAMPLES=$(( 5000 * GBS + 1000 * GBS ))
# Flat LR: warmup=0 and the recipe's huge decay window keep LR constant.
export LR_WARMUP_SAMPLES=0
export LR_WSD_DECAY_SAMPLES=1
export EXIT_MIN=235        # ~5 min drain before the 4 h wall
export LOG_INTERVAL=1
export EVAL_INTERVAL=99999999999
export EVAL_ITERS=0
export SAVE_INTERVAL=99999999999
export USE_DYNAMIC_RES=0
export SEQUENCE_PARALLEL=1

# Ckpt paths
export LOAD_CHECKPOINT_DIR="${CKPT_RUN_ROOT}/checkpoints"
export SAVE_CHECKPOINT_DIR="${RUN_DIR}/save"
export OUTPUT="${RUN_DIR}"
export LOGS_DIR="${RUN_DIR}/logs"
export TENSORBOARD_DIR="${RUN_DIR}/tb"
export WANDB_DIR="${RUN_DIR}/wandb"
export RUN_NAME

export HYBRID_LAYER_PATTERN="MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME"
export DISABLE_RECOMPUTE=1
# --ckpt-step pins the resume iter. --calculate-per-token-loss aligns gradient
# formula with hetero. --no-load-rng restarts the scheduler from seed=1234.
# --mtp-num-layers 0 sidesteps the MTP autograd issue under this resume config.
export EXTRA_MEGATRON_ARGS="--ckpt-step ${CKPT_STEP} --calculate-per-token-loss --no-load-rng --no-load-optim --mtp-num-layers 0 --pixel-shuffle"

export CONTAINER_IMAGE_OVERRIDE="${CONTAINER_IMAGE}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL}"
export MEGATRON_ROOT="${SANJEEV_REPO}"
export SBATCH_NODES=8

export MULTIMODAL_DATA_ROOT=/home/sasatheesh/data/multimodal_data

export DUMP_DATA_ONLY="${DUMP_DATA_ONLY:-0}"
export DUMP_N_STEPS="${DUMP_N_STEPS:-5}"
export DUMP_OUTPUT_DIR="${DUMP_OUTPUT_DIR:-${RUN_DIR}/dumps}"

cd "${SANJEEV_REPO}"
exec bash "${SANJEEV_REPO}/examples/multimodal/v3/pretrain_3b_nano_vlm_sota_90t_10v.sh"
