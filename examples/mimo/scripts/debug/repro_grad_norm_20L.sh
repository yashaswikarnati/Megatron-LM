#!/bin/bash
# ============================================================================
# NMFW-27 Debug Script: Reproduce large grad norms with Adapter + LLM training
# ============================================================================
#
# PURPOSE:
#   Reproduce the flat lm_loss + huge grad norm behavior seen in production
#   stage2 (64-node) training on a single interactive node. This helps debug
#   whether the issue is intrinsic to the training setup or scale-dependent.
#
# APPROACH:
#   Two modes controlled by MODEL_SIZE env var:
#
#   MODEL_SIZE=20L (default):
#     - 20-layer model, loads first 20 layers from 52L MIMO stage1 checkpoint
#     - Uses --finetune + --dist-ckpt-strictness ignore_all to handle 52->20
#       layer mismatch (checkpoint layers 20-51 silently ignored)
#     - Layer pattern MEMEM*EMEMEM*EMEMEM* matches first 20 chars of 52L pattern
#     - Pro: fast iteration, lower memory
#     - Con: only 20/52 LLM layers loaded from checkpoint
#
#   MODEL_SIZE=52L:
#     - Full 52-layer model, exact architecture match with production
#     - Loads full checkpoint with no layer mismatch
#     - Uses aggressive recompute (full granularity) to fit on 8x H100 80GB
#     - Pro: exact reproduction of production model
#     - Con: slower, tighter memory
#
# CHECKPOINT OPTIONS (set via CKPT_SOURCE env var):
#   mimo  (default) - MIMO stage1 adapter-only checkpoint (--load path)
#   sasatheesh      - Non-MIMO checkpoint from sasatheesh (--nemotron-checkpoint)
#   none            - Train from scratch (no checkpoint loading)
#
# TRAINING SETTINGS (match stage2 64-node script):
#   - LR: 5e-5 with WSD schedule (warmup=500, decay=500)
#   - ViT frozen, projection + LLM trainable
#   - Loss scaling enabled
#   - tiktoken-pattern v2
#   - Sequence packing with 8k seq length
#
# USAGE (from repo root, on interactive node with 8x H100):
#   # Default: 20L model with MIMO checkpoint
#   bash examples/mimo/scripts/debug/repro_grad_norm_20L.sh
#
#   # Full 52L model with MIMO checkpoint
#   MODEL_SIZE=52L bash examples/mimo/scripts/debug/repro_grad_norm_20L.sh
#
#   # 20L model with non-MIMO (sasatheesh) checkpoint
#   CKPT_SOURCE=sasatheesh bash examples/mimo/scripts/debug/repro_grad_norm_20L.sh
#
#   # Try higher LR (5e-4) to see if grad norms are worse
#   LR_OVERRIDE=5e-4 bash examples/mimo/scripts/debug/repro_grad_norm_20L.sh
#
#   # Train from scratch (baseline comparison)
#   CKPT_SOURCE=none bash examples/mimo/scripts/debug/repro_grad_norm_20L.sh
#
# ============================================================================

set -euo pipefail

# ── Configuration knobs ─────────────────────────────────────────────────────
MODEL_SIZE="${MODEL_SIZE:-20L}"              # 20L or 52L
CKPT_SOURCE="${CKPT_SOURCE:-mimo}"           # mimo, sasatheesh, or none
CKPT_ITER="${CKPT_ITER:-3000}"               # which iteration to load
LR_OVERRIDE="${LR_OVERRIDE:-5e-5}"           # default matches stage2 prod
TRAIN_ITERS="${TRAIN_ITERS:-500}"            # short run to see grad norms
WANDB_LOGGING="${WANDB_LOGGING:-false}"      # set to "true" to enable wandb

# ── Environment ─────────────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Ensure Megatron and MIMO example modules are on PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${MEGATRON_ROOT}:${MEGATRON_ROOT}/examples/mimo:${PYTHONPATH:-}"

GPUS_PER_NODE=8
NUM_NODES=1

LOG_DIR=./logs/repro_grad_norm_${MODEL_SIZE}_${CKPT_SOURCE}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# ── Checkpoint paths ────────────────────────────────────────────────────────
MIMO_STAGE1_CKPT="/workspace/Megatron-LM/logs/mimo_vlm_adapter_only_stage1/checkpoints"
SASATHEESH_CKPT="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/users/sasatheesh/data/models/v0/3B_moe_vlm_omnicorpus_adapter_only/checkpoints"

# ── Data path (same as 20L script / production) ────────────────────────────
DATA_PATH="${DATA_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/kshih/workspace/blends/eagle_recipe_online_packing/final_recipe/pretrain_base_non_sft_cw_dfw.yaml}"

# ── Tokenizer ───────────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/checkpoints/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-multimodal-pretraining/snapshots/7344a79074e20d9ab548e14c25b0492345394f67}"

# ── Distributed launch (torchrun for interactive node) ──────────────────────
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --tee 3
    --log-dir "$LOG_DIR"
)

# ── Model parallelism (TP=2, EP=4, PP=1 — same as stage1/stage2) ──────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)

# ── Model architecture (20L or 52L) ────────────────────────────────────────
case "$MODEL_SIZE" in
    20L)
        NUM_LAYERS=20
        HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*"
        # recompute-activations is sufficient for 20L
        RECOMPUTE_ARGS=(--recompute-activations)
        ;;
    52L)
        NUM_LAYERS=52
        HYBRID_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        # Full recompute needed to fit 52L on 8x H100 80GB
        RECOMPUTE_ARGS=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
        ;;
    *)
        echo "ERROR: MODEL_SIZE must be 20L or 52L, got '$MODEL_SIZE'"
        exit 1
        ;;
esac

GPT_MODEL_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --num-layers "$NUM_LAYERS"
    --hidden-size 2688
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --ffn-hidden-size 1856
    --max-position-embeddings 8192
    --encoder-seq-length 8192
    --position-embedding-type none
    --normalization RMSNorm
    --disable-bias-linear
    --squared-relu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.0173
    --no-create-attention-mask-in-dataloader
    --tiktoken-pattern v2
)

# ── MoE / Mamba hybrid ─────────────────────────────────────────────────────
HYBRID_ARGS=(
    --num-experts 128
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --moe-aux-loss-coeff 0.0001
    --moe-router-topk-scaling-factor 2.5
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32
    --moe-router-load-balancing-type seq_aux_loss
    --moe-shared-expert-intermediate-size 3712
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-overlap
    --moe-permute-fusion
    --use-fused-weighted-squared-relu
    --enable-experimental
    --is-hybrid-model
    --hybrid-override-pattern "$HYBRID_PATTERN"
    --mamba-num-heads 64
    --mamba-head-dim 64
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
    --attention-backend flash
)

# ── Vision (RADIO) — stage2 freeze: ViT frozen, projection+LLM trainable ──
VISION_ARGS=(
    --img-h 512
    --img-w 512
    --patch-dim 16
    --pixel-shuffle
    --disable-vision-class-token
    --vision-model-type radio
    --class-token-len 8
    --max-num-tiles 12
    --use-tiling
    --use-thumbnail
    --freeze-vit
)

# ── Training (match stage2 64-node settings) ───────────────────────────────
#
# Key settings from production stage2:
#   - WSD LR schedule with minus_sqrt decay
#   - warmup=500, decay=500
#   - lr=5e-5 (can override with LR_OVERRIDE=5e-4 to test)
#   - loss scaling enabled
#
# For short test runs (TRAIN_ITERS < 1000), cap warmup/decay to fit.
LR_WARMUP_ITERS=500
LR_WSD_DECAY_ITERS=500
if [ "$TRAIN_ITERS" -lt 1000 ]; then
    # Ensure warmup + decay fit within total iters, and warmup < decay_iters
    LR_WARMUP_ITERS=$(( TRAIN_ITERS / 5 ))
    LR_WSD_DECAY_ITERS=$(( TRAIN_ITERS / 5 ))
    if [ "$LR_WARMUP_ITERS" -lt 1 ]; then
        LR_WARMUP_ITERS=1
    fi
fi

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters "$TRAIN_ITERS"
    --lr "$LR_OVERRIDE"
    --min-lr 0
    --weight-decay 0.05
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --model-provider nemotron_moe_vlm
    --dataset-provider energon_multimodal
    --bf16
)

# ── Data / Tokenizer ───────────────────────────────────────────────────────
DATA_ARGS=(
    --dataloader-type external
    --data-path "$DATA_PATH"
    --total-seq-length 8192
    --num-workers 2
    --packing-buffer-size 128
    --use-loss-scaling
)

TOKENIZER_ARGS=(
    --tokenizer-type MultimodalTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --tokenizer-prompt-format nemotron6-moe
    --special-tokens "<image>"
)

# ── Eval / logging (log every step for grad norm analysis) ──────────────────
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 99999
    --eval-interval 99999
    --eval-iters 0
    --tensorboard-dir "$LOG_DIR/tensorboard"
    --log-throughput
    --distributed-timeout-minutes 30
)

# Optional wandb logging
if [ "$WANDB_LOGGING" = "true" ]; then
    EVAL_AND_LOGGING_ARGS+=(
        --wandb-project "vlm-grad-norm-debug"
        --wandb-exp-name "repro-${MODEL_SIZE}-${CKPT_SOURCE}-lr${LR_OVERRIDE}"
    )
fi

# ── Performance ─────────────────────────────────────────────────────────────
PERF_ARGS=(
    --no-check-for-nan-in-loss-and-grad
)

# ── Checkpoint loading (the key part) ───────────────────────────────────────
#
# Three modes:
#
# 1. "mimo" — Load from MIMO stage1 checkpoint via --load
#    For 20L: uses --finetune --dist-ckpt-strictness ignore_all to handle
#    the 52->20 layer mismatch. The model's sharded_state_dict only requests
#    layers 0-19, and the checkpoint's layers 20-51 are silently ignored.
#    For 52L: exact match, straightforward finetune load.
#
# 2. "sasatheesh" — Load from non-MIMO checkpoint via --nemotron-checkpoint
#    This uses the submodule-by-submodule loader in model_helpers.py.
#    For 20L: the language_model loader will hit shape/key mismatches for
#    layers 20-51 that exist in checkpoint but not in model. This will FAIL
#    with the current code because _load_submodule_from_ckpt raises on
#    missing/unexpected keys. Only use with MODEL_SIZE=52L.
#    For 52L: works correctly (same as adapter-only script).
#
# 3. "none" — Train from scratch (no checkpoint), useful as baseline.
#
CKPT_ARGS=()
case "$CKPT_SOURCE" in
    mimo)
        CKPT_ARGS=(
            --load "$MIMO_STAGE1_CKPT"
            --ckpt-step "$CKPT_ITER"
            --finetune
            --no-load-optim
            --no-load-rng
        )
        if [ "$MODEL_SIZE" = "20L" ]; then
            # 52L checkpoint -> 20L model: tell dist_checkpointing to ignore
            # the extra layers in the checkpoint that don't map to our model.
            CKPT_ARGS+=(--dist-ckpt-strictness ignore_all)
        fi
        ;;
    sasatheesh)
        if [ "$MODEL_SIZE" = "20L" ]; then
            echo "WARNING: --nemotron-checkpoint with 20L model will likely fail"
            echo "  because the language model submodule loader does strict key checking."
            echo "  Use MODEL_SIZE=52L with CKPT_SOURCE=sasatheesh, or use CKPT_SOURCE=mimo."
            echo "  Proceeding anyway (will fail at checkpoint load time)..."
        fi
        CKPT_ARGS=(
            --nemotron-checkpoint "$SASATHEESH_CKPT"
            --skip-projection-checkpoint
        )
        ;;
    none)
        echo ">>> No checkpoint loading — training from scratch"
        ;;
    *)
        echo "ERROR: CKPT_SOURCE must be mimo, sasatheesh, or none. Got '$CKPT_SOURCE'"
        exit 1
        ;;
esac

# ── Summary ─────────────────────────────────────────────────────────────────
TP=2; EP=4; DP=$((GPUS_PER_NODE / TP))
echo "============================================================"
echo "  NMFW-27 Grad Norm Repro — ${MODEL_SIZE} model"
echo "============================================================"
echo "Model:     ${NUM_LAYERS} layers | Pattern: ${HYBRID_PATTERN}"
echo "GPUs:      ${GPUS_PER_NODE} | TP=${TP} EP=${EP} DP=${DP} (DP_eff=1)"
echo "Checkpoint: ${CKPT_SOURCE} (iter ${CKPT_ITER})"
echo "Stage:     stage2 — ViT frozen, projection + LLM trainable"
echo "LR:        ${LR_OVERRIDE} -> 0 (WSD minus_sqrt, warmup=500, decay=500)"
echo "GBS:       8 | MBS: 1 | Seq: 8192 | Iters: ${TRAIN_ITERS}"
echo "Loss scaling: enabled"
echo "Recompute: ${RECOMPUTE_ARGS[*]}"
if [ ${#CKPT_ARGS[@]} -gt 0 ]; then
echo "Ckpt args: ${CKPT_ARGS[*]}"
fi
echo "Data:      ${DATA_PATH}"
echo "Logs:      ${LOG_DIR}"
echo "Wandb:     ${WANDB_LOGGING}"
echo ""
echo "WHAT TO LOOK FOR:"
echo "  - grad_norm values in log output (should be column in training log)"
echo "  - lm_loss trend (should decrease; flat = problem)"
echo "  - Compare grad_norm magnitude vs. baseline (CKPT_SOURCE=none)"
echo "============================================================"

uv run python -m torch.distributed.run \
    "${DISTRIBUTED_ARGS[@]}" \
    examples/mimo/train.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${HYBRID_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${TOKENIZER_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${PERF_ARGS[@]}"
