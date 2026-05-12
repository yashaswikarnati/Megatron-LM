#!/bin/bash
# Run controlled homogeneous-vs-heterogeneous MIMO training parity checks.

set -euo pipefail

PARITY_CASE="${PARITY_CASE:-${1:-short}}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-python}"
DEFAULT_OUT_BASE="${RUN_DIR:-/tmp}"
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_BASE}/mimo_training_parity_${PARITY_CASE}_${SLURM_JOB_ID:-local}}"
STATE_ATOL="${STATE_ATOL:-2.0e-4}"
STATE_RTOL="${STATE_RTOL:-2.0e-4}"
LOSS_ATOL="${LOSS_ATOL:-1.0e-5}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

COMMON_MODEL_ARGS=(
  --model-provider mock
  --fp32
  --hidden-size 64
  --num-layers 2
  --num-attention-heads 4
  --vocab-size 512
  --seq-length 32
  --image-seq-length 8
  --image-token-id 511
  --pad-token-id 0
  --num-moe-experts 0
)

COMMON_TRAIN_ARGS=(
  --micro-batch-size 1
  --num-microbatches 2
  --lr 2.0e-4
  --min-lr 0.0
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.999
  --clip-grad 1.0
  --no-overlap-grad-reduce
  --ddp-bucket-size 0
  --seed 12345
)

run_distributed() {
  local nproc="$1"
  shift
  "${PYTHON_CMD[@]}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node "${nproc}" \
    examples/mimo/validation/training_parity.py \
    "$@"
}

run_case() {
  local case_name="$1"
  local train_iters="$2"
  local snapshot_interval="$3"
  local llm_dp="$4"
  local encoder_dp="$5"
  local require_state="$6"
  local loss_tolerance="${7:-${LOSS_ATOL}}"
  local state_atol="${8:-${STATE_ATOL}}"
  local state_rtol="${9:-${STATE_RTOL}}"

  local out_dir="${OUT_ROOT}/${case_name}"
  local init_state="${out_dir}/initial_state.pt"
  local hetero_world=$((encoder_dp + llm_dp))

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  run_distributed 1 \
    --mode init \
    --output-dir "${out_dir}" \
    --initial-state-path "${init_state}" \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --train-iters 1

  run_distributed "${llm_dp}" \
    --mode homo \
    --output-dir "${out_dir}" \
    --initial-state-path "${init_state}" \
    --llm-dp "${llm_dp}" \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --global-batch-size $((llm_dp * 2)) \
    --train-iters "${train_iters}" \
    --snapshot-interval "${snapshot_interval}"

  run_distributed "${hetero_world}" \
    --mode hetero \
    --output-dir "${out_dir}" \
    --initial-state-path "${init_state}" \
    --encoder-offset 0 \
    --encoder-tp 1 \
    --encoder-pp 1 \
    --encoder-dp "${encoder_dp}" \
    --llm-offset "${encoder_dp}" \
    --llm-tp 1 \
    --llm-pp 1 \
    --llm-dp "${llm_dp}" \
    --llm-ep 1 \
    --llm-expt-tp 1 \
    --llm-expt-dp "${llm_dp}" \
    "${COMMON_MODEL_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --global-batch-size $((llm_dp * 2)) \
    --train-iters "${train_iters}" \
    --snapshot-interval "${snapshot_interval}"

  COMPARE_ARGS=(
    --homo-dir "${out_dir}/homo"
    --hetero-dir "${out_dir}/hetero"
    --output "${out_dir}/parity_summary.json"
    --atol "${state_atol}"
    --rtol "${state_rtol}"
    --max-loss-diff "${loss_tolerance}"
  )
  if [[ "${require_state}" == "yes" ]]; then
    COMPARE_ARGS+=(--require-state)
  fi
  "${PYTHON_CMD[@]}" examples/mimo/validation/compare_training_parity.py "${COMPARE_ARGS[@]}"
}

case "${PARITY_CASE}" in
  short)
    run_case short 4 1 1 1 yes 1.0e-5
    ;;
  curve)
    run_case curve 250 0 1 1 no 2.0e-4
    ;;
  fanout)
    run_case fanout 4 1 2 1 yes 1.0e-4 1.0e-3 1.0e-3
    ;;
  all)
    run_case short 4 1 1 1 yes 1.0e-5
    run_case curve 250 0 1 1 no 2.0e-4
    run_case fanout 4 1 2 1 yes 1.0e-4 1.0e-3 1.0e-3
    ;;
  *)
    echo "Unknown PARITY_CASE=${PARITY_CASE}; use short, curve, fanout, or all." >&2
    exit 1
    ;;
esac

echo "Parity artifacts: ${OUT_ROOT}"
