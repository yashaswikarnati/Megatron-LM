#!/bin/bash

# Run from the repository root:
#   ./examples/mimo/scripts/run_hetero_mock_train.sh

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TRAIN_ITERS=${TRAIN_ITERS:-2}
PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${GPUS_PER_NODE}" \
  examples/mimo/train_hetero.py \
  --train-iters "${TRAIN_ITERS}" \
  "$@"
