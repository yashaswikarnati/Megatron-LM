#!/bin/bash
# Run MIMO throughput benchmark on a single node.
# Usage:
#   ./run.sh --config benchmarks/mimo_throughput/configs/baseline_8gpu.yaml
#   ./run.sh --configs-dir benchmarks/mimo_throughput/configs/
set -euo pipefail

NPROC=${NPROC:-8}

uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    -m benchmarks.mimo_throughput.runner "$@"
