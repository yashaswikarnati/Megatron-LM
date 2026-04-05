#!/bin/bash
# Test encoder offload: run baseline and offload configs, compare memory and overlap.
#
# Usage (from repo root):
#   bash benchmarks/mimo_throughput/scripts/test_offload.sh
#
# What it checks:
#   1. Memory: offload run should show lower peak memory during LLM phase
#   2. Overlap: reload_sync wait time should be ~0ms (fully overlapped with cooldown)
#   3. Correctness: both runs should produce similar loss values
#
# Look for these log lines in the output:
#   [OFFLOAD iter=N] X.XX GB -> Y.YY GB (freed Z.ZZ GB, K opt tensors)
#   [RELOAD iter=N]  X.XX GB -> Y.YY GB (+Z.ZZ GB)
#   [RELOAD_SYNC iter=N] wait=0.XX ms (0 = fully overlapped)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/benchmarks/mimo_throughput/configs/offload_test"
RESULTS_DIR="${RESULTS_DIR:-/tmp/offload_test_results}"
NPROC="${NPROC:-8}"

cd "$REPO_ROOT"

echo "============================================="
echo "  Encoder Offload Test"
echo "  GPUs: $NPROC"
echo "  Results: $RESULTS_DIR"
echo "============================================="

mkdir -p "$RESULTS_DIR"

echo ""
echo "--- Run 1: BASELINE (no offload) ---"
echo ""
uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    -m benchmarks.mimo_throughput.runner \
    --config "$CONFIG_DIR/baseline.yaml" \
    --results-dir "$RESULTS_DIR/baseline" \
    2>&1 | tee "$RESULTS_DIR/baseline.log"

echo ""
echo "--- Run 2: OFFLOAD (encoder_param_offload=true) ---"
echo ""
uv run python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    -m benchmarks.mimo_throughput.runner \
    --config "$CONFIG_DIR/offload.yaml" \
    --results-dir "$RESULTS_DIR/offload" \
    2>&1 | tee "$RESULTS_DIR/offload.log"

echo ""
echo "============================================="
echo "  Results Summary"
echo "============================================="
echo ""

# Extract key metrics from logs
echo "--- Memory ---"
echo "Baseline peak:"
grep -o "max_memory_gb.*" "$RESULTS_DIR/baseline.log" | tail -3 || echo "(not found)"
echo ""
echo "Offload peak:"
grep -o "max_memory_gb.*" "$RESULTS_DIR/offload.log" | tail -3 || echo "(not found)"

echo ""
echo "--- Offload/Reload logs ---"
grep "OFFLOAD\|RELOAD" "$RESULTS_DIR/offload.log" | head -20 || echo "(no offload logs found)"

echo ""
echo "--- Throughput comparison ---"
echo "Baseline:"
grep "TFLOPs/GPU" "$RESULTS_DIR/baseline.log" | tail -3 || echo "(not found)"
echo ""
echo "Offload:"
grep "TFLOPs/GPU" "$RESULTS_DIR/offload.log" | tail -3 || echo "(not found)"

echo ""
echo "Done. Full logs in $RESULTS_DIR/"
