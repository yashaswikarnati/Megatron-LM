#!/bin/bash
# Non-intrusive wrapper around run_hetero_nemotron_54l_hel_train.sh that
# launches a py-spy sidecar before exec-ing the existing runner.
#
# Why a wrapper instead of editing the runner: keeps the production runner
# untouched. The sbatch script swaps this in via one-line change.
#
# Activation: set PYSPY_SIDECAR=1 in the env. With PYSPY_SIDECAR unset/0 this
# wrapper is a transparent passthrough.
#
# Sidecar is spawned on LOCAL_RANK=0 only (one per node). Output goes to
# PYSPY_OUT_DIR (default ${RUN_DIR}/pyspy when RUN_DIR is exported by the
# sbatch).
#
# The sidecar is detached via setsid + nohup so it survives the exec at the
# end of this script.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_hetero_nemotron_54l_hel_train.sh"

if [[ ! -x "$RUNNER" && ! -r "$RUNNER" ]]; then
  echo "[pyspy-wrap] FATAL: runner not found at $RUNNER" >&2
  exit 1
fi

# Local rank derivation mirrors the runner's own logic so we don't depend on
# its env-export ordering.
LOCAL_RANK_DERIVED="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"

if [[ "${PYSPY_SIDECAR:-0}" == "1" && "${LOCAL_RANK_DERIVED}" == "0" ]]; then
  PYSPY_OUT_DIR="${PYSPY_OUT_DIR:-${RUN_DIR:-/tmp}/pyspy}"
  mkdir -p "${PYSPY_OUT_DIR}"
  echo "[pyspy-wrap] spawning sidecar (node=${SLURMD_NODENAME:-$(hostname -s)}) out=${PYSPY_OUT_DIR}"
  # Detach so the sidecar survives the `exec` below. setsid + new session
  # prevents the runner's signal mask from killing it. stdin redirected from
  # /dev/null so the sidecar doesn't accidentally consume the runner's stdin.
  setsid nohup bash "${SCRIPT_DIR}/pyspy_sidecar.sh" "${PYSPY_OUT_DIR}" \
    > "${PYSPY_OUT_DIR}/sidecar.out" 2>&1 < /dev/null &
  SIDECAR_PID=$!
  disown "${SIDECAR_PID}" 2>/dev/null || true
  echo "[pyspy-wrap] sidecar pid=${SIDECAR_PID}"
fi

# Hand off to the original runner unmodified. All script args + env are
# preserved.
exec bash "${RUNNER}" "$@"
