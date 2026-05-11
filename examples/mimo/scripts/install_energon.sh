#!/bin/bash
# Install the custom Megatron-Energon build used by the MIMO multimodal data path.

set -euo pipefail

DEFAULT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/public/Megatron-Energon-sasatheesh"
ENERGON_PATH="${1:-$DEFAULT_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Installing Megatron-Energon from: ${ENERGON_PATH}"
if [[ ! -d "${ENERGON_PATH}" ]]; then
  echo "ERROR: Directory not found: ${ENERGON_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -e "${ENERGON_PATH}[multimodal]"

"${PYTHON_BIN}" - <<'PY'
from megatron.energon.task_encoder.multimodal import MultiModalPackingEncoder
import megatron.energon

print(f"Megatron-Energon installed from: {megatron.energon.__file__}")
print(f"MultiModalPackingEncoder OK: {MultiModalPackingEncoder.__name__}")
PY
