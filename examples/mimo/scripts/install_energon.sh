#!/bin/bash
# Install custom Megatron-Energon with multimodal support for MIMO.
#
# Usage:
#   ./examples/mimo/scripts/install_energon.sh              # use default path
#   ./examples/mimo/scripts/install_energon.sh /path/to/energon  # custom path
#
# Default: ykarnati's public Megatron-Energon (sasatheesh fork with multimodal)

set -euo pipefail

DEFAULT_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ykarnati/public/Megatron-Energon-sasatheesh"
ENERGON_PATH="${1:-$DEFAULT_PATH}"

echo "Installing Megatron-Energon from: ${ENERGON_PATH}"

if [[ ! -d "$ENERGON_PATH" ]]; then
    echo "ERROR: Directory not found: ${ENERGON_PATH}"
    exit 1
fi

uv pip install -e "${ENERGON_PATH}[multimodal]"

echo ""
echo "Verifying installation..."
uv run python -c "
from megatron.energon.task_encoder.multimodal import (
    MultiModalPackingEncoder,
    VisionConfig,
    PackingConfig,
)
import megatron.energon
print(f'Megatron-Energon installed from: {megatron.energon.__file__}')
print('MultiModalPackingEncoder imported OK')
"
