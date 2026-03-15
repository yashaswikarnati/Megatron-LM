#!/bin/bash
# Build a new sqsh container with custom energon installed.
# Usage: bash scripts/build_energon_container.sh
set -euo pipefail

# --- Configuration ---
BASE_SQSH="/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_42663997.sqsh"
OUTPUT_SQSH="/lustre/fsw/portfolios/coreai/users/ykarnati/containers/mcore_ci_dev_42663997_energon_v2.sqsh"
ENERGON_REPO="git+ssh://git@gitlab-master.nvidia.com:12051/sasatheesh/Megatron-Energon.git@claude/fork-energon-ReMyw"
CONTAINER_NAME="mcore_energon_build"

# Use local SSD for enroot working data (fast + plenty of space)
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot_data_$$}"
mkdir -p "${ENROOT_DATA_PATH}"

echo "=== Building energon container ==="
echo "Base:   ${BASE_SQSH}"
echo "Output: ${OUTPUT_SQSH}"
echo "Energon: ${ENERGON_REPO}"
echo "Enroot data: ${ENROOT_DATA_PATH}"
echo ""

# --- Cleanup any previous build ---
if enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing previous build container..."
    enroot remove "${CONTAINER_NAME}"
fi

# --- Step 1: Create writable container from base sqsh ---
echo "=== Step 1: Creating writable container from base sqsh ==="
enroot create --name "${CONTAINER_NAME}" "${BASE_SQSH}"
echo "Done."

# --- Step 2: Install energon inside the container ---
echo "=== Step 2: Installing energon ==="
enroot start --rw "${CONTAINER_NAME}" \
    bash -c "
        GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -i /home/ykarnati/.ssh/id_ed25519_gitlab_master' \
            uv pip install --no-deps '${ENERGON_REPO}' && \
        uv pip install rapidyaml && \
        echo '--- Energon installed successfully ---' && \
        python -c 'from megatron.energon.task_encoder.multimodal import MultiModalPackingEncoder; print(\"MultiModalPackingEncoder import OK\")' && \
        pip show megatron-energon 2>/dev/null || echo 'Package name unknown, but install succeeded'
    "
echo "Done."

# --- Step 3: Export to new sqsh ---
echo "=== Step 3: Exporting to new sqsh ==="
if [ -f "${OUTPUT_SQSH}" ]; then
    echo "WARNING: ${OUTPUT_SQSH} already exists. Backing up..."
    mv "${OUTPUT_SQSH}" "${OUTPUT_SQSH}.bak.$(date +%Y%m%d%H%M%S)"
fi
enroot export --output "${OUTPUT_SQSH}" "${CONTAINER_NAME}"
echo "Done."

# --- Step 4: Cleanup ---
echo "=== Step 4: Cleaning up ==="
enroot remove "${CONTAINER_NAME}"
echo "Done."

echo ""
echo "=== Container built successfully ==="
echo "Output: ${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
