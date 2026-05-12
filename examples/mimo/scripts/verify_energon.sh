#!/bin/bash
# Verify the custom Megatron-Energon build used by the MIMO multimodal data path.

set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import json
from importlib import metadata

try:
    import megatron.energon
    import torchvision
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: missing Energon multimodal runtime dependency. "
        "Run through a PyTorch base image/Cog synced venv that already provides torch and "
        "torchvision, then install repo deps with `uv sync --locked --extra dev --extra mlm`. "
        "For a non-container local env, install torch/torchvision separately with versions that "
        "match your CUDA stack before syncing this project. "
        f"Original error: {exc}"
    ) from exc

from megatron.energon.task_encoder.multimodal import MultiModalPackingEncoder, PackingConfig, VisionConfig
from megatron.energon.task_encoder.multimodal.sample_types import PackedSample
from megatron.energon.task_encoder.multimodal.vision_tokens import get_num_image_embeddings
from packaging.version import InvalidVersion, Version

EXPECTED_COMMIT = "d456cbd4a9a8a760b20be51194a0209c9a945b0a"
EXPECTED_LOCAL = f"g{EXPECTED_COMMIT[:9]}"

dist = metadata.distribution("megatron-energon")
version = dist.version
direct_url = dist.read_text("direct_url.json")
commit = None
if direct_url:
    commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")

try:
    local_version = Version(version).local
except InvalidVersion:
    local_version = None

if commit != EXPECTED_COMMIT and local_version != EXPECTED_LOCAL:
    raise SystemExit(
        "ERROR: megatron-energon is not the pinned MIMO fork "
        f"({EXPECTED_COMMIT}); found version={version!r}, commit={commit!r}"
    )

print(f"Megatron-Energon path: {megatron.energon.__file__}")
print(f"Megatron-Energon version: {version}")
print(f"Megatron-Energon commit: {commit or 'version-local-tag'}")
print(f"torchvision OK: {torchvision.__version__}")
print(f"MultiModalPackingEncoder OK: {MultiModalPackingEncoder.__name__}")
print(f"PackingConfig OK: {PackingConfig.__name__}")
print(f"VisionConfig OK: {VisionConfig.__name__}")
print(f"PackedSample OK: {PackedSample.__name__}")
print(f"get_num_image_embeddings OK: {get_num_image_embeddings.__name__}")
PY
