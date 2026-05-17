# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the hetero MIMO `--load-vision-from` loader.

These tests target the deterministic helpers in
``examples.mimo.training.hetero.checkpointing``:

* ``_tp_slice``                — pure-tensor TP slicing logic.
* DCP prefix filter + read     — write a tiny Megatron-Bridge-shaped DCP with
  ``model.vision_model.*`` keys and verify that the filter selects the right
  subset and that ``dcp.load`` rehydrates the tensors.

The full ``load_vision_from_checkpoint`` end-to-end path requires a real
``torch.distributed`` world plus a built ``MimoModel`` (see
``tests/unit_tests/models/test_mimo_checkpoint.py`` for that level of test).
We deliberately keep these tests single-process so they can run inside the
unit-test buckets without GPUs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Module under test — import here so a collection-time ImportError surfaces as
# a clear test failure rather than a cryptic skip.
from examples.mimo.training.hetero.checkpointing import (
    _VISION_DCP_PREFIX,
    _resolve_vision_dcp_dir,
    _tp_slice,
)

# ---------------------------------------------------------------------------
# _tp_slice — column- and row-parallel sharding behavior.
# ---------------------------------------------------------------------------


def test_tp_slice_tp1_is_passthrough():
    """TP=1 must return the input tensor unchanged."""
    t = torch.arange(8).view(4, 2)
    out = _tp_slice(t, t.shape, tp_rank=0, tp_size=1)
    assert out is t


def test_tp_slice_matching_shape_is_passthrough():
    """Tensor already at the per-rank shape must be returned as-is."""
    t = torch.arange(4).view(2, 2)
    out = _tp_slice(t, (2, 2), tp_rank=1, tp_size=2)
    assert out is t


def test_tp_slice_column_parallel_splits_first_dim():
    """Column-parallel weight: full[out_size, in] split along dim 0."""
    full = torch.arange(16, dtype=torch.float32).view(8, 2)
    param_shape = (4, 2)
    shard0 = _tp_slice(full, param_shape, tp_rank=0, tp_size=2)
    shard1 = _tp_slice(full, param_shape, tp_rank=1, tp_size=2)
    assert tuple(shard0.shape) == param_shape
    assert tuple(shard1.shape) == param_shape
    torch.testing.assert_close(shard0, full[:4])
    torch.testing.assert_close(shard1, full[4:])


def test_tp_slice_row_parallel_splits_second_dim():
    """Row-parallel weight: full[out, in_size] split along dim 1."""
    full = torch.arange(16, dtype=torch.float32).view(2, 8)
    param_shape = (2, 4)
    shard0 = _tp_slice(full, param_shape, tp_rank=0, tp_size=2)
    shard1 = _tp_slice(full, param_shape, tp_rank=1, tp_size=2)
    assert tuple(shard0.shape) == param_shape
    assert tuple(shard1.shape) == param_shape
    torch.testing.assert_close(shard0, full[:, :4])
    torch.testing.assert_close(shard1, full[:, 4:])


# ---------------------------------------------------------------------------
# _resolve_vision_dcp_dir — tracker vs flat layout.
# ---------------------------------------------------------------------------


def test_resolve_vision_dcp_dir_flat(tmp_path: Path):
    """Without a tracker file, the loader treats the path as a flat DCP."""
    assert _resolve_vision_dcp_dir(str(tmp_path)) == str(tmp_path)


def test_resolve_vision_dcp_dir_with_tracker(tmp_path: Path):
    """A tracker file makes the loader descend into iter_NNNNNNN/."""
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("42\n")
    expected = os.path.join(str(tmp_path), "iter_0000042")
    assert _resolve_vision_dcp_dir(str(tmp_path)) == expected


# ---------------------------------------------------------------------------
# DCP prefix filter — write a tiny Bridge-shaped DCP and round-trip it.
# ---------------------------------------------------------------------------


def _write_mock_vision_dcp(dcp_dir: str) -> dict[str, torch.Tensor]:
    """Write a tiny DCP with two `model.vision_model.*` keys and one decoy key.

    Returns the saved state-dict so callers can assert exact tensor equality.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter

    sd = {
        "model.vision_model.embedder.weight": torch.arange(12, dtype=torch.float32).view(3, 4),
        "model.vision_model.embedder.bias": torch.arange(3, dtype=torch.float32),
        # Decoy: not under the vision prefix; loader must skip it.
        "model.language_model.embed_tokens.weight": torch.zeros(2, 4, dtype=torch.float32),
    }
    writer = FileSystemWriter(dcp_dir, single_file_per_rank=True)
    dcp.save(sd, storage_writer=writer, no_dist=True)
    return sd


def test_dcp_prefix_filter_and_read(tmp_path: Path):
    """The loader's prefix filter selects only `model.vision_model.*` keys
    and `dcp.load` rehydrates them with the saved values.

    This test is process-local (no distributed init) — it exercises the same
    metadata-read + filter + load path the real loader uses, isolated from
    the MimoModel build.
    """
    pytest.importorskip("torch.distributed.checkpoint")
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    dcp_dir = tmp_path / "post-c-radio-omni"
    dcp_dir.mkdir()
    saved = _write_mock_vision_dcp(str(dcp_dir))

    reader = FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata().state_dict_metadata

    # 1. Prefix filter: same one-liner the loader uses.
    vision_keys = {
        k
        for k, meta in metadata.items()
        if k.startswith(_VISION_DCP_PREFIX) and isinstance(meta, TensorStorageMetadata)
    }
    assert vision_keys == {
        "model.vision_model.embedder.weight",
        "model.vision_model.embedder.bias",
    }, "decoy `model.language_model.*` key leaked through the prefix filter"

    # 2. Round-trip: build the empty-tensor request dict and dcp.load it.
    load_sd = {
        k: torch.empty(metadata[k].size, dtype=metadata[k].properties.dtype) for k in vision_keys
    }
    dcp.load(load_sd, storage_reader=reader)

    for k in vision_keys:
        torch.testing.assert_close(load_sd[k], saved[k])
