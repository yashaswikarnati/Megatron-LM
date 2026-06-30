# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the logical data-parallel-size override in argument validation."""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args


def _parse_stock_args(monkeypatch, *, world_size: int, tensor_parallel_size: int):
    argv = [
        "pytest",
        "--num-layers",
        "2",
        "--hidden-size",
        "24",
        "--num-attention-heads",
        "6",
        "--max-position-embeddings",
        "8",
        "--seq-length",
        "8",
        "--micro-batch-size",
        "2",
        "--tensor-model-parallel-size",
        str(tensor_parallel_size),
        "--vocab-size",
        "128",
        "--tokenizer-type",
        "NullTokenizer",
    ]
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(sys, "argv", argv)
    return parse_args()


def test_validate_args_uses_logical_data_parallel_size_for_batch_derivations(monkeypatch):
    """The override bypasses only homogeneous factorization and drives batch defaults."""
    args = _parse_stock_args(monkeypatch, world_size=8, tensor_parallel_size=3)

    validated = validate_args(args, data_parallel_size_override=2)

    assert validated is args
    assert args.data_parallel_size == 2
    assert args.global_batch_size == 4
    assert args.eval_global_batch_size == 4


@pytest.mark.parametrize("override", [0, -1, 1.5, True])
def test_validate_args_rejects_invalid_data_parallel_size_override(monkeypatch, override):
    args = _parse_stock_args(monkeypatch, world_size=8, tensor_parallel_size=2)

    with pytest.raises(ValueError, match="data_parallel_size_override.*positive integer"):
        validate_args(args, data_parallel_size_override=override)


def test_validate_args_default_data_parallel_derivation_is_unchanged(monkeypatch):
    args = _parse_stock_args(monkeypatch, world_size=8, tensor_parallel_size=2)

    validate_args(args)

    assert args.data_parallel_size == 4
    assert args.global_batch_size == 8
    assert args.eval_global_batch_size == 8
