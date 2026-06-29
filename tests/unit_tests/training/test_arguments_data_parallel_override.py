# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for explicit data-parallel sizing during heterogeneous startup."""

import sys
from types import SimpleNamespace

import pytest

from examples.mimo import pretrain_mimo
from megatron.training.arguments import parse_args, validate_args


def _minimal_training_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_arguments_data_parallel_override.py"])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 1024
    args.world_size = 8
    return args


def test_validate_args_uses_data_parallel_override_for_batch_sizes(monkeypatch):
    args = _minimal_training_args(monkeypatch)

    validate_args(args, data_parallel_size_override=2)

    assert args.data_parallel_size == 2
    assert args.global_batch_size == 2
    assert args.eval_global_batch_size == 2


@pytest.mark.parametrize("override", [0, -1, 1.5, True])
def test_validate_args_rejects_invalid_data_parallel_override(override):
    with pytest.raises(ValueError, match="positive integer"):
        validate_args(SimpleNamespace(), data_parallel_size_override=override)


@pytest.mark.parametrize(
    ("field", "value"),
    (("eval_global_batch_size", 0), ("eval_micro_batch_size", 0)),
)
def test_validate_args_rejects_zero_evaluation_batch_dimension(
    monkeypatch, field, value
):
    args = _minimal_training_args(monkeypatch)
    setattr(args, field, value)

    with pytest.raises(AssertionError, match="must be greater than zero"):
        validate_args(args, data_parallel_size_override=2)


def test_mimo_validates_grid_before_stock_args_and_passes_llm_dp(mocker, monkeypatch):
    class ValidationStopped(Exception):
        pass

    args = SimpleNamespace(world_size=8, llm_dp=2)
    events = []
    monkeypatch.setenv("WORLD_SIZE", "8")
    mocker.patch.object(pretrain_mimo, "parse_args", return_value=args)
    mocker.patch.object(
        pretrain_mimo,
        "validate_hetero_grid_args",
        side_effect=lambda actual, world: events.append(("hetero", actual, world)),
    )

    def stop_stock_validation(actual, defaults, data_parallel_size_override=None):
        events.append(("stock", actual, data_parallel_size_override))
        raise ValidationStopped

    mocker.patch.object(
        pretrain_mimo, "validate_args", side_effect=stop_stock_validation
    )

    with pytest.raises(ValidationStopped):
        pretrain_mimo._parse_and_validate()

    assert events == [("hetero", args, 8), ("stock", args, 2)]
