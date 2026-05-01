# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for MIMO throughput benchmark configuration helpers."""

import textwrap

import pytest

from benchmarks.mimo_throughput.config import (
    BenchmarkConfig,
    DataSpec,
    ExperimentSpec,
    ModuleArch,
    ParallelSpec,
)
from benchmarks.mimo_throughput.config_loader import load_config
from benchmarks.mimo_throughput.data import (
    compute_non_colocated_encoder_batch_size,
    get_non_colocated_data_ownership,
)


def _benchmark_config(**overrides):
    fields = {
        "experiment": ExperimentSpec(name="unit_non_colocated"),
        "encoder_arch": ModuleArch(
            num_layers=1, hidden_size=8, num_attention_heads=1, seq_length=4
        ),
        "llm_arch": ModuleArch(
            num_layers=1, hidden_size=8, num_attention_heads=1, seq_length=16, vocab_size=128
        ),
        "encoder_parallel": ParallelSpec(tp=1, dp=2, pp=1, cp=1, offset=0),
        "llm_parallel": ParallelSpec(tp=2, dp=2, pp=1, cp=1, offset=2),
        "data": DataSpec(micro_batch_size=2, num_microbatches=2, num_images_per_sample=1),
        "pp_mode": "non_colocated",
        "sequence_parallel": False,
    }
    fields.update(overrides)
    return BenchmarkConfig(**fields)


def test_non_colocated_config_accepts_disjoint_offsets():
    cfg = _benchmark_config()

    cfg.validate(world_size=6)


def test_non_colocated_config_rejects_overlapping_ranges():
    cfg = _benchmark_config(llm_parallel=ParallelSpec(tp=2, dp=2, pp=1, cp=1, offset=1))

    with pytest.raises(AssertionError, match="must not overlap"):
        cfg.validate(world_size=6)


def test_non_colocated_config_rejects_world_size_mismatch():
    cfg = _benchmark_config()

    with pytest.raises(AssertionError, match="must equal distributed world_size"):
        cfg.validate(world_size=7)


def test_non_colocated_config_accepts_sequence_parallel():
    cfg = _benchmark_config(sequence_parallel=True)

    cfg.validate(world_size=6)


def test_non_colocated_sequence_parallel_rejects_misaligned_llm_offset():
    cfg = _benchmark_config(
        llm_parallel=ParallelSpec(tp=2, dp=2, pp=1, cp=1, offset=3), sequence_parallel=True
    )

    with pytest.raises(AssertionError, match="LLM rank offset .* divisible by LLM TP"):
        cfg.validate(world_size=7)


def test_non_colocated_sequence_parallel_rejects_world_size_not_tp_divisible():
    cfg = _benchmark_config(
        encoder_parallel=ParallelSpec(tp=1, dp=1, pp=1, cp=1, offset=4),
        llm_parallel=ParallelSpec(tp=2, dp=2, pp=1, cp=1, offset=0),
        sequence_parallel=True,
    )

    with pytest.raises(AssertionError, match="distributed world_size .* divisible by LLM TP"):
        cfg.validate(world_size=5)


def test_loader_accepts_parallelism_offsets(tmp_path):
    config_path = tmp_path / "non_colocated.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: yaml_non_colocated
            model:
              encoder:
                num_layers: 1
                hidden_size: 8
                num_attention_heads: 1
                seq_length: 4
              llm:
                num_layers: 1
                hidden_size: 8
                num_attention_heads: 1
                seq_length: 16
                vocab_size: 128
            parallelism:
              encoder:
                tp: 1
                dp: 2
                offset: 0
              llm:
                tp: 2
                dp: 2
                offset: 2
            data:
              micro_batch_size: 2
              num_microbatches: 2
              num_images_per_sample: 1
            pp_mode: non_colocated
            sequence_parallel: false
            """
        )
    )

    cfg = load_config(str(config_path))

    assert cfg.encoder_parallel.offset == 0
    assert cfg.llm_parallel.offset == 2
    cfg.validate(world_size=6)


def test_non_colocated_encoder_batch_size_helper():
    assert (
        compute_non_colocated_encoder_batch_size(
            micro_batch_size=1, num_images_per_sample=4, encoder_dp=8, llm_dp=64
        )
        == 32
    )


def test_non_colocated_encoder_batch_size_helper_rejects_uneven_split():
    with pytest.raises(ValueError, match="must be divisible by encoder DP"):
        compute_non_colocated_encoder_batch_size(
            micro_batch_size=1, num_images_per_sample=1, encoder_dp=3, llm_dp=2
        )


def test_non_colocated_data_ownership_matches_multimodule_schedule_pattern():
    assert get_non_colocated_data_ownership(
        in_encoder_grid=True,
        encoder_pp_rank=0,
        in_llm_grid=False,
        llm_pp_rank=None,
        llm_pp_size=None,
    ) == (False, True)

    assert get_non_colocated_data_ownership(
        in_encoder_grid=False, encoder_pp_rank=None, in_llm_grid=True, llm_pp_rank=0, llm_pp_size=4
    ) == (True, False)

    assert get_non_colocated_data_ownership(
        in_encoder_grid=False, encoder_pp_rank=None, in_llm_grid=True, llm_pp_rank=2, llm_pp_size=4
    ) == (False, False)

    assert get_non_colocated_data_ownership(
        in_encoder_grid=False, encoder_pp_rank=None, in_llm_grid=True, llm_pp_rank=3, llm_pp_size=4
    ) == (True, False)
