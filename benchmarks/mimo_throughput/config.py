# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen dataclasses for MIMO throughput benchmark configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ModuleArch:
    """Architecture specification for a single module (encoder or LLM)."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    vocab_size: int = 0  # 0 for vision encoder (no embedding layer)


@dataclass(frozen=True)
class ParallelSpec:
    """Parallelism specification for a single module."""

    tp: int = 1
    dp: int = 1
    pp: int = 1  # encoder always 1, LLM can be >1
    # CP=1 always for now

    @property
    def world_size(self) -> int:
        return self.tp * self.dp * self.pp


@dataclass(frozen=True)
class DataSpec:
    """Data configuration for the benchmark."""

    micro_batch_size: int  # per LLM DP replica
    num_microbatches: int
    image_token_id: int = 32000


@dataclass(frozen=True)
class ExperimentSpec:
    """Experiment metadata and run parameters."""

    name: str
    num_iterations: int = 10
    warmup_iterations: int = 2
    log_interval: int = 1


@dataclass(frozen=True)
class ModuleMemorySpec:
    """Memory optimization for one module (encoder or LLM).

    Fields map 1:1 to ModuleMemoryConfig.  None/empty means "leave default".
    """

    # Recompute (TransformerLayer internals)
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: Optional[List[str]] = None

    # Fine-grained offload (TransformerLayer internals)
    offload_modules: Optional[List[str]] = None

    # MIMO inter-module boundary
    recompute_projection: bool = False
    offload_projection: bool = False
    offload_encoder_output: bool = False


@dataclass(frozen=True)
class MemorySpec:
    """Per-module memory optimization config for the benchmark."""

    encoder: Optional[ModuleMemorySpec] = None
    llm: Optional[ModuleMemorySpec] = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration combining all specs."""

    experiment: ExperimentSpec
    encoder_arch: ModuleArch
    llm_arch: ModuleArch
    encoder_parallel: ParallelSpec
    llm_parallel: ParallelSpec
    data: DataSpec
    memory: Optional[MemorySpec] = None

    @property
    def llm_has_pp(self) -> bool:
        return self.llm_parallel.pp > 1

    @property
    def global_batch_size(self) -> int:
        return self.data.micro_batch_size * self.llm_parallel.dp

    def validate(self):
        """Validate cross-field constraints across the benchmark configuration."""
        # Encoder must be PP=1
        assert self.encoder_parallel.pp == 1, "Encoder must have PP=1"

        # All GPUs accounted for: encoder TP*DP == LLM TP*DP*PP
        assert (
            self.encoder_parallel.tp * self.encoder_parallel.dp
            == self.llm_parallel.world_size
        ), (
            f"Encoder world_size ({self.encoder_parallel.tp * self.encoder_parallel.dp}) "
            f"!= LLM world_size ({self.llm_parallel.world_size})"
        )

        # Global batch must be divisible by both encoder and LLM DP
        gbs = self.global_batch_size
        assert gbs % self.encoder_parallel.dp == 0, (
            f"Global batch {gbs} not divisible by encoder DP {self.encoder_parallel.dp}"
        )
        assert gbs % self.llm_parallel.dp == 0, (
            f"Global batch {gbs} not divisible by LLM DP {self.llm_parallel.dp}"
        )
