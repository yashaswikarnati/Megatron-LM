# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen dataclasses for MIMO throughput benchmark configuration."""

from dataclasses import dataclass


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
    num_images_per_sample: int = 1  # images per text sample; scales encoder batch


@dataclass(frozen=True)
class ExperimentSpec:
    """Experiment metadata and run parameters."""

    name: str
    num_iterations: int = 10
    warmup_iterations: int = 2
    log_interval: int = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration combining all specs."""

    experiment: ExperimentSpec
    encoder_arch: ModuleArch
    llm_arch: ModuleArch
    encoder_parallel: ParallelSpec
    llm_parallel: ParallelSpec
    data: DataSpec

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

        # Global batch must be divisible by LLM DP (always true by construction)
        gbs = self.global_batch_size
        assert gbs % self.llm_parallel.dp == 0, (
            f"Global batch {gbs} not divisible by LLM DP {self.llm_parallel.dp}"
        )

        # For encoder DP: with multi-image, images from one LLM sample can be
        # split across encoder DP ranks. The real constraint is that the total
        # encoder batch (num_images * mbs) is divisible by the fan-in scale,
        # which is checked below. The old GBS % enc_dp check was overly strict
        # for num_images > 1.
        enc_dp = self.encoder_parallel.dp
        llm_dp = self.llm_parallel.dp
        total_encoder_samples = self.data.num_images_per_sample * gbs
        assert total_encoder_samples % enc_dp == 0, (
            f"Total encoder samples (num_images={self.data.num_images_per_sample} * GBS={gbs}"
            f" = {total_encoder_samples}) not divisible by encoder DP {enc_dp}"
        )

        # Multi-image: total image tokens must fit in LLM seq_length
        total_image_tokens = self.data.num_images_per_sample * self.encoder_arch.seq_length
        assert total_image_tokens <= self.llm_arch.seq_length, (
            f"num_images({self.data.num_images_per_sample}) * enc_seq({self.encoder_arch.seq_length})"
            f" = {total_image_tokens} exceeds llm_seq({self.llm_arch.seq_length})"
        )

        # Fan-in: encoder batch (num_images * mbs) must be divisible by scale
        if enc_dp > llm_dp:
            scale = enc_dp // llm_dp
            encoder_batch = self.data.num_images_per_sample * self.data.micro_batch_size
            assert encoder_batch % scale == 0, (
                f"num_images({self.data.num_images_per_sample}) * mbs({self.data.micro_batch_size})"
                f" = {encoder_batch} not divisible by fan-in scale {scale}"
            )
