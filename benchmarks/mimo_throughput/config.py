# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen dataclasses for MIMO throughput benchmark configuration."""

import os
from dataclasses import dataclass
from typing import Any, List, Optional


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
    cp: int = 1
    offset: int = 0

    @property
    def world_size(self) -> int:
        return self.tp * self.dp * self.pp * self.cp


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
class ModuleMemorySpec:
    """Memory optimization for one module (encoder or LLM).

    Fields map 1:1 to ModuleMemoryConfig. None/empty means "leave default".
    """

    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: Optional[List[str]] = None
    offload_modules: Optional[List[str]] = None
    recompute_combined_embeddings: bool = False
    offload_combined_embeddings: bool = False


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
    pp_mode: str = "colocated"  # "colocated", "non_colocated", or "homo"
    encoder_num_dist_opt_instances: int = 1
    encoder_use_distributed_optimizer: bool = True
    encoder_offload: bool = False  # Offload encoder DDP params + optimizer states to CPU
    sequence_parallel: bool = True  # SP + tp_comm_overlap on LLM (should always be on for TP>1)
    pipeline_timers: bool = False  # Enable per-microbatch fwd/bwd timers (profiling only)
    llm_num_layers_in_first_pipeline_stage: Optional[int] = None
    llm_num_layers_in_last_pipeline_stage: Optional[int] = None
    llm_pipeline_model_parallel_layout: Optional[Any] = None

    @property
    def llm_has_pp(self) -> bool:
        return self.llm_parallel.pp > 1

    @property
    def dp_batch_size(self) -> int:
        """Samples per microbatch across all DP ranks: mbs × llm_dp."""
        return self.data.micro_batch_size * self.llm_parallel.dp

    @property
    def global_batch_size(self) -> int:
        """Total samples per optimizer step: mbs × llm_dp × num_microbatches."""
        return self.dp_batch_size * self.data.num_microbatches

    @property
    def llm_has_custom_pipeline_split(self) -> bool:
        """Whether LLM PP uses Megatron's uneven/custom layer placement."""
        return (
            self.llm_num_layers_in_first_pipeline_stage is not None
            or self.llm_num_layers_in_last_pipeline_stage is not None
            or self.llm_pipeline_model_parallel_layout is not None
        )

    def validate(self, world_size: Optional[int] = None):
        """Validate cross-field constraints across the benchmark configuration."""
        if world_size is None and "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])

        assert self.pp_mode in ("colocated", "non_colocated", "homo"), (
            f"pp_mode must be 'colocated', 'non_colocated', or 'homo', got '{self.pp_mode}'"
        )

        # Encoder must be PP=1
        assert self.encoder_parallel.pp == 1, "Encoder must have PP=1"
        self._validate_llm_pipeline_split()

        assert self.encoder_parallel.offset >= 0, "Encoder rank offset must be non-negative"
        assert self.llm_parallel.offset >= 0, "LLM rank offset must be non-negative"
        if self.pp_mode in ("colocated", "homo"):
            assert self.encoder_parallel.offset == 0 and self.llm_parallel.offset == 0, (
                f"{self.pp_mode} mode does not use rank offsets; set both offsets to 0"
            )

        # Colocated/non-colocated bridge communicator does not support encoder CP yet.
        # Homo: encoder CP must match LLM CP (parallel_state forces same CP on both)
        if self.pp_mode in ("colocated", "non_colocated"):
            assert self.encoder_parallel.cp == 1, (
                f"{self.pp_mode} mode requires encoder CP=1 (bridge communicator limitation)"
            )
        if self.pp_mode == "non_colocated":
            assert self.llm_parallel.cp == 1, (
                "non_colocated mode requires LLM CP=1 (bridge communicator limitation)"
            )

        # Encoder offload requires colocated mode with LLM PP > 1
        if self.encoder_offload:
            assert self.pp_mode == "colocated" and self.llm_parallel.pp > 1, (
                "encoder_offload requires colocated mode with LLM PP > 1"
            )

        if self.pp_mode == "homo":
            # Homo: encoder shares TP/DP with LLM (lives only on PP stage 0)
            assert self.encoder_parallel.tp == self.llm_parallel.tp, (
                f"Homo mode requires encoder TP ({self.encoder_parallel.tp}) "
                f"== LLM TP ({self.llm_parallel.tp})"
            )
            assert self.encoder_parallel.dp == self.llm_parallel.dp, (
                f"Homo mode requires encoder DP ({self.encoder_parallel.dp}) "
                f"== LLM DP ({self.llm_parallel.dp})"
            )
            assert self.encoder_parallel.cp == self.llm_parallel.cp, (
                f"Homo mode requires encoder CP ({self.encoder_parallel.cp}) "
                f"== LLM CP ({self.llm_parallel.cp})"
            )
        elif self.pp_mode == "colocated":
            # Colocated: all GPUs accounted for: encoder world_size == LLM world_size
            assert (
                self.encoder_parallel.world_size
                == self.llm_parallel.world_size
            ), (
                f"Encoder world_size ({self.encoder_parallel.world_size}) "
                f"!= LLM world_size ({self.llm_parallel.world_size})"
            )
            assert self.encoder_parallel.offset == self.llm_parallel.offset, (
                "Colocated mode requires encoder and LLM rank offsets to match"
            )
        else:
            self._validate_non_colocated_placement(world_size)

        if world_size is not None and self.pp_mode in ("colocated", "homo"):
            assert self.llm_parallel.world_size == world_size, (
                f"{self.pp_mode} LLM world_size ({self.llm_parallel.world_size}) "
                f"must equal distributed world_size ({world_size})"
            )

        # Per-microbatch samples must be divisible by LLM DP (always true by construction)
        dp_bs = self.dp_batch_size
        assert dp_bs % self.llm_parallel.dp == 0, (
            f"dp_batch_size {dp_bs} not divisible by LLM DP {self.llm_parallel.dp}"
        )

        # For encoder DP: with multi-image, images from one LLM sample can be
        # split across encoder DP ranks. At PP=1 and non-colocated, each
        # microbatch is processed independently so we check per-dp-batch. At
        # colocated PP>1, the schedule concatenates ALL microbatches before
        # distributing to encoder ranks, so the total (num_images * GBS) is the
        # correct check.
        enc_dp = self.encoder_parallel.dp
        llm_dp = self.llm_parallel.dp
        if self.pp_mode == "colocated" and self.llm_parallel.pp > 1:
            total_encoder_samples = (self.data.num_images_per_sample
                                     * self.global_batch_size)
        else:
            total_encoder_samples = self.data.num_images_per_sample * dp_bs
        assert total_encoder_samples % enc_dp == 0, (
            f"Total encoder samples (num_images={self.data.num_images_per_sample}"
            f" * {'GBS=' + str(self.global_batch_size) if self.llm_parallel.pp > 1 else 'dp_batch_size=' + str(dp_bs)}"
            f" = {total_encoder_samples}) not divisible by encoder DP {enc_dp}"
        )

        # Multi-image: total image tokens must fit in LLM seq_length
        total_image_tokens = self.data.num_images_per_sample * self.encoder_arch.seq_length
        assert total_image_tokens <= self.llm_arch.seq_length, (
            f"num_images({self.data.num_images_per_sample}) * enc_seq({self.encoder_arch.seq_length})"
            f" = {total_image_tokens} exceeds llm_seq({self.llm_arch.seq_length})"
        )

        # CP: world_size must be divisible by CP
        cp = self.llm_parallel.cp
        if cp > 1:
            ws = self.llm_parallel.world_size
            assert ws % cp == 0, (
                f"LLM world_size ({ws}) must be divisible by CP ({cp})"
            )
            # LLM seq_length must be divisible by CP
            assert self.llm_arch.seq_length % cp == 0, (
                f"LLM seq_length ({self.llm_arch.seq_length}) must be divisible by CP ({cp})"
            )

        # Fan-in: encoder batch must be divisible by scale (enc_dp // llm_dp).
        # At PP=1 and non-colocated, forward_step processes one microbatch at
        # a time -> check per-mb. At colocated PP>1, the schedule concatenates
        # ALL microbatches before slicing by scale, so the total batch
        # (nmb * num_images * mbs) is the correct divisibility target.
        if enc_dp > llm_dp:
            scale = enc_dp // llm_dp
            if self.pp_mode == "colocated" and self.llm_parallel.pp > 1:
                encoder_batch = (self.data.num_images_per_sample
                                 * self.data.micro_batch_size
                                 * self.data.num_microbatches)
            else:
                encoder_batch = self.data.num_images_per_sample * self.data.micro_batch_size
            assert encoder_batch % scale == 0, (
                f"encoder_batch={encoder_batch} (num_images={self.data.num_images_per_sample}"
                f" * mbs={self.data.micro_batch_size}"
                f"{'* nmb=' + str(self.data.num_microbatches) if self.llm_parallel.pp > 1 else ''})"
                f" not divisible by fan-in scale {scale}"
                f" (enc_dp={enc_dp} / llm_dp={llm_dp})"
            )

    def _validate_llm_pipeline_split(self):
        """Validate LLM PP layer placement before constructing TransformerConfig."""
        pp = self.llm_parallel.pp
        num_layers = self.llm_arch.num_layers
        first = self.llm_num_layers_in_first_pipeline_stage
        last = self.llm_num_layers_in_last_pipeline_stage
        layout = self.llm_pipeline_model_parallel_layout

        has_uneven_edges = first is not None or last is not None
        has_layout = layout is not None
        assert not (has_uneven_edges and has_layout), (
            "llm_pipeline_model_parallel_layout cannot be set with "
            "llm_num_layers_in_first_pipeline_stage or "
            "llm_num_layers_in_last_pipeline_stage"
        )

        if has_layout:
            # Megatron validates decoder/embedding/loss placement when it builds
            # TransformerConfig. The harness only needs to avoid the even split
            # divisibility assertion.
            return

        if not has_uneven_edges:
            assert num_layers % pp == 0, (
                f"LLM num_layers ({num_layers}) must be divisible by LLM PP ({pp}) "
                "unless an LLM uneven/custom pipeline split is configured"
            )
            return

        remaining_layers = num_layers
        remaining_stages = pp

        if first is not None:
            assert first > 0, "llm_num_layers_in_first_pipeline_stage must be positive"
            remaining_layers -= first
            remaining_stages -= 1

        if last is not None:
            assert last > 0, "llm_num_layers_in_last_pipeline_stage must be positive"
            remaining_layers -= last
            remaining_stages -= 1

        assert remaining_layers >= 0, (
            "LLM uneven pipeline split assigns more edge-stage layers than "
            f"total layers ({num_layers})"
        )
        assert remaining_stages >= 0, (
            "LLM uneven pipeline split configures more edge stages than "
            f"pipeline stages ({pp})"
        )
        assert bool(remaining_layers) == bool(remaining_stages), (
            f"Mismatch: {remaining_layers} middle LLM layers remain but "
            f"{remaining_stages} middle PP stages are available"
        )
        if remaining_stages:
            assert remaining_layers % remaining_stages == 0, (
                f"Remaining LLM layers ({remaining_layers}) must be divisible by "
                f"remaining PP stages ({remaining_stages})"
            )

    def _validate_non_colocated_placement(self, world_size: Optional[int]):
        """Validate disjoint encoder/LLM rank ranges for non-colocated mode."""
        ep = self.encoder_parallel
        lp = self.llm_parallel

        assert ep.dp % lp.dp == 0 or lp.dp % ep.dp == 0, (
            f"Non-colocated mode requires encoder DP ({ep.dp}) and LLM DP ({lp.dp}) "
            "to be evenly divisible in one direction for bridge fan-in/fan-out"
        )
        if self.sequence_parallel and lp.tp > 1:
            assert lp.offset % lp.tp == 0, (
                "Non-colocated sequence parallel with TP communication overlap requires "
                f"LLM rank offset ({lp.offset}) to be divisible by LLM TP ({lp.tp})"
            )
            if world_size is not None:
                assert world_size % lp.tp == 0, (
                    "Non-colocated sequence parallel with TP communication overlap requires "
                    f"distributed world_size ({world_size}) to be divisible by LLM TP ({lp.tp})"
                )

        enc_start = ep.offset
        enc_end = enc_start + ep.world_size
        llm_start = lp.offset
        llm_end = llm_start + lp.world_size

        assert enc_end <= llm_start or llm_end <= enc_start, (
            "Non-colocated rank ranges must not overlap: "
            f"encoder=[{enc_start}, {enc_end}), llm=[{llm_start}, {llm_end})"
        )

        if world_size is not None:
            required_world_size = ep.world_size + lp.world_size
            assert required_world_size == world_size, (
                f"Non-colocated encoder+LLM world_size ({required_world_size}) must equal "
                f"distributed world_size ({world_size})"
            )
            assert enc_end <= world_size and llm_end <= world_size, (
                "Non-colocated rank ranges exceed distributed world_size: "
                f"encoder=[{enc_start}, {enc_end}), llm=[{llm_start}, {llm_end}), "
                f"world_size={world_size}"
            )
