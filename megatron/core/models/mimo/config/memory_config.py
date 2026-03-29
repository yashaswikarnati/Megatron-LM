# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModuleMemoryConfig:
    """Memory optimization config for one MIMO module (encoder or LLM).

    Recompute and fine-grained offload fields map directly to TransformerConfig
    and are applied before module construction.  MIMO-specific fields
    (recompute_projection, offload_projection, offload_encoder_output) are
    handled by ModalitySubmodules wrappers at the inter-module boundary.

    Args:
        recompute_granularity: 'full' or 'selective'. None disables recompute.
        recompute_method: 'uniform' or 'block'. Only used when granularity='full'.
        recompute_num_layers: Number of layers to recompute.
        recompute_modules: Sub-modules to selectively recompute
            (e.g. ['core_attn', 'mlp']). Only used when granularity='selective'.
        offload_modules: Sub-modules whose saved tensors are offloaded to CPU
            (e.g. ['attn_norm', 'qkv_linear', 'core_attn']).
        recompute_projection: Wrap projection forward with checkpoint().
            Saves MLP intermediates for MLP-type projections.
        offload_projection: Offload projection's saved-for-backward tensors
            (including encoder output) to CPU via FineGrainedActivationOffloadingInterface.
        offload_encoder_output: Offload encoder input saved tensors to CPU
            via FineGrainedActivationOffloadingInterface.
        recompute_projection_output: Wrap align_embeddings_by_token_positions with
            checkpoint(). Discards combined_embeddings intermediate ([B, S, H]).
            Cheap to replay (just indexing + masked_scatter). Does NOT cascade
            back through the projection — checkpoint saves projection_output as input.
        offload_projection_output: Offload align_embeddings saved tensors to CPU
            via FineGrainedActivationOffloadingInterface.
    """

    # Recompute (maps to TransformerConfig)
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: Optional[List[str]] = None

    # Fine-grained offload (maps to TransformerConfig)
    offload_modules: Optional[List[str]] = None

    # MIMO inter-module boundary
    recompute_projection: bool = False
    offload_projection: bool = False
    offload_encoder_output: bool = False
    recompute_projection_output: bool = False
    offload_projection_output: bool = False
