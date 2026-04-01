# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModuleMemoryConfig:
    """Memory optimization config for a MIMO module.

    Two categories of fields:

    1. TransformerConfig pass-through — stamped onto each module's TransformerConfig
       before construction so TransformerBlock/TransformerLayer read them natively.

    2. MIMO boundary — controls how combined_embeddings (the merged tensor fed to
       the LLM) is handled. Either recompute (checkpoint align_embeddings, frees the
       combined_embeddings intermediate) or offload (CPU offload via PipelineOffloadManager,
       frees both combined_embeddings AND the projection output tensors).
    """

    # TransformerConfig pass-through (encoder internals / LLM internals)
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: Optional[List[str]] = None
    offload_modules: Optional[List[str]] = None

    # MIMO boundary (mutually exclusive)
    recompute_combined_embeddings: bool = False
    offload_combined_embeddings: bool = False

    def __post_init__(self):
        if self.recompute_combined_embeddings and self.offload_combined_embeddings:
            raise ValueError(
                "recompute_combined_embeddings and offload_combined_embeddings are "
                "mutually exclusive. Use offload to free both projection_output and "
                "combined_embeddings, or recompute to free only combined_embeddings."
            )
