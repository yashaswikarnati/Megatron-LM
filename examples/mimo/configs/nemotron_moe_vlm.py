# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the Nemotron6-MoE VLM with RADIO vision encoder.

Provides TransformerConfig builders for:
- Nemotron6-MoE (3B hybrid Mamba-MoE) language model
- RADIO ViT vision encoder
- Vision-to-language MLP projector
- Layer specs for each component
"""

import dataclasses
from typing import Optional

import torch

from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


# ---------------------------------------------------------------------------
# Language model config: Nemotron6-MoE (3B hybrid Mamba-MoE)
# ---------------------------------------------------------------------------
def get_nemotron_moe_language_model_config(args=None) -> TransformerConfig:
    """Return a TransformerConfig for **Nemotron6-MoE** (3B hybrid Mamba-MoE).

    Builds the config in two layers:

    1. **Nemotron defaults** — model-specific values (MoE routing, Mamba heads,
       hidden sizes, etc.) that define the 3B architecture.
    2. **CLI overrides** — if ``args`` (the Megatron args namespace) is provided,
       any field whose value differs from TransformerConfig's dataclass default
       is applied on top.  This covers parallelism, precision, and any arch
       field the training script explicitly sets.

    When ``args`` is ``None`` the pure Nemotron defaults are returned (useful
    for unit tests or standalone usage).
    """
    num_layers = getattr(args, "num_layers", 52) if args else 52

    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=2688,
        num_attention_heads=32,
    )

    # GQA
    cfg.num_query_groups = 2
    cfg.kv_channels = 128

    # FFN
    cfg.ffn_hidden_size = 1856
    cfg.activation_func = squared_relu
    cfg.gated_linear_unit = False

    # Normalisation
    cfg.normalization = "RMSNorm"

    # No bias
    cfg.add_bias_linear = False

    # Mamba SSM
    cfg.mamba_num_heads = 64
    cfg.mamba_head_dim = 64

    # MoE
    cfg.num_moe_experts = 128
    cfg.moe_ffn_hidden_size = 1856
    cfg.moe_router_topk = 6
    cfg.moe_grouped_gemm = True
    cfg.moe_router_score_function = "sigmoid"
    cfg.moe_router_topk_scaling_factor = 2.5
    cfg.moe_router_enable_expert_bias = True
    cfg.moe_router_dtype = "fp32"
    cfg.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.moe_aux_loss_coeff = 0.0001
    cfg.moe_shared_expert_intermediate_size = 3712
    cfg.moe_shared_expert_overlap = True
    cfg.moe_token_dispatcher_type = "alltoall"

    # Positional embeddings (Mamba handles position internally)
    cfg.position_embedding_type = "none"

    # Sequence length
    cfg.seq_length = 4096
    cfg.max_position_embeddings = 4096

    # Dropout
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    # TE / kernel fusions
    cfg.bias_activation_fusion = False
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = False

    # ── CLI overrides ───────────────────────────────────────────────────
    # Override Nemotron defaults with values from CLI args.  A CLI value
    # is considered "explicitly set" when it differs from the
    # TransformerConfig dataclass default — this preserves Nemotron-
    # specific values (e.g. moe_router_topk=6) when the script doesn't
    # pass them (CLI default also equals the dataclass default of 2).
    if args is not None:
        for field in dataclasses.fields(TransformerConfig):
            if field.default is dataclasses.MISSING:
                continue
            arg_val = getattr(args, field.name, None)
            if arg_val is None:
                continue
            if arg_val == field.default and getattr(cfg, field.name) != field.default:
                continue
            setattr(cfg, field.name, arg_val)

    return cfg


# ---------------------------------------------------------------------------
# Vision encoder config: RADIO ViT
# ---------------------------------------------------------------------------
def get_radio_vision_config(
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the **RADIO** vision encoder.

    Parameters match ``examples/multimodal/config.py`` for ``vision_model_type == "radio"``.
    """
    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
    )

    cfg.kv_channels = 80
    cfg.num_query_groups = 16
    cfg.ffn_hidden_size = 5120
    cfg.gated_linear_unit = False
    cfg.activation_func = fast_gelu

    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True

    cfg.normalization = "LayerNorm"
    cfg.layernorm_epsilon = 1e-6
    cfg.layernorm_zero_centered_gamma = False

    cfg.apply_rope_fusion = False
    cfg.qk_layernorm = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True

    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


# ---------------------------------------------------------------------------
# Vision → language projection MLP
# ---------------------------------------------------------------------------
def get_vlm_projection_config(
    hidden_size: int = 2688,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision→language projection MLP.

    ``hidden_size`` should match the language model's hidden size.

    Must match the original pretrain_vlm.py architecture:
    - activation_func = squared_relu (inherited from language model base config)
    - normalization = "RMSNorm" (inherited from language model base config)
    - bias_activation_fusion = False
    - bias_dropout_fusion = False
    """
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=1,
    )
    cfg.ffn_hidden_size = 4 * 5120
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.add_bias_linear = False
    cfg.activation_func = squared_relu
    cfg.normalization = "RMSNorm"

    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


# ---------------------------------------------------------------------------
# Layer specs
# ---------------------------------------------------------------------------
def get_radio_vision_layer_spec() -> ModuleSpec:
    """Layer spec for the RADIO ViT encoder (Transformer-Engine)."""
    return get_vit_layer_with_transformer_engine_spec()


def get_nemotron_moe_language_layer_spec() -> ModuleSpec:
    """Layer spec for the Nemotron6-MoE hybrid Mamba stack.

    Returns the ``mamba_stack_spec`` from ``mamba_layer_specs`` which
    supports Mamba / attention / MLP / MoE layers via
    ``hybrid_override_pattern``.
    """
    return mamba_stack_spec


def get_vlm_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision→language projection MLP.

    Uses TELayerNormColumnParallelLinear for fc1 to match the original
    pretrain_vlm.py architecture (examples/multimodal/model.py). The fused
    layer norm normalizes the vision encoder output before the first linear
    layer, which is critical for training stability.
    """
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
