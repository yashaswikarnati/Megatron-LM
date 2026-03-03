# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for Nemotron6-MoE VLM with RADIO vision encoder.

Assembles a MIMO model consisting of:
- RADIO ViT vision encoder (1280 hidden, pixel shuffle → 5120)
- MLP projector (5120 → language hidden size)
- Nemotron6-MoE hybrid Mamba language model
"""

import torch

from configs.nemotron_moe_vlm import (
    get_nemotron_moe_language_model_config,
    get_nemotron_moe_language_layer_spec,
    get_radio_vision_config,
    get_radio_vision_layer_spec,
    get_vlm_projection_config,
    get_vlm_projection_layer_spec,
)

from examples.mimo.model_providers.radio_encoder import RADIOEncoderWrapper
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec


def add_nemotron_moe_vlm_args(parser):
    """Add Nemotron-MoE VLM specific arguments.

    These are args specific to the RADIO + Mamba-MoE model provider.
    Note: --img-h, --img-w, --patch-dim are already in Megatron core args.
    """
    group = parser.add_argument_group(
        'Nemotron-MoE VLM', 'Nemotron-MoE VLM model provider arguments'
    )
    group.add_argument('--pixel-shuffle', action='store_true', default=False,
                       help='Apply pixel shuffle post-processing to vision encoder output')
    group.add_argument('--max-num-tiles', type=int, default=1,
                       help='Max number of image tiles')
    group.add_argument('--use-tiling', action='store_true', default=False,
                       help='Enable image tiling')
    group.add_argument('--use-thumbnail', action='store_true', default=False,
                       help='Enable thumbnail generation')
    group.add_argument('--disable-vision-class-token', action='store_true', default=False,
                       help='Do not drop class tokens from vision encoder')
    group.add_argument('--freeze-lm', action='store_true', default=False,
                       help='Freeze language model parameters')
    group.add_argument('--freeze-vit', action='store_true', default=False,
                       help='Freeze vision encoder parameters')
    group.add_argument('--vision-model-type', type=str, default='radio',
                       help='Vision model type (e.g. radio)')
    group.add_argument('--class-token-len', type=int, default=None,
                       help='Number of class tokens in vision encoder')
    return parser


def model_provider_nemotron_moe_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    **kwargs,
):
    """Build a Nemotron6-MoE VLM MIMO model.

    Composes RADIO vision encoder + MLP projector + MambaModel (hybrid
    Mamba-MoE) into a single :class:`MimoModel`.

    Args:
        pre_process: Include embedding layer (pipeline parallelism).
        post_process: Include output layer (pipeline parallelism).
        add_encoder: Unused (PP not yet supported in MIMO).
        add_decoder: Unused (PP not yet supported in MIMO).

    Reads ``--image-token-id`` directly from CLI args (same source as the
    data provider) to keep both sides in sync.
    """
    from megatron.training import get_args

    args = get_args()

    # Read image_token_id directly from CLI args — single source of truth
    # shared with the data provider (energon_multimodal_provider).
    image_special_token_id = args.image_token_id

    # ── Configs ──────────────────────────────────────────────────────────
    # Language config: Nemotron defaults + CLI overrides (parallelism,
    # precision, and any explicitly-set arch fields) — all handled inside.
    language_config = get_nemotron_moe_language_model_config(args)

    # Vision / projection: fixed architectures, only sync precision.
    vision_config = get_radio_vision_config()
    projection_config = get_vlm_projection_config(
        hidden_size=language_config.hidden_size,
    )
    for cfg in (vision_config, projection_config):
        cfg.params_dtype = language_config.params_dtype
        cfg.bf16 = language_config.bf16
        cfg.fp16 = language_config.fp16
        cfg.use_cpu_initialization = language_config.use_cpu_initialization
        cfg.perform_initialization = language_config.perform_initialization

    # ── Vision encoder (RADIO) ───────────────────────────────────────────
    # Image args from CLI
    img_h = getattr(args, "img_h", 512)
    img_w = getattr(args, "img_w", 512)
    patch_dim = getattr(args, "patch_dim", 16)
    apply_pixel_shuffle = getattr(args, "pixel_shuffle", False)
    class_token_len = getattr(args, "class_token_len", 8) or 8
    disable_vision_class_token = getattr(args, "disable_vision_class_token", False)

    # After pixel shuffle: hidden * 4 = 1280 * 4 = 5120
    vision_input_size = vision_config.hidden_size * 4 if apply_pixel_shuffle else vision_config.hidden_size

    vision_encoder = ModuleSpec(
        module=RADIOEncoderWrapper,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_radio_vision_layer_spec(),
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
            "class_token_len": class_token_len,
            "drop_class_token": disable_vision_class_token,
            "apply_pixel_shuffle": apply_pixel_shuffle,
        },
    )

    # ── Vision → language projection ─────────────────────────────────────
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_vlm_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_input_size,
        },
    )

    # ── Vision submodule spec ────────────────────────────────────────────
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"radio_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    # ── Language model (MambaModel for hybrid Mamba-MoE) ─────────────────
    hybrid_override_pattern = getattr(args, "hybrid_override_pattern", None)

    language_model_spec = ModuleSpec(
        module=MambaModel,
        params={
            "config": language_config,
            "mamba_stack_spec": get_nemotron_moe_language_layer_spec(),
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "hybrid_override_pattern": hybrid_override_pattern,
            "position_embedding_type": "none",
            # Disable scatter in embedding — MIMO combines modality embeddings
            # at full sequence length, then scatters to SP in forward().
            "scatter_embedding_sequence_parallel": False,
        },
    )

    # ── Assemble MIMO model ──────────────────────────────────────────────
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id},
    )

    mimo_model = MimoModel(mimo_model_config)
    print("*" * 100)
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # ── Load pre-trained checkpoints ──────────────────────────────────────
    if getattr(args, "language_model_checkpoint", None) is not None:
        load_submodule_ckpt(mimo_model.language_model, args.language_model_checkpoint)
        print(f"Successfully loaded language model from {args.language_model_checkpoint}")

    if getattr(args, "vision_encoder_checkpoint", None) is not None:
        load_submodule_ckpt(
            mimo_model.modality_submodules.images.encoders.radio_encoder.radio_model,
            args.vision_encoder_checkpoint,
            ignore_missing_keys=("class_token",),
        )
        print(f"Successfully loaded vision encoder from {args.vision_encoder_checkpoint}")

    # ── Freeze / unfreeze based on CLI flags ─────────────────────────────
    if getattr(args, "freeze_vit", False):
        for p in mimo_model.modality_submodules.images.encoders.radio_encoder.parameters():
            p.requires_grad = False

    if getattr(args, "freeze_lm", False):
        for p in mimo_model.language_model.parameters():
            p.requires_grad = False

    return mimo_model
