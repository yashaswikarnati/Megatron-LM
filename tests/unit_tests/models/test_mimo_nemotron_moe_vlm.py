# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_nemotron_moe_vlm.py -xvs
'''

import torch
import torch.nn as nn

import pytest

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Import the RADIO encoder wrapper from examples
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'mimo'))
from model_providers.radio_encoder import RADIOEncoderWrapper


def get_tiny_radio_vision_config():
    """Tiny RADIO config for testing (1 layer, small hidden size)."""
    cfg = TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
    )
    cfg.kv_channels = 16
    cfg.num_query_groups = 4
    cfg.ffn_hidden_size = 256
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.normalization = "LayerNorm"
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    return cfg


def get_tiny_mamba_language_config():
    """Tiny MambaModel config for testing (2 layers, small hidden size)."""
    cfg = TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
    )
    cfg.kv_channels = 16
    cfg.num_query_groups = 2
    cfg.ffn_hidden_size = 64
    cfg.normalization = "RMSNorm"
    cfg.add_bias_linear = False
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    # Mamba SSM params: nheads = hidden_size * expand / mamba_head_dim
    # must satisfy nheads % mamba_num_groups == 0
    cfg.mamba_head_dim = 16  # nheads = 64*2/16 = 8
    cfg.mamba_num_groups = 8  # 8 % 8 == 0
    cfg.mamba_state_dim = 16
    return cfg


def get_radio_vision_submodule_spec(hidden_size, img_h, img_w, patch_dim, apply_pixel_shuffle):
    """Build vision submodule spec with RADIO encoder for testing."""
    vision_config = get_tiny_radio_vision_config()
    vision_layer_spec = get_vit_layer_with_transformer_engine_spec()

    # After pixel shuffle: hidden * 4
    vision_output_size = (
        vision_config.hidden_size * 4 if apply_pixel_shuffle else vision_config.hidden_size
    )

    vision_encoder_spec = ModuleSpec(
        module=RADIOEncoderWrapper,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
            "class_token_len": 8,
            "drop_class_token": True,
            "apply_pixel_shuffle": apply_pixel_shuffle,
        },
    )

    # Simple linear projection for test (vision hidden → language hidden)
    vision_projection_spec = ModuleSpec(
        module=nn.Linear, params={"in_features": vision_output_size, "out_features": hidden_size}
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"radio_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )
    return vision_submodule_spec


def get_mamba_language_model_spec(hidden_size, vocab_size, seq_len):
    """Build MambaModel spec with tiny config for testing."""
    language_config = get_tiny_mamba_language_config()

    language_model_spec = ModuleSpec(
        module=MambaModel,
        params={
            "config": language_config,
            "mamba_stack_spec": mamba_stack_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
            "hybrid_override_pattern": "MM",  # 2 Mamba layers
            "position_embedding_type": "none",
        },
    )
    return language_model_spec


def get_nemotron_moe_vlm_mimo_model(
    hidden_size,
    vocab_size,
    seq_len,
    img_h,
    img_w,
    patch_dim,
    special_token_ids,
    apply_pixel_shuffle=False,
):
    """Build a test-sized Nemotron-MoE VLM MIMO model."""
    language_model_spec = get_mamba_language_model_spec(hidden_size, vocab_size, seq_len)
    vision_submodule_spec = get_radio_vision_submodule_spec(
        hidden_size, img_h, img_w, patch_dim, apply_pixel_shuffle
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids=special_token_ids,
    )

    mimo_model = MimoModel(mimo_config)
    return mimo_model


class TestRADIOEncoderWrapper:
    """Test the RADIOEncoderWrapper standalone."""

    def setup_method(self, method):
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_constructor(self):
        """Test RADIOEncoderWrapper instantiation."""
        vision_config = get_tiny_radio_vision_config()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()

        wrapper = RADIOEncoderWrapper(
            transformer_config=vision_config,
            transformer_layer_spec=vision_layer_spec,
            img_h=64,
            img_w=64,
            patch_dim=16,
            class_token_len=8,
            drop_class_token=True,
            apply_pixel_shuffle=False,
        )
        assert hasattr(wrapper, 'radio_model')
        assert isinstance(wrapper.radio_model, RADIOViTModel)

    def test_forward_no_pixel_shuffle(self):
        """Test forward pass without pixel shuffle."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_config = get_tiny_radio_vision_config()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()

        img_h, img_w, patch_dim = 64, 64, 16
        class_token_len = 8
        num_patches = (img_h // patch_dim) * (img_w // patch_dim)  # 16

        wrapper = RADIOEncoderWrapper(
            transformer_config=vision_config,
            transformer_layer_spec=vision_layer_spec,
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            class_token_len=class_token_len,
            drop_class_token=True,
            apply_pixel_shuffle=False,
        ).to(device)

        # Input: [num_tiles, 3, img_h, img_w]
        x = torch.randn(2, 3, img_h, img_w, device=device)
        out = wrapper(x=x)

        # After dropping class tokens: [2, num_patches, hidden_size]
        assert out.shape == (2, num_patches, vision_config.hidden_size)

    def test_forward_with_pixel_shuffle(self):
        """Test forward pass with pixel shuffle (4x token reduction, 4x hidden increase)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_config = get_tiny_radio_vision_config()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()

        img_h, img_w, patch_dim = 64, 64, 16
        class_token_len = 8
        num_patches = (img_h // patch_dim) * (img_w // patch_dim)  # 16
        reduced_seq = num_patches // 4  # 4

        wrapper = RADIOEncoderWrapper(
            transformer_config=vision_config,
            transformer_layer_spec=vision_layer_spec,
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            class_token_len=class_token_len,
            drop_class_token=True,
            apply_pixel_shuffle=True,
        ).to(device)

        x = torch.randn(2, 3, img_h, img_w, device=device)
        out = wrapper(x=x)

        # After pixel shuffle: [2, reduced_seq, hidden_size * 4]
        assert out.shape == (2, reduced_seq, vision_config.hidden_size * 4)


class TestNemotronMoEVLM:
    """Test the full Nemotron-MoE VLM MIMO model (RADIO + MambaModel)."""

    def setup_method(self, method):
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")

        model_parallel_cuda_manual_seed(123)

        self.hidden_size = 64
        self.batch_size = 2
        self.seq_len = 256
        self.img_h = 64
        self.img_w = 64
        self.patch_dim = 16
        self.vocab_size = 48000
        self.special_token_ids = {"images": 50257}

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
        except Exception as e:
            print(f"Warning: Could not destroy model parallel: {e}")

    def test_constructor(self):
        """Test MimoModel construction with RADIO + MambaModel."""
        mimo_model = get_nemotron_moe_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mimo_model = mimo_model.to(device)

        # Check modality submodules
        assert "images" in mimo_model.modality_submodules
        assert isinstance(mimo_model.modality_submodules["images"], VisionModalitySubmodules)

        # Check encoder is RADIO
        assert hasattr(mimo_model.modality_submodules.images.encoders, "radio_encoder")
        assert isinstance(
            mimo_model.modality_submodules.images.encoders.radio_encoder, RADIOEncoderWrapper
        )

        # Check language model is MambaModel
        assert hasattr(mimo_model, "language_model")
        assert isinstance(mimo_model.language_model, MambaModel)

        # Check special tokens
        assert mimo_model.special_token_ids == self.special_token_ids

    def test_forward_text_only(self):
        """Test forward pass with only text (no images)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mimo_model = get_nemotron_moe_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        ).to(device)

        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        position_ids = (
            torch.arange(self.seq_len, device=device).unsqueeze(0).expand(self.batch_size, -1)
        )

        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=None
        )

        assert outputs is not None
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_images(self):
        """Test forward pass with RADIO-encoded images."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mimo_model = get_nemotron_moe_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        ).to(device)

        # RADIO with 64x64 images, patch_dim=16 → 16 patches, drop 8 class tokens → 16 patches
        # No pixel shuffle → hidden=64
        num_patches_per_image = (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim)
        num_images = 2  # 1 per batch item

        # Create images
        images = torch.randn(num_images, 3, self.img_h, self.img_w, device=device)

        # Create input_ids with image tokens
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        position_ids = (
            torch.arange(self.seq_len, device=device).unsqueeze(0).expand(self.batch_size, -1)
        )

        image_token_id = self.special_token_ids["images"]
        start_pos = 5
        for b in range(self.batch_size):
            input_ids[b, start_pos : start_pos + num_patches_per_image] = image_token_id

        modality_inputs = {"images": {"radio_encoder": {"x": images}}}

        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )

        assert outputs is not None
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_freeze_vit(self):
        """Test that ViT parameters can be frozen."""
        mimo_model = get_nemotron_moe_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )

        # Freeze ViT
        for p in mimo_model.modality_submodules.images.encoders.radio_encoder.parameters():
            p.requires_grad = False

        # Check all RADIO params are frozen
        for p in mimo_model.modality_submodules.images.encoders.radio_encoder.parameters():
            assert not p.requires_grad

        # Check language model params are NOT frozen
        for p in mimo_model.language_model.parameters():
            assert p.requires_grad

    def test_state_dict(self):
        """Test state dict contains expected keys."""
        mimo_model = get_nemotron_moe_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        )

        state_dict = mimo_model.state_dict()
        assert len(state_dict) > 0

        has_lm_keys = any(k.startswith("language_model.") for k in state_dict)
        has_radio_keys = any("radio_encoder" in k for k in state_dict)
        has_projection_keys = any("input_projections" in k for k in state_dict)

        assert has_lm_keys, "Missing language_model keys in state dict"
        assert has_radio_keys, "Missing radio_encoder keys in state dict"
        assert has_projection_keys, "Missing projection keys in state dict"
