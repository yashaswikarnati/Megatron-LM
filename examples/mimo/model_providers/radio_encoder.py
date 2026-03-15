# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""RADIO ViT encoder wrapper for MIMO.

Wraps RADIOViTModel to conform to the MIMO encoder forward(**kwargs) interface,
with optional class-token dropping and pixel-shuffle post-processing.
"""

import torch
import torch.nn as nn

from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class RADIOEncoderWrapper(nn.Module):
    """RADIO encoder wrapper for MIMO's encoder interface.

    Instantiates a ``RADIOViTModel`` and adds optional class-token dropping
    and pixel-shuffle post-processing so that the output is ready for the
    multimodal projector.

    After pixel shuffle the hidden dimension is ``hidden_size * 4`` and the
    sequence length is reduced by 4×.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        img_h: int = 512,
        img_w: int = 512,
        patch_dim: int = 16,
        class_token_len: int = 8,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = False,
        max_img_h: int = 2048,
        max_img_w: int = 2048,
        has_cpe: bool = True,
        embedder_bias: bool = False,
        force_eval_mode: bool = False,
    ):
        super().__init__()
        self.drop_class_token = drop_class_token
        self.class_token_len = class_token_len
        self._apply_pixel_shuffle = apply_pixel_shuffle

        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            has_cpe=has_cpe,
            embedder_bias=embedder_bias,
            force_eval_mode=force_eval_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run RADIO encoder.

        Args:
            x: Input images of shape ``[num_tiles, 3, img_h, img_w]``.

        Returns:
            Encoded embeddings.  With default settings the shape is
            ``[num_tiles, reduced_seq, hidden_size * 4]``.
        """
        x = x.to(dtype=self.radio_model.embedder.weight.dtype)
        embeddings = self.radio_model(x)  # [num_tiles, seq_len, hidden]

        if self.drop_class_token:
            embeddings = embeddings[:, self.class_token_len :, :]

        if self._apply_pixel_shuffle:
            embeddings = pixel_shuffle(embeddings, scale_factor=0.5)

        return embeddings
