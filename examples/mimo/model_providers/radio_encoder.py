# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""RADIO ViT encoder wrapper for MIMO.

Wraps RADIOViTModel to conform to the MIMO encoder forward(**kwargs) interface,
with optional class-token dropping and pixel-shuffle post-processing.
"""

import torch
import torch.nn as nn

from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


# ---------------------------------------------------------------------------
# Pixel shuffle (adapted from InternVL via megatron/core/models/multimodal/llava_model.py)
# ---------------------------------------------------------------------------
def pixel_shuffle(x, scale_factor=0.5, version=2):
    """Pixel shuffle that reduces spatial tokens and increases hidden dim.

    Args:
        x (torch.Tensor): [num_tiles, img_seq_len, h_vision]
        scale_factor (float): Spatial scaling factor (0.5 → 4x token reduction).
        version (int): Layout variant.

    Returns:
        Tensor of shape [num_tiles, reduced_seq, h_vision * 4]
    """
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(x.shape[0], h, w, -1)

    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(
        n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
    )

    if version == 2:
        x = x.permute(0, 2, 1, 3).contiguous()

    x = x.reshape(x.shape[0], -1, x.shape[-1])
    return x


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
