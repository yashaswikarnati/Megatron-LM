# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.models.mimo.model.base import MimoModel


def test_attach_modality_split_sizes_includes_zero_image_lanes():
    """Bridge split metadata follows per-sample image token counts."""
    model = object.__new__(MimoModel)
    model.special_token_ids = {"images": 18}

    input_ids = torch.tensor([[18, 1, 2, 3], [4, 5, 6, 7], [18, 18, 8, 9]])
    output = torch.empty((3, 4))

    model._attach_modality_split_sizes(output, input_ids, "images")

    assert output._mimo_bridge_split_sizes == [1, 0, 2]
