# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator


def test_attach_modality_split_sizes_includes_zero_image_lanes():
    """Bridge split metadata follows per-sample image token counts."""
    model = object.__new__(MimoModel)
    model.special_token_ids = {"images": 18}

    input_ids = torch.tensor([[18, 1, 2, 3], [4, 5, 6, 7], [18, 18, 8, 9]])
    output = torch.empty((3, 4))

    model._attach_modality_split_sizes(output, input_ids, "images")

    assert output._mimo_bridge_split_sizes == [1, 0, 2]


def test_bridge_split_sizes_allow_text_only_encoder_output():
    """The bridge can split a text-only encoder payload with no modality tokens."""
    bridge = BridgeCommunicator.__new__(BridgeCommunicator)
    bridge.tensor_ndim = 2
    bridge.dim_mapping = {'s': 0, 'h': 1, 'b': 0}

    output = torch.empty((0, 4))
    output._mimo_bridge_split_sizes = [0, 0]

    splits = bridge._split_tensor_at_batch_dim(output, 2)

    assert [tuple(split.shape) for split in splits] == [(0, 4), (0, 4)]
