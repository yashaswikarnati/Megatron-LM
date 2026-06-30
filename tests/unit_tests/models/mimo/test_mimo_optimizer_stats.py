# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for heterogeneous MIMO optimizer statistics."""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo


def test_num_zeros_uses_world_max_per_module_then_sums():
    encoder_optimizer = SimpleNamespace(count_zeros=mock.Mock(return_value=4))
    optimizer = MimoOptimizer(
        module_infos={
            "encoder": ModuleOptimizerInfo(encoder_optimizer, None, None, True),
            "language": ModuleOptimizerInfo(None, None, None, False),
        },
        config=SimpleNamespace(),
    )
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    def world_max(values, op):
        assert op is torch.distributed.ReduceOp.MAX
        values.copy_(torch.tensor([4, 9], dtype=values.dtype))

    with (
        mock.patch.object(torch, "zeros", side_effect=cpu_zeros),
        mock.patch.object(torch.distributed, "all_reduce", side_effect=world_max) as all_reduce,
    ):
        assert optimizer.count_zeros() == 13
    all_reduce.assert_called_once()
    encoder_optimizer.count_zeros.assert_called_once_with()
