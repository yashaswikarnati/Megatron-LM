# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import random

import torch

from examples.mimo.data import hetero_energon
from examples.mimo.data.hetero_energon import EnergonIterator


class RandomLoader:
    """Small dataloader stub that consumes Python's global random module."""

    def __iter__(self):
        return self

    def __next__(self):
        return {"value": random.randrange(1_000_000)}


def test_energon_iterator_uses_isolated_python_random_state():
    """Same DP-lane iterators should match without perturbing caller RNG state."""
    first = EnergonIterator(RandomLoader(), random_seed=12345)
    second = EnergonIterator(RandomLoader(), random_seed=12345)

    first_values = []
    second_values = []
    for _ in range(8):
        random.seed(111)
        caller_state = random.getstate()
        first_values.append(next(first)["value"])
        assert random.getstate() == caller_state

        random.seed(222)
        caller_state = random.getstate()
        second_values.append(next(second)["value"])
        assert random.getstate() == caller_state

    assert first_values == second_values
    assert len(set(first_values)) > 1


def test_combine_encoder_batches_drops_packing_and_concatenates_modalities():
    """Encoder fan-out combines whole packed samples without carrying LLM packing metadata."""
    first = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, 4]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "packing_kwargs": {"cu_seqlens_q": torch.tensor([0, 3])},
        "modality_inputs": {"images": {"radio_encoder": {"x": torch.ones(1, 3, 4, 4)}}},
    }
    second = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "labels": torch.tensor([[6, 7, 8]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "packing_kwargs": {"cu_seqlens_q": torch.tensor([0, 3])},
        "modality_inputs": {"images": {"radio_encoder": {"x": torch.zeros(2, 3, 4, 4)}}},
    }

    combined = hetero_energon._combine_encoder_batches([first, second])

    assert "packing_kwargs" not in combined
    assert combined["input_ids"].tolist() == [[1, 2, 3], [5, 6, 7]]
    images = combined["modality_inputs"]["images"]["radio_encoder"]["x"]
    assert images.shape == (3, 3, 4, 4)
    assert torch.all(images[:1] == 1)
    assert torch.all(images[1:] == 0)
