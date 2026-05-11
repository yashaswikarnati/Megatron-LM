# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import random

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
