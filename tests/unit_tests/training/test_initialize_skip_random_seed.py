# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for seed-free Megatron initialization."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import initialize as initialize_mod


@pytest.mark.parametrize("skip_random_seed, expected_seed_calls", [(False, 1), (True, 0)])
def test_skip_random_seed_only_skips_seed_setup(skip_random_seed, expected_seed_calls):
    args = SimpleNamespace(
        async_save=False,
        use_persistent_ckpt_worker=False,
        rerun_mode="disabled",
        error_injection_rate=0,
        error_injection_type="correct_result",
        result_rejected_tracker_filename=None,
        batch_invariant_mode=False,
        lazy_mpu_init=False,
        seed=123,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        cuda_graph_impl="none",
        num_experts=None,
        tp_comm_overlap=False,
    )
    with (
        mock.patch.object(initialize_mod.torch.cuda, "is_available", return_value=True),
        mock.patch.object(initialize_mod, "get_args", return_value=args),
        mock.patch.object(initialize_mod, "setup_logging"),
        mock.patch.object(initialize_mod, "initialize_rerun_state_machine"),
        mock.patch.object(initialize_mod, "_initialize_distributed") as initialize_distributed,
        mock.patch.object(initialize_mod, "_set_random_seed") as set_seed,
        mock.patch.object(initialize_mod, "_init_autoresume") as init_autoresume,
        mock.patch.object(initialize_mod, "_compile_dependencies") as compile_dependencies,
    ):
        initialize_mod.initialize_megatron(skip_random_seed=skip_random_seed)
    assert set_seed.call_count == expected_seed_calls
    initialize_distributed.assert_called_once()
    init_autoresume.assert_called_once()
    compile_dependencies.assert_called_once()
