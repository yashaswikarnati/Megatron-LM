# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import gc

import pytest
import torch

from megatron.core import config
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    permute_fusion_params,
    token_permutation,
    token_unpermutation,
)
from megatron.core.typed_torch import apply_module


def test_placeholder():
    """This is here because otherwise there's no other test in this module (all disabled)
    and pytest would fail."""
    pass


class TestAlltoAllDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_forward_backward(self, tp_size, ep_size, permute_fusion, deterministic, monkeypatch):
        if deterministic:
            # We only need to exercise the deterministic branches in moe_utils.
            # Enabling global determinism (torch.use_deterministic_algorithms(True))
            # would require CUBLAS_WORKSPACE_CONFIG and can slow other tests.
            # Monkeypatching here is per-test scoped and avoids global side effects.
            monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
            # Deterministic branch is exercised on the unfused path
            if permute_fusion:
                pytest.skip("Deterministic path tested only for unfused (permute_fusion=False)")
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_capacity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_padding_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.6,
            moe_pad_expert_input_to_capacity=True,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_drop_and_pad_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not is_te_min_version("2.1.0"), reason="TE 2.1.0 is required for permute fusion."
    )
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1)])
    def test_memory_efficient_permute(self, tp_size, ep_size):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            recompute_granularity="selective",
            recompute_modules=["moe_permute"],
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not is_te_min_version("2.1.0"), reason="TE 2.1.0 is required for permute fusion."
    )
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    def test_memory_efficient_permute_saves_memory(self, tp_size, ep_size):
        """Verify that memory_efficient_permute reduces peak GPU memory."""

        def _run_forward_backward(container):
            moe_layer = container.moe_layer
            bs, seql = 32, 8
            hidden_states = torch.randn(
                (bs, seql, moe_layer.config.hidden_size), dtype=torch.float32
            ).cuda()
            hidden_states.requires_grad = True
            probs, indices = apply_module(moe_layer.router)(hidden_states)
            probs = torch.ones_like(probs) / moe_layer.router.topk

            permuted_states, tokens_per_expert, permuted_probs = token_permutation(
                moe_layer.token_dispatcher, hidden_states, probs, indices
            )
            permuted_states = permuted_states * permuted_probs.unsqueeze(-1)
            restored, _ = token_unpermutation(moe_layer.token_dispatcher, permuted_states)
            torch.autograd.backward(restored, torch.ones_like(restored))

        # --- Baseline: fused permute (no memory_efficient_permute) ---
        baseline_container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
        )
        # Warm-up run to stabilize allocator
        _run_forward_backward(baseline_container)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run_forward_backward(baseline_container)
        torch.cuda.synchronize()
        baseline_peak = torch.cuda.max_memory_allocated()
        del baseline_container
        Utils.destroy_model_parallel()

        # --- Optimized: memory_efficient_permute ---
        opt_container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            recompute_granularity="selective",
            recompute_modules=["moe_permute"],
        )
        # Warm-up run to stabilize allocator
        _run_forward_backward(opt_container)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run_forward_backward(opt_container)
        torch.cuda.synchronize()
        opt_peak = torch.cuda.max_memory_allocated()
        del opt_container
        Utils.destroy_model_parallel()

        saved_bytes = baseline_peak - opt_peak
        assert saved_bytes > 0, (
            f"Expected memory reduction with memory_efficient_permute, but "
            f"baseline_peak={baseline_peak / (1024**2):.2f}MiB, "
            f"opt_peak={opt_peak / (1024**2):.2f}MiB, "
            f"saved={saved_bytes / (1024**2):.2f}MiB"
        )

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("experimental_fusion", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self, tp_size, ep_size, permute_fusion, experimental_fusion
    ):
        if experimental_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=4,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False
