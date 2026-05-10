# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end MIMO + MoE smoke test (NMFW-464 Phase 1).

Builds a small MimoModel (TransformerBlock encoder + MoE GPT LLM) with process
groups built from HyperCommGrid alt-factorization, wraps in DDP, and drives one
``forward_backward_no_pipelining`` step on 8 GPUs with mock data. No global
``parallel_state`` is initialized for model-parallel groups.

This is the colocated half of the Phase-1 smoke matrix; encoder and LLM share
the same 8 ranks but the LLM also carries an EP/ETP/EDP alt-factorization
overlapping its TP/CP/DP axes (the NMFW-464 expert-overlap fix).
"""

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state, pipeline_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, ModelType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TERowParallelLinear = None


def _build_grids_and_pgs():
    """Build a single HCG that hosts both encoder and LLM (colocated) on 8 ranks.

    The LLM grid carries an expert alt-factorization (etp=1, ep=4, edp=2) over
    the same tp=2 cp=1 dp=4 slab. Encoder uses the primary axes only.
    """
    encoder_grid = HyperCommGrid(
        shape=[1, 1, 8, 1], dim_names=["tp", "cp", "dp", "pp"], backend="nccl"
    )
    llm_grid = HyperCommGrid(
        shape=[1, 1, 8, 1],
        dim_names=["tp", "cp", "dp", "pp"],
        backend="nccl",
        alt_factorizations={
            "expert": {
                # Re-factor the 8-rank slab as ep=4, edp=2 (no expert TP) overlaying dp=8.
                "shape": [1, 4, 2],
                "dim_names": ["etp", "ep", "edp"],
                "replaces": ["tp", "cp", "dp"],
            }
        },
    )

    encoder_pg = ProcessGroupCollection.from_hyper_comm_grid(encoder_grid)
    llm_pg = ProcessGroupCollection.from_hyper_comm_grid(llm_grid)

    # PP=1 singleton groups stand in for embd / pos_embd.
    encoder_pg.embd = encoder_pg.pp
    encoder_pg.pos_embd = encoder_pg.pp
    llm_pg.embd = llm_pg.pp
    llm_pg.pos_embd = llm_pg.pp

    return encoder_grid, llm_grid, encoder_pg, llm_pg


def _build_mamba_moe_language_spec(pg, num_layers, hidden, num_experts, vocab_size, seq_len):
    """Build a MambaModel/HybridModel language spec with Nemotron-MoE shape (Mamba + MoE)."""
    from megatron.core.activations import squared_relu

    # ``pg.tp`` is None on ranks that aren't members of this grid (non-colocated layout).
    tp_size = pg.tp.size() if pg.tp is not None else 1
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        ffn_hidden_size=hidden * 2,
        num_moe_experts=num_experts,
        moe_router_topk=min(num_experts, 4),
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=hidden,
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        variable_seq_lengths=True,
    )
    config.activation_func = squared_relu
    config.gated_linear_unit = False
    config.normalization = "RMSNorm"
    config.position_embedding_type = "none"
    # Mamba SSM dims (smallest values that pass Mamba's divisibility checks for hidden=128).
    config.mamba_state_dim = 64
    config.mamba_head_dim = 64
    config.mamba_num_heads = 2
    config.mamba_num_groups = 1
    return ModuleSpec(
        module=HybridModel,
        params={
            "config": config,
            "hybrid_stack_spec": hybrid_stack_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            # 4-layer Nemotron-style mini pattern: Mamba, attention, Mamba, Expert.
            "hybrid_layer_pattern": "M*ME"[:num_layers],
            "pre_process": True,
            "post_process": True,
            "pg_collection": pg,
        },
    )


def _build_language_spec(
    pg, num_layers, hidden, num_experts, vocab_size, seq_len, nemotron_flavor=False
):
    """Build a GPT-MoE language spec. Tolerates ``pg.tp is None`` for non-colocated ranks.

    When ``nemotron_flavor=True`` the config matches Nemotron6-MoE in every dimension
    that doesn't require the Mamba SSM kernels: squared_relu activation, RMSNorm,
    GQA with num_query_groups=2, sigmoid router with topk=6 (capped to num_experts),
    shared experts, alltoall dispatcher, no bias. Mamba SSM layers themselves are
    out of scope (they need ``mamba_ssm``); the rest of the architecture reachable
    through ``TransformerConfig`` is exercised here.
    """
    from megatron.core.activations import squared_relu

    tp_size = pg.tp.size() if pg.tp is not None else 1
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=4,
        ffn_hidden_size=4 * hidden,
        num_moe_experts=num_experts,
        moe_router_topk=min(num_experts, 6) if nemotron_flavor else 2,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=2 * hidden,
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        variable_seq_lengths=True,
    )
    if nemotron_flavor:
        # Nemotron6-MoE knobs that are reachable through TransformerConfig (everything
        # below comes from configs/nemotron_moe_vlm.py), minus the Mamba-specific fields.
        config.activation_func = squared_relu
        config.gated_linear_unit = False
        config.normalization = "RMSNorm"
        config.num_query_groups = 2
        config.kv_channels = 128
        config.moe_router_score_function = "sigmoid"
        config.moe_router_topk_scaling_factor = 2.5
        config.moe_router_enable_expert_bias = True
        config.moe_router_dtype = "fp32"
        config.moe_router_load_balancing_type = "seq_aux_loss"
        config.moe_aux_loss_coeff = 0.0001
        config.moe_shared_expert_intermediate_size = 2 * hidden
        # Note: shared_expert_overlap reaches for an internal MP group not set in our
        # HCG-only setup; turn off for the smoke test.
        config.moe_shared_expert_overlap = False
        config.position_embedding_type = "none"
        config.attention_dropout = 0.0
        config.hidden_dropout = 0.0
        config.bias_activation_fusion = False
        config.masked_softmax_fusion = True
        config.persist_layer_norm = True
        config.bias_dropout_fusion = False
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_experts, moe_grouped_gemm=False
            ),
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
            "pg_collection": pg,
        },
    )


def _build_radio_submodules_spec(
    pg, num_layers, hidden, language_hidden, img_h, img_w, patch_dim, class_token_len
):
    """Build a vision-modality submodules spec using the literal RADIOEncoderWrapper."""
    from examples.mimo.model_providers.radio_encoder import RADIOEncoderWrapper

    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        pytest.skip("TE column/row parallel linear not available")
    tp_size = pg.tp.size() if pg.tp is not None else 1
    radio_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=4,
        ffn_hidden_size=4 * hidden,
        kv_channels=hidden // 4,
        num_query_groups=4,
        gated_linear_unit=False,
        add_bias_linear=True,
        add_qkv_bias=True,
        normalization="LayerNorm",
        layernorm_epsilon=1e-6,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        moe_token_dispatcher_type="alltoall",
        variable_seq_lengths=True,
    )
    radio_layer_spec = get_vit_layer_with_transformer_engine_spec()
    encoder_spec = ModuleSpec(
        module=RADIOEncoderWrapper,
        params={
            "transformer_config": radio_config,
            "transformer_layer_spec": radio_layer_spec,
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
            "class_token_len": class_token_len,
            "drop_class_token": True,
            "apply_pixel_shuffle": False,
            "max_img_h": img_h,
            "max_img_w": img_w,
            "has_cpe": True,
            "embedder_bias": False,
        },
    )
    proj_config = TransformerConfig(
        num_layers=1,
        hidden_size=language_hidden,
        num_attention_heads=tp_size,
        ffn_hidden_size=language_hidden,
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        bf16=False,
        params_dtype=torch.float32,
    )
    proj_config.activation_func = torch.nn.functional.gelu
    proj_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": proj_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            "projector_type": "mlp",
            "input_size": hidden,  # RADIO hidden -> projection -> language hidden
            "tp_group": pg.tp,
        },
    )
    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg},
        submodules={"encoders": {"radio_encoder": encoder_spec}, "input_projections": [proj_spec]},
    )


def _build_vision_submodules_spec(pg, num_layers, hidden, language_hidden):
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        pytest.skip("TE column/row parallel linear not available")
    tp_size = pg.tp.size() if pg.tp is not None else 1
    vision_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=4,
        ffn_hidden_size=4 * hidden,
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=False,
        params_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        moe_token_dispatcher_type="alltoall",
        variable_seq_lengths=True,
    )
    encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": get_gpt_layer_with_transformer_engine_spec(),
            "pg_collection": pg,
            "pre_process": True,
            "post_process": True,
        },
    )
    proj_config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=tp_size,
        ffn_hidden_size=hidden,
        add_bias_linear=False,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        bf16=False,
        params_dtype=torch.float32,
    )
    proj_config.activation_func = torch.nn.functional.gelu
    proj_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": proj_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            "projector_type": "mlp",
            "input_size": hidden,
            "tp_group": pg.tp,
        },
    )
    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg},
        submodules={"encoders": {"clip_encoder": encoder_spec}, "input_projections": [proj_spec]},
    )


def _mock_radio_data_iterator(
    seq_length, micro_batch_size, vocab_size, num_image_tokens, img_h, img_w
):
    """Mock data iterator that emits raw images (for RADIO) + token IDs with placeholders."""
    while True:
        input_ids = torch.randint(1, vocab_size, (micro_batch_size, seq_length), device="cuda")
        # Place exactly num_image_tokens placeholder tokens (id 0) per batch row.
        input_ids[:, :num_image_tokens] = 0
        loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
        position_ids = (
            torch.arange(seq_length, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)
        )
        # RADIO takes [num_tiles, 3, H, W]. With one tile per batch row, num_tiles == B.
        images = torch.randn(micro_batch_size, 3, img_h, img_w, device="cuda")
        yield {
            "tokens": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": None,
            "modality_inputs": {"radio_encoder": {"radio_encoder": {"x": images}}},
        }


def _mock_mimo_data_iterator(seq_length, micro_batch_size, vocab_size, image_seq_len, hidden):
    while True:
        # Reserve token id 0 as the image-placeholder; sample remaining tokens from [1, V).
        input_ids = torch.randint(1, vocab_size, (micro_batch_size, seq_length), device="cuda")
        # Place exactly image_seq_len placeholder tokens (id 0) per batch row.
        input_ids[:, :image_seq_len] = 0
        loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
        position_ids = (
            torch.arange(seq_length, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)
        )
        # Vision input: image features [seq=image_seq_len, batch, hidden].
        image_features = torch.randn(image_seq_len, micro_batch_size, hidden, device="cuda")
        yield {
            "tokens": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": None,
            "modality_inputs": {
                "clip_encoder": {
                    "clip_encoder": {"hidden_states": image_features, "attention_mask": None}
                }
            },
        }


def _loss_func(loss_mask, output_tensor):
    if isinstance(output_tensor, (tuple, list)):
        output_tensor = output_tensor[0]
    loss = output_tensor.float().sum()
    return loss, {"lm_loss": loss.detach()}


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    output_tensor = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        modality_inputs=batch["modality_inputs"],
    )
    return output_tensor, partial(_loss_func, batch["loss_mask"])


class TestMimoMoEColocated:
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl")
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    @staticmethod
    def _reset_parallel_state(tp=1, pp=1, ep=1, etp=1):
        """Destroy any existing parallel_state and re-initialize with the given topology.

        Required because ``HybridModel`` still calls into ``parallel_state``
        (e.g. ``log_on_each_pipeline_stage``); without an unconditional
        destroy-then-init, tests that need different topologies become order-dependent.
        Also resets the global memory buffer and the CUDA RNG tracker so prior tests'
        state doesn't bleed into RNG-sensitive paths in this one.
        """
        from megatron.core.tensor_parallel import random as tp_random

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state._GLOBAL_MEMORY_BUFFER = None
        # ``_CUDA_RNG_STATE_TRACKER`` is a private singleton; ``.reset()`` is its public
        # surface for clearing seed names. ``initialize_rng_tracker`` recreates the
        # tracker on next ``model_parallel_cuda_manual_seed`` call.
        try:
            tp_random.get_cuda_rng_tracker().reset()
        except Exception:
            pass
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
        )

    def _run_mimo_step(
        self,
        language_spec,
        hidden,
        seq_length,
        image_seq_len,
        micro_batch_size,
        vocab_size,
        encoder_pg,
        llm_pg,
    ):
        """Build MimoModel from the given language spec and drive one fwd/bwd step."""
        vision_spec = _build_vision_submodules_spec(
            pg=encoder_pg, num_layers=1, hidden=hidden, language_hidden=hidden
        )
        mimo_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec={"clip_encoder": vision_spec},
            special_token_ids={"clip_encoder": 0},
        )
        # Pass tp_group from llm_pg so PartitionConfig doesn't fall back to parallel_state.
        mimo_model = MimoModel(mimo_config, tp_group=llm_pg.tp, cp_group=llm_pg.cp).cuda()
        mimo_model.model_type = ModelType.encoder_or_decoder

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
            bucket_size=None,
            average_in_collective=False,
        )
        ddp_model = DistributedDataParallel(
            config=language_spec.params["config"],
            ddp_config=ddp_config,
            module=mimo_model,
            pg_collection=llm_pg,
        )

        data_iter = _mock_mimo_data_iterator(
            seq_length, micro_batch_size, vocab_size, image_seq_len, hidden
        )
        losses = pipeline_parallel.schedules.forward_backward_no_pipelining(
            forward_step_func=_forward_step,
            data_iterator=data_iter,
            model=ddp_model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            pg_collection=llm_pg,
        )
        assert isinstance(losses, list) and len(losses) == 1
        loss_dict = losses[0]
        assert "lm_loss" in loss_dict
        assert torch.isfinite(
            loss_dict["lm_loss"]
        ).item(), f"loss not finite: {loss_dict['lm_loss']}"

        any_grad = False
        for p in mimo_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
                break
            if hasattr(p, "main_grad") and p.main_grad is not None and p.main_grad.abs().sum() > 0:
                any_grad = True
                break
        assert any_grad, "no parameter received a non-zero gradient"
        return mimo_model

    def _setup_pgs_and_rng(self):
        torch.manual_seed(12345)
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        encoder_grid, llm_grid, encoder_pg, llm_pg = _build_grids_and_pgs()
        assert sorted(dist.get_process_group_ranks(llm_pg.ep)) != sorted(
            dist.get_process_group_ranks(llm_pg.dp)
        ), "ep / dp must not alias under alt factorization with these shapes"
        tp_rank = dist.get_rank(group=llm_pg.tp)
        ep_rank = dist.get_rank(group=llm_pg.ep)
        etp_rank = dist.get_rank(group=llm_pg.expt_tp)
        model_parallel_cuda_manual_seed(
            seed=12345, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank
        )
        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()
        return encoder_grid, llm_grid, encoder_pg, llm_pg

    @pytest.mark.parametrize("nemotron_flavor", [False, True])
    def test_mimo_moe_colocated_8gpu(self, nemotron_flavor):
        """Smoke: GPT-style MoE LLM, basic + Nemotron-flavor configs."""
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")
        # GPT path doesn't strictly need ``parallel_state``, but resetting it
        # explicitly makes the test order-independent — matches the other tests
        # in the class.
        self._reset_parallel_state(tp=1, pp=1, ep=4, etp=1)
        encoder_grid, llm_grid, encoder_pg, llm_pg = self._setup_pgs_and_rng()
        hidden = 64
        language_spec = _build_language_spec(
            pg=llm_pg,
            num_layers=2,
            hidden=hidden,
            num_experts=4,
            vocab_size=128,
            seq_len=32,
            nemotron_flavor=nemotron_flavor,
        )
        self._run_mimo_step(
            language_spec,
            hidden=hidden,
            seq_length=32,
            image_seq_len=8,
            micro_batch_size=2,
            vocab_size=128,
            encoder_pg=encoder_pg,
            llm_pg=llm_pg,
        )
        encoder_grid.destroy()
        llm_grid.destroy()

    def test_mimo_nemotron_radio_mamba_moe_colocated_8gpu(self):
        """E2E: RADIO ViT encoder + MLP projection + literal Mamba-MoE LLM.

        This is the literal Nemotron VLM assembly (small variant): RADIOEncoderWrapper
        as the vision encoder, MultimodalProjector for vision→language, and
        ``MambaModel`` (HybridModel with Mamba + MoE) as the language model. Wired
        through HyperCommGrid alt-factorization, ``forward_backward_no_pipelining``,
        and DDP. Mock images + token IDs with placeholders. Colocated mode.
        """
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")
        self._reset_parallel_state(tp=1, pp=1, ep=4, etp=1)
        encoder_grid, llm_grid, encoder_pg, llm_pg = self._setup_pgs_and_rng()

        hidden = 128
        num_experts = 4
        seq_length = 64
        # 64x64 image / patch=16 -> 16 patches; +8 class tokens; drop class -> 16 image tokens.
        img_h = img_w = 64
        patch_dim = 16
        class_token_len = 8
        num_image_tokens_per_image = (img_h // patch_dim) * (img_w // patch_dim)
        micro_batch_size = 2
        # Total image tokens placed in input_ids: one tile per batch row
        # contributes num_image_tokens_per_image tokens; with B rows, the per-row
        # placeholder count must equal num_image_tokens_per_image.
        vocab_size = 128

        language_spec = _build_mamba_moe_language_spec(
            pg=llm_pg,
            num_layers=4,
            hidden=hidden,
            num_experts=num_experts,
            vocab_size=vocab_size,
            seq_len=seq_length,
        )
        vision_spec = _build_radio_submodules_spec(
            pg=encoder_pg,
            num_layers=2,
            hidden=hidden,
            language_hidden=hidden,
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            class_token_len=class_token_len,
        )

        mimo_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec={"radio_encoder": vision_spec},
            special_token_ids={"radio_encoder": 0},
        )
        mimo_model = MimoModel(mimo_config, tp_group=llm_pg.tp, cp_group=llm_pg.cp).cuda()
        mimo_model.model_type = ModelType.encoder_or_decoder

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
            bucket_size=None,
            average_in_collective=False,
        )
        ddp_model = DistributedDataParallel(
            config=language_spec.params["config"],
            ddp_config=ddp_config,
            module=mimo_model,
            pg_collection=llm_pg,
        )

        data_iter = _mock_radio_data_iterator(
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            vocab_size=vocab_size,
            num_image_tokens=num_image_tokens_per_image,
            img_h=img_h,
            img_w=img_w,
        )

        losses = pipeline_parallel.schedules.forward_backward_no_pipelining(
            forward_step_func=_forward_step,
            data_iterator=data_iter,
            model=ddp_model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            pg_collection=llm_pg,
        )
        assert isinstance(losses, list) and len(losses) == 1
        assert torch.isfinite(losses[0]["lm_loss"]).item()
        any_grad = any(
            (p.grad is not None and p.grad.abs().sum() > 0)
            or (hasattr(p, "main_grad") and p.main_grad is not None and p.main_grad.abs().sum() > 0)
            for p in mimo_model.parameters()
        )
        assert any_grad, "no parameter received a non-zero gradient"
        encoder_grid.destroy()
        llm_grid.destroy()

    def test_mimo_nemotron_radio_mamba_moe_non_colocated_8gpu(self):
        """E2E non-colocated literal Nemotron VLM: RADIO encoder (ranks 0-3) + Mamba-MoE
        LLM (ranks 4-7) bridged by ``MultiModulePipelineCommunicator``."""
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")
        self._reset_parallel_state(tp=1, pp=1, ep=2, etp=1)
        torch.manual_seed(12345)
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

        encoder_grid = HyperCommGrid(
            shape=[1, 1, 4, 1], dim_names=["tp", "cp", "dp", "pp"], rank_offset=0, backend="nccl"
        )
        llm_grid = HyperCommGrid(
            shape=[1, 1, 4, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            rank_offset=4,
            backend="nccl",
            alt_factorizations={
                "expert": {
                    "shape": [1, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        encoder_pg = ProcessGroupCollection.from_hyper_comm_grid(encoder_grid)
        llm_pg = ProcessGroupCollection.from_hyper_comm_grid(llm_grid)

        rank = dist.get_rank()
        in_encoder = 0 <= rank < 4
        in_llm = 4 <= rank < 8
        if in_encoder:
            encoder_pg.embd = encoder_pg.pp
            encoder_pg.pos_embd = encoder_pg.pp
        if in_llm:
            llm_pg.embd = llm_pg.pp
            llm_pg.pos_embd = llm_pg.pp

        # This test assumes symmetric DP (no MBS scaling across the bridge).
        if in_encoder:
            assert encoder_pg.dp.size() == 4
        if in_llm:
            assert llm_pg.dp.size() == 4

        ep_rank = dist.get_rank(group=llm_pg.ep) if in_llm else 0
        model_parallel_cuda_manual_seed(seed=12345, tp_rank=0, ep_rank=ep_rank, etp_rank=0)
        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        hidden = 128
        num_experts = 4
        seq_length = 64
        img_h = img_w = 64
        patch_dim = 16
        class_token_len = 8
        num_image_tokens_per_image = (img_h // patch_dim) * (img_w // patch_dim)
        micro_batch_size = 2
        vocab_size = 128
        encoder_name = "radio_encoder"

        language_spec = _build_mamba_moe_language_spec(
            pg=llm_pg,
            num_layers=4,
            hidden=hidden,
            num_experts=num_experts,
            vocab_size=vocab_size,
            seq_len=seq_length,
        )
        vision_spec = _build_radio_submodules_spec(
            pg=encoder_pg,
            num_layers=2,
            hidden=hidden,
            language_hidden=hidden,
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            class_token_len=class_token_len,
        )

        module_to_grid_map = {encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid}
        topology = {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}
        mimo_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec={encoder_name: vision_spec},
            special_token_ids={encoder_name: 0},
            module_to_grid_map=module_to_grid_map,
        )
        tp_group = llm_pg.tp if in_llm else encoder_pg.tp
        cp_group = llm_pg.cp if in_llm else encoder_pg.cp
        mimo_model = MimoModel(mimo_config, tp_group=tp_group, cp_group=cp_group).cuda()
        mimo_model.model_type = ModelType.encoder_or_decoder

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
            bucket_size=None,
            average_in_collective=False,
        )
        if in_llm and mimo_model.language_model is not None:
            mimo_model.language_model = DistributedDataParallel(
                config=mimo_model.language_model.config,
                ddp_config=ddp_config,
                module=mimo_model.language_model,
                pg_collection=llm_pg,
            )
        if in_encoder and encoder_name in mimo_model.modality_submodules:
            sub = mimo_model.modality_submodules[encoder_name]
            if sub is not None:
                # The encoder-spec submodule key is "radio_encoder" here.
                ddp_cfg_src = sub.encoders["radio_encoder"].radio_model.config
                mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                    config=ddp_cfg_src, ddp_config=ddp_config, module=sub, pg_collection=encoder_pg
                )

        communicator = MultiModulePipelineCommunicator(
            module_to_grid_map,
            topology,
            mimo_model.config,
            dim_mapping={"s": 0, "h": 2, "b": 1},
            module_output_ndim={encoder_name: 2},
        )

        def _data_iter():
            while True:
                images = torch.randn(micro_batch_size, 3, img_h, img_w, device="cuda")
                input_ids = torch.randint(
                    1, vocab_size, (micro_batch_size, seq_length), device="cuda"
                )
                input_ids[:, :num_image_tokens_per_image] = 0
                position_ids = (
                    torch.arange(seq_length, device="cuda")
                    .unsqueeze(0)
                    .expand(micro_batch_size, -1)
                )
                loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
                yield {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": None,
                    "loss_mask": loss_mask,
                    "modality_inputs": {encoder_name: {"radio_encoder": {"x": images}}},
                }

        module_pgs = {}
        language_model_module_name = None
        if in_encoder:
            module_pgs[encoder_name] = encoder_pg
        if in_llm:
            module_pgs[MIMO_LANGUAGE_MODULE_KEY] = llm_pg
            language_model_module_name = MIMO_LANGUAGE_MODULE_KEY
        sched_pg = MultiModuleProcessGroupCollection(
            module_pgs=module_pgs, language_model_module_name=language_model_module_name
        )

        def step_func(data_iterator, model):
            def loss_func(loss_mask, output_tensor):
                if output_tensor is None:
                    return torch.tensor(0.0, device="cuda", requires_grad=True), {
                        "loss_reduced": 0.0
                    }
                if isinstance(output_tensor, dict):
                    out = output_tensor.get(
                        MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
                    )
                else:
                    out = output_tensor
                if out is None:
                    return torch.tensor(0.0, device="cuda", requires_grad=True), {
                        "loss_reduced": 0.0
                    }
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = out.float().sum()
                return loss, {"loss_reduced": loss.detach()}

            batch = next(data_iterator) if data_iterator is not None else {}
            output_tensor = model(**batch)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            return output_tensor, partial(loss_func, batch.get("loss_mask"))

        losses = pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=step_func,
            data_iterator=_data_iter(),
            model=[mimo_model],
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            p2p_communicator=communicator,
            pg_collection=sched_pg,
        )
        if in_llm:
            assert isinstance(losses, list)
            for ld in losses:
                assert "loss_reduced" in ld
        encoder_grid.destroy()
        llm_grid.destroy()

    def test_mimo_mamba_moe_non_colocated_8gpu(self):
        """E2E non-colocated: encoder on ranks 0-3, Mamba-MoE LLM on ranks 4-7.

        Bridge between the two rank slabs is ``MultiModulePipelineCommunicator``;
        the schedule is ``forward_backward_pipelining_without_interleaving``.
        Encoder DP=4, LLM DP=4 (symmetric, no MBS scaling needed). LLM grid carries
        an alt-factorization ep=2 edp=2 over its dp=4 axis.
        """
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")
        # parallel_state needs to match the LLM topology because HybridModel still
        # reaches into log_on_each_pipeline_stage / get_tensor_model_parallel_*. We
        # initialize it once with TP=1 EP=2 PP=1 (matching the LLM slab on ranks 4-7).
        self._reset_parallel_state(tp=1, pp=1, ep=2, etp=1)

        torch.manual_seed(12345)
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

        # Disjoint grids: encoder on ranks [0..3], LLM on ranks [4..7]. Each tp=1 dp=4 pp=1.
        encoder_grid = HyperCommGrid(
            shape=[1, 1, 4, 1], dim_names=["tp", "cp", "dp", "pp"], rank_offset=0, backend="nccl"
        )
        llm_grid = HyperCommGrid(
            shape=[1, 1, 4, 1],
            dim_names=["tp", "cp", "dp", "pp"],
            rank_offset=4,
            backend="nccl",
            alt_factorizations={
                "expert": {
                    "shape": [1, 2, 2],
                    "dim_names": ["etp", "ep", "edp"],
                    "replaces": ["tp", "cp", "dp"],
                }
            },
        )
        encoder_pg = ProcessGroupCollection.from_hyper_comm_grid(encoder_grid)
        llm_pg = ProcessGroupCollection.from_hyper_comm_grid(llm_grid)

        rank = dist.get_rank()
        in_encoder = 0 <= rank < 4
        in_llm = 4 <= rank < 8

        # PP=1 stand-ins for embd / pos_embd, only on ranks that are in the relevant grid
        # (``from_hyper_comm_grid`` only populates fields for ranks that are members).
        if in_encoder:
            encoder_pg.embd = encoder_pg.pp
            encoder_pg.pos_embd = encoder_pg.pp
        if in_llm:
            llm_pg.embd = llm_pg.pp
            llm_pg.pos_embd = llm_pg.pp

        # This test assumes symmetric DP (no MBS scaling across the bridge).
        if in_encoder:
            assert encoder_pg.dp.size() == 4
        if in_llm:
            assert llm_pg.dp.size() == 4

        # RNG init on whichever grid this rank belongs to (TP rank for both is 0
        # since tp=1; ep_rank only meaningful on LLM ranks).
        ep_rank = dist.get_rank(group=llm_pg.ep) if in_llm else 0
        model_parallel_cuda_manual_seed(seed=12345, tp_rank=0, ep_rank=ep_rank, etp_rank=0)
        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        hidden = 128
        num_experts = 4
        seq_length = 32
        image_seq_len = 8
        micro_batch_size = 2
        vocab_size = 128

        encoder_name = "clip_encoder"
        # Build language + vision specs with their own pg_collections.
        language_spec = _build_mamba_moe_language_spec(
            pg=llm_pg,
            num_layers=4,
            hidden=hidden,
            num_experts=num_experts,
            vocab_size=vocab_size,
            seq_len=seq_length,
        )
        # Use the simpler TransformerBlock encoder for this smoke test (RADIO would
        # add image-tile shape constraints across the bridge — orthogonal to what
        # this test verifies).
        vision_spec = _build_vision_submodules_spec(
            pg=encoder_pg, num_layers=1, hidden=hidden, language_hidden=hidden
        )

        module_to_grid_map = {encoder_name: encoder_grid, MIMO_LANGUAGE_MODULE_KEY: llm_grid}
        topology = {encoder_name: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}

        mimo_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec={encoder_name: vision_spec},
            special_token_ids={encoder_name: 0},
            module_to_grid_map=module_to_grid_map,
        )
        # tp_group / cp_group come from whichever side this rank lives on.
        tp_group = llm_pg.tp if in_llm else encoder_pg.tp
        cp_group = llm_pg.cp if in_llm else encoder_pg.cp
        mimo_model = MimoModel(mimo_config, tp_group=tp_group, cp_group=cp_group).cuda()
        mimo_model.model_type = ModelType.encoder_or_decoder

        # DDP-wrap each side independently with its own pg_collection.
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,
            bucket_size=None,
            average_in_collective=False,
        )
        if in_llm and mimo_model.language_model is not None:
            mimo_model.language_model = DistributedDataParallel(
                config=mimo_model.language_model.config,
                ddp_config=ddp_config,
                module=mimo_model.language_model,
                pg_collection=llm_pg,
            )
        if in_encoder and encoder_name in mimo_model.modality_submodules:
            sub = mimo_model.modality_submodules[encoder_name]
            if sub is not None:
                mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                    config=sub.encoders["clip_encoder"].config,
                    ddp_config=ddp_config,
                    module=sub,
                    pg_collection=encoder_pg,
                )

        # Multi-module bridge: encoder hidden flows from encoder ranks to LLM ranks.
        communicator = MultiModulePipelineCommunicator(
            module_to_grid_map,
            topology,
            mimo_model.config,
            dim_mapping={"s": 0, "h": 2, "b": 1},
            module_output_ndim={encoder_name: 2},
        )

        # Per-rank data iterator: only encoder ranks (which feed images) and LLM ranks
        # (which read text + run loss) need data. With encoder_dp == llm_dp the MBS is
        # the same on both sides.
        def _data_iter():
            while True:
                # Mock encoder hidden states (the "image features" the bridge will ship).
                encoder_hidden_states = torch.randn(
                    image_seq_len, micro_batch_size, hidden, device="cuda"
                )
                input_ids = torch.randint(
                    1, vocab_size, (micro_batch_size, seq_length), device="cuda"
                )
                input_ids[:, :image_seq_len] = 0
                position_ids = (
                    torch.arange(seq_length, device="cuda")
                    .unsqueeze(0)
                    .expand(micro_batch_size, -1)
                )
                loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
                yield {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": None,
                    "loss_mask": loss_mask,
                    "modality_inputs": {
                        encoder_name: {
                            "clip_encoder": {
                                "hidden_states": encoder_hidden_states,
                                "attention_mask": None,
                            }
                        }
                    },
                }

        data_iterator = _data_iter()

        # Schedule's MultiModuleProcessGroupCollection: only this rank's pg.
        module_pgs = {}
        language_model_module_name = None
        if in_encoder:
            module_pgs[encoder_name] = encoder_pg
        if in_llm:
            module_pgs[MIMO_LANGUAGE_MODULE_KEY] = llm_pg
            language_model_module_name = MIMO_LANGUAGE_MODULE_KEY
        sched_pg = MultiModuleProcessGroupCollection(
            module_pgs=module_pgs, language_model_module_name=language_model_module_name
        )

        def step_func(data_iterator, model):
            def loss_func(loss_mask, output_tensor):
                if output_tensor is None:
                    return torch.tensor(0.0, device="cuda", requires_grad=True), {
                        "loss_reduced": 0.0
                    }
                if isinstance(output_tensor, dict):
                    out = output_tensor.get(
                        MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()), None)
                    )
                else:
                    out = output_tensor
                if out is None:
                    return torch.tensor(0.0, device="cuda", requires_grad=True), {
                        "loss_reduced": 0.0
                    }
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = out.float().sum()
                return loss, {"loss_reduced": loss.detach()}

            batch = next(data_iterator) if data_iterator is not None else {}
            output_tensor = model(**batch)
            # MoE LLMs return ``(logits, extras)``; the schedule expects a single tensor.
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            return output_tensor, partial(loss_func, batch.get("loss_mask"))

        losses = pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=step_func,
            data_iterator=data_iterator,
            model=[mimo_model],
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            p2p_communicator=communicator,
            pg_collection=sched_pg,
        )

        # On LLM-last-stage ranks the schedule returns loss dicts; elsewhere [].
        if in_llm:
            assert isinstance(losses, list)
            for ld in losses:
                assert "loss_reduced" in ld
                # loss_reduced may be 0.0 sentinel on intermediate stages;
                # at least one rank in the LLM dp group should have a real tensor.
        encoder_grid.destroy()
        llm_grid.destroy()

    def test_mimo_nemotron_mamba_moe_colocated_8gpu(self):
        """Smoke: literal Nemotron-shape Mamba-MoE LLM via HybridModel.

        HybridModel still has a few hard ``parallel_state`` touchpoints
        (notably ``log_on_each_pipeline_stage``); for this smoke test we
        minimally initialize parallel_state with shapes matching the
        HyperCommGrid so those calls succeed. Real training would route
        these through pg_collection — that's a follow-up cleanup beyond
        the NMFW-464 Phase-1 substrate.
        """
        if not dist.is_initialized() or dist.get_world_size() != 8:
            pytest.skip("Requires exactly 8 GPUs")
        # Match LLM grid topology: tp=1 cp=1 dp=8 pp=1 with ep=4 etp=1 edp=2.
        self._reset_parallel_state(tp=1, pp=1, ep=4, etp=1)
        encoder_grid, llm_grid, encoder_pg, llm_pg = self._setup_pgs_and_rng()
        hidden = 128
        language_spec = _build_mamba_moe_language_spec(
            pg=llm_pg, num_layers=4, hidden=hidden, num_experts=4, vocab_size=128, seq_len=32
        )
        self._run_mimo_step(
            language_spec,
            hidden=hidden,
            seq_length=32,
            image_seq_len=8,
            micro_batch_size=2,
            vocab_size=128,
            encoder_pg=encoder_pg,
            llm_pg=llm_pg,
        )
        encoder_grid.destroy()
        llm_grid.destroy()
