# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model providers and configs for MIMO VLM examples."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Optional

import torch

from examples.mimo.utils.hetero import (
    debug_rank,
    get_grid_dim_size,
    get_group_rank_or,
    get_group_size_or,
    is_process_group_member,
)
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TELayerNormColumnParallelLinear = None
    TERowParallelLinear = None

MOCK_MODEL_PROVIDER = "mock"
NEMOTRON_20L_MODEL_PROVIDER = "nemotron-moe-vlm-20l"
NEMOTRON_54L_MODEL_PROVIDER = "nemotron-moe-vlm-54l"
NEMOTRON_20L_IMAGE_SEQ_PER_TILE = 256
NEMOTRON_20L_MAX_NUM_TILES = 12
NEMOTRON_20L_DEFAULT_STAGE = "stage2"
MOCK_VISION_ENCODER_KEY = "clip_encoder"
NEMOTRON_VISION_ENCODER_KEY = "radio_encoder"


def is_nemotron_20l(args: argparse.Namespace) -> bool:
    """Return whether the Nemotron6-MoE VLM 20L provider is active."""
    return args.model_provider == NEMOTRON_20L_MODEL_PROVIDER


def is_nemotron_moe_vlm(args: argparse.Namespace) -> bool:
    """Return whether a Nemotron6-MoE VLM provider is active."""
    return args.model_provider in (NEMOTRON_20L_MODEL_PROVIDER, NEMOTRON_54L_MODEL_PROVIDER)


def add_model_provider_args(parser: argparse.ArgumentParser) -> None:
    """Register model-provider arguments for hetero MIMO examples."""
    provider = parser.add_argument_group("model provider")
    provider.add_argument(
        "--model-provider",
        choices=[MOCK_MODEL_PROVIDER, NEMOTRON_20L_MODEL_PROVIDER, NEMOTRON_54L_MODEL_PROVIDER],
        default=MOCK_MODEL_PROVIDER,
    )
    provider.add_argument("--hidden-size", type=int, default=128)
    provider.add_argument("--num-layers", type=int, default=2)
    provider.add_argument("--num-attention-heads", type=int, default=8)
    provider.add_argument("--vocab-size", type=int, default=512)
    provider.add_argument("--seq-length", type=int, default=32)
    provider.add_argument("--image-seq-length", type=int, default=None)
    provider.add_argument("--image-token-id", type=int, default=511)
    provider.add_argument("--pad-token-id", type=int, default=0)
    provider.add_argument("--image-token", type=str, default="<image>")
    provider.add_argument("--tokenizer-model", type=str, default=None)
    provider.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    provider.add_argument("--image-tag-type", type=str, default="")
    provider.add_argument("--force-system-message", action="store_true")
    provider.add_argument("--num-moe-experts", type=int, default=4)
    provider.add_argument("--moe-router-topk", type=int, default=1)
    provider.add_argument(
        "--moe-router-force-load-balancing",
        action="store_true",
        help="Use random router logits to force MoE load balancing for benchmark/debug runs.",
    )
    provider.add_argument("--moe-grouped-gemm", action="store_true")
    provider.add_argument("--img-h", type=int, default=512)
    provider.add_argument("--img-w", type=int, default=512)
    provider.add_argument("--patch-dim", type=int, default=16)
    provider.add_argument("--class-token-len", type=int, default=8)
    provider.add_argument(
        "--num-image-tiles",
        "--max-num-tiles",
        dest="num_image_tiles",
        type=int,
        default=NEMOTRON_20L_MAX_NUM_TILES,
    )
    provider.add_argument("--vision-model-type", type=str, default="radio")
    provider.add_argument("--pixel-shuffle", action="store_true")
    provider.add_argument("--disable-vision-class-token", action="store_true")
    provider.add_argument("--use-tiling", action="store_true")
    provider.add_argument("--use-thumbnail", action="store_true")
    provider.add_argument(
        "--dynamic-resolution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Patchify each image at its native aspect ratio with a token budget instead of "
            "fixed-tile resize. Enabled by default for Nemotron6-MoE VLM providers. "
            "Pass --no-dynamic-resolution to disable."
        ),
    )
    provider.add_argument(
        "--dynamic-resolution-min-patches",
        type=int,
        default=4,
        help="Lower bound on per-image patch count under dynamic resolution.",
    )
    provider.add_argument(
        "--dynamic-resolution-max-patches",
        type=int,
        default=0,
        help="Upper bound on per-image patch count under dynamic resolution; 0 = uncapped.",
    )
    provider.add_argument("--freeze-lm", action="store_true")
    provider.add_argument("--freeze-vit", action="store_true")
    provider.add_argument("--freeze-projection", action="store_true")
    provider.add_argument("--training-stage", choices=["stage1", "stage2", "stage3"], default=None)
    provider.add_argument("--fp32", action="store_true")


def prepare_model_provider_args(args: argparse.Namespace) -> None:
    """Apply provider defaults and derived tokenizer/vision settings."""
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    resolve_image_token_id(args)
    args.vision_encoder_key = get_encoder_module_name(args)
    args.vision_input_mode = "pixels" if is_nemotron_moe_vlm(args) else "hidden_states"


def apply_model_provider_defaults(args: argparse.Namespace) -> None:
    """Apply Nemotron6-MoE VLM model defaults."""
    if not is_nemotron_moe_vlm(args):
        return

    args.num_layers = 54 if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER else 20
    args.hidden_size = 2688
    args.num_attention_heads = 32
    args.num_moe_experts = 128
    args.moe_router_topk = 6
    args.moe_grouped_gemm = True
    args.hybrid_layer_pattern = (
        "MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME"
        if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER
        else "MEMEM*EMEMEM*EMEMEM*"
    )
    args.seq_length = 8192
    args.image_seq_length = NEMOTRON_20L_IMAGE_SEQ_PER_TILE * args.num_image_tiles
    args.pixel_shuffle = True
    args.disable_vision_class_token = True
    if args.dynamic_resolution is None:
        args.dynamic_resolution = True
    if args.dynamic_resolution:
        # Dynamic-resolution strategy reads `use_thumbnail` inside
        # `DynamicResolutionImageTilingStrategy` and emits an extra thumbnail
        # tile when True. `use_tiling` is inert in this branch (the fixed-tile
        # path is unreachable), but pin it False for args-dump parity.
        args.use_tiling = False
        args.use_thumbnail = False
    else:
        args.use_tiling = True
        args.use_thumbnail = True


def apply_training_stage(args: argparse.Namespace) -> None:
    """Apply stage-specific freeze flags for the Nemotron VLM recipe."""
    if not is_nemotron_moe_vlm(args):
        return

    stage = args.training_stage or NEMOTRON_20L_DEFAULT_STAGE
    if stage == "stage1":
        args.freeze_vit = True
        args.freeze_lm = True
    elif stage == "stage2":
        args.freeze_vit = True
    elif stage != "stage3":
        raise ValueError(f"unsupported Nemotron VLM training stage: {stage}")
    args.training_stage = stage


def resolve_image_token_id(args: argparse.Namespace) -> None:
    """Resolve image, pad, and vocab ids from the configured tokenizer."""
    if not is_nemotron_moe_vlm(args) or not args.tokenizer_model:
        return

    from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
        MegatronMultimodalTokenizer,
    )

    tokenizer = MegatronMultimodalTokenizer(
        path=args.tokenizer_model,
        prompt_format=args.tokenizer_prompt_format,
        special_tokens=[args.image_token],
        image_tag_type=args.image_tag_type,
        force_system_message=args.force_system_message,
    )
    image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    if image_token_id is None:
        raise RuntimeError(
            f"tokenizer at {args.tokenizer_model} did not produce an id for {args.image_token}"
        )
    args.image_token_id = int(image_token_id)
    if tokenizer.pad is not None:
        args.pad_token_id = int(tokenizer.pad)
    if tokenizer.vocab_size is not None:
        args.vocab_size = int(tokenizer.vocab_size)


def validate_model_provider_args(args: argparse.Namespace) -> None:
    """Validate derived model-provider arguments."""
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads")
    if not 0 <= args.image_token_id < args.vocab_size:
        raise ValueError("--image-token-id must be within --vocab-size")
    if not 0 <= args.pad_token_id < args.vocab_size:
        raise ValueError("--pad-token-id must be within --vocab-size")


def _pixel_shuffle_dynamic_res(x, imgs_sizes, patch_dim, scale_factor=0.5, version=2):
    """Pixel shuffle for dynamic resolution (variable tile sizes).

    Splits the packed sequence by per-tile lengths, applies pixel shuffle to each
    tile, then re-concatenates. Mirrors sasatheesh/pre-vlm-05's
    llava_model.pixel_shuffle_dynamic_res; vendored here to avoid touching the
    upstream-owned llava_model.py.
    """
    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
    splits = torch.split(x, seq_lens.tolist(), dim=-2)

    out = []
    for i, sv in enumerate(splits):
        h = imgs_sizes[i][0] // patch_dim
        w = imgs_sizes[i][1] // patch_dim
        sv = sv.reshape(sv.shape[0], h, w, -1)

        n, h, w, c = sv.size()
        sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
        sv = sv.permute(0, 2, 1, 3).contiguous()
        sv = sv.view(
            n,
            int(w * scale_factor),
            int(h * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )

        if version == 2:
            sv = sv.permute(0, 2, 1, 3).contiguous()

        sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
        out.append(sv)

    return torch.cat(out, dim=-2)


class RADIOEncoderWrapper(torch.nn.Module):
    """RADIO encoder wrapper matching the Nemotron6-MoE VLM provider."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection],
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
        dynamic_resolution: bool = False,
    ) -> None:
        super().__init__()
        self.class_token_len = class_token_len
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle
        self.force_eval_mode = force_eval_mode
        self.dynamic_resolution = dynamic_resolution
        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            pg_collection=pg_collection,
        )
        if self.force_eval_mode:
            self.radio_model.eval()

    def train(self, mode: bool = True):
        """Keep frozen RADIO in eval mode while allowing the projection to train."""
        super().train(mode)
        if self.force_eval_mode:
            self.radio_model.eval()
        return self

    @property
    def config(self):
        """Expose the underlying RADIO config for DDP wrapping."""
        return self.radio_model.config

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Run RADIO, drop class tokens, and apply pixel shuffle."""
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        debug_rank(f"RADIO forward start: input_shape={tuple(x.shape)}")
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            embeddings = self.radio_model(
                x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params
            )
        debug_rank(f"RADIO forward done: output_shape={tuple(embeddings.shape)}")
        if self.drop_class_token:
            if (
                self.dynamic_resolution
                and imgs_sizes is not None
                and self.class_token_len > 0
            ):
                # Class tokens are interleaved between tiles; build mask to remove them.
                remove_mask = torch.full(
                    (embeddings.shape[-2],), True, dtype=torch.bool, device=embeddings.device
                )
                patch_dim = self.radio_model.patch_dim
                if torch.is_tensor(imgs_sizes):
                    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
                else:
                    seq_lens = torch.tensor(
                        [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes]
                    )
                current_length = 0
                for sl in seq_lens:
                    remove_mask[current_length : current_length + self.class_token_len] = False
                    current_length += int(sl) + self.class_token_len
                embeddings = embeddings[:, remove_mask, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]
            debug_rank(f"RADIO class tokens dropped: output_shape={tuple(embeddings.shape)}")
        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_res(
                    embeddings, imgs_sizes, self.radio_model.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
            debug_rank(f"RADIO pixel shuffle done: output_shape={tuple(embeddings.shape)}")
        return embeddings

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate checkpoint sharding to the wrapped RADIO model."""
        sharded_sd = {}
        for name, child in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(child, f"{prefix}{name}.", sharded_offsets, metadata)
            )
        return sharded_sd


def get_encoder_module_name(args: argparse.Namespace) -> str:
    """Return the concrete encoder key for the active vision provider."""
    return NEMOTRON_VISION_ENCODER_KEY if is_nemotron_moe_vlm(args) else MOCK_VISION_ENCODER_KEY


def get_vision_encoder_module(args: argparse.Namespace, vision_submodule):
    """Return the provider-owned encoder module used for DDP config and freezing."""
    return vision_submodule.encoders[get_encoder_module_name(args)]


def iter_vision_projection_modules(vision_submodule):
    """Return the provider-owned projection modules used for freeze-stage policy."""
    return iter(vision_submodule.input_projections)


def projection_layer_spec() -> ModuleSpec:
    """Return the TE-backed projection MLP spec."""
    if TEColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TEColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )


def nemotron_projection_layer_spec() -> ModuleSpec:
    """Return the Nemotron VLM RADIO-to-language projector layer spec."""
    if TELayerNormColumnParallelLinear is None or TERowParallelLinear is None:
        raise RuntimeError("TELayerNormColumnParallelLinear and TERowParallelLinear are required")
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )


def nemotron_language_config(
    args: argparse.Namespace, tp_size: int, pp_size: int, ep_size: int, expt_tp_size: int
) -> TransformerConfig:
    """Build the Nemotron6-MoE language TransformerConfig."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=54 if args.model_provider == NEMOTRON_54L_MODEL_PROVIDER else 20,
        hidden_size=2688,
        num_attention_heads=32,
        attention_backend=AttnBackend.flash,
        num_query_groups=8,
        ffn_hidden_size=1856,
        kv_channels=128,
        activation_func=squared_relu,
        gated_linear_unit=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        normalization="RMSNorm",
        add_bias_linear=False,
        init_method_std=0.0173,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=expt_tp_size,
        sequence_parallel=tp_size > 1,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        bias_activation_fusion=False,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        recompute_granularity="selective",
        recompute_modules=["core_attn"],
        moe_ffn_hidden_size=1856,
        num_moe_experts=128,
        moe_router_topk=6,
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_force_load_balancing=args.moe_router_force_load_balancing,
        moe_router_fusion=False,
        moe_aux_loss_coeff=1.0e-4,
        moe_shared_expert_intermediate_size=3712,
        moe_shared_expert_overlap=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        use_fused_weighted_squared_relu=True,
        is_hybrid_model=True,
        mamba_num_heads=64,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_state_dim=128,
        linear_conv_kernel_dim=4,
    )
    config.position_embedding_type = "none"
    config.seq_length = 8192
    config.max_position_embeddings = 8192
    return config


def require_per_token_loss(config: TransformerConfig) -> None:
    """The hetero MIMO loop scales both language and vision grads by real LM tokens."""
    if not config.calculate_per_token_loss:
        raise ValueError("train_hetero.py requires calculate_per_token_loss=True")


def radio_vision_config(args: argparse.Namespace, tp_size: int, pp_size: int) -> TransformerConfig:
    """Build the exact RADIO vision TransformerConfig from the 20L reference provider."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=32,
        hidden_size=1280,
        num_attention_heads=16,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
    config.kv_channels = 80
    config.num_query_groups = 16
    config.ffn_hidden_size = 5120
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.normalization = "LayerNorm"
    config.layernorm_epsilon = 1.0e-6
    config.layernorm_zero_centered_gamma = False
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    # Trigger TransformerBlock's final_layernorm allocation (matches sanj path).
    config.mtp_num_layers = 0
    return config


def nemotron_projection_config(args: argparse.Namespace, tp_size: int) -> TransformerConfig:
    """Build the exact RADIO-to-Nemotron projection config."""
    bf16 = not args.fp32
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2688,
        num_attention_heads=1,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=bf16,
    )
    config.tensor_model_parallel_size = tp_size
    config.ffn_hidden_size = 4 * 5120
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.add_bias_linear = False
    config.activation_func = squared_relu
    config.normalization = "RMSNorm"
    return config


def language_model_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    llm_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the language ModuleSpec for the local language grid."""
    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    ep_pg = getattr(pg_collection, "ep", None) if pg_collection is not None else None
    expt_tp_pg = getattr(pg_collection, "expt_tp", None) if pg_collection is not None else None

    fallback_tp_size = get_grid_dim_size(llm_grid, "tp")
    pp_rank = get_group_rank_or(pp_pg)
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(llm_grid, "pp"))
    tp_size = get_group_size_or(tp_pg, fallback_tp_size)
    ep_size = get_group_size_or(ep_pg, args.llm_ep)
    expt_tp_size = get_group_size_or(expt_tp_pg, args.llm_expt_tp or fallback_tp_size)
    if is_nemotron_moe_vlm(args):
        config = nemotron_language_config(args, tp_size, pp_size, ep_size, expt_tp_size)
        require_per_token_loss(config)
        return ModuleSpec(
            module=MambaModel,
            params={
                "config": config,
                "mamba_stack_spec": mamba_stack_spec,
                "vocab_size": args.vocab_size,
                "max_sequence_length": args.seq_length,
                "pre_process": pp_rank == 0,
                "post_process": pp_rank == pp_size - 1,
                "hybrid_layer_pattern": args.hybrid_layer_pattern,
                "position_embedding_type": "none",
                "share_embeddings_and_output_weights": False,
                "scatter_embedding_sequence_parallel": False,
                "pg_collection": pg_collection,
            },
        )

    num_moe_experts = args.num_moe_experts if args.num_moe_experts > 0 else None
    bf16 = not args.fp32
    moe_kwargs = {}
    if num_moe_experts is not None:
        moe_kwargs = {
            "num_moe_experts": num_moe_experts,
            "moe_router_topk": args.moe_router_topk,
            "moe_router_pre_softmax": args.moe_router_topk == 1,
            "expert_model_parallel_size": ep_size,
            "expert_tensor_parallel_size": expt_tp_size,
            "moe_grouped_gemm": args.moe_grouped_gemm,
        }

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
        **moe_kwargs,
    )
    require_per_token_loss(config)
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_moe_experts, moe_grouped_gemm=args.moe_grouped_gemm
            ),
            "vocab_size": args.vocab_size,
            "max_sequence_length": args.seq_length,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
            "pg_collection": pg_collection,
        },
    )


def vision_submodules_spec(
    args: argparse.Namespace,
    pg_collection: Optional[ProcessGroupCollection],
    encoder_grid: HyperCommGrid,
) -> ModuleSpec:
    """Create the vision ModuleSpec for the local encoder grid."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    pp_pg = getattr(pg_collection, "pp", None) if pg_collection is not None else None
    tp_pg = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    tp_size = get_group_size_or(tp_pg, get_grid_dim_size(encoder_grid, "tp"))
    pp_size = get_group_size_or(pp_pg, get_grid_dim_size(encoder_grid, "pp"))
    pp_rank = get_group_rank_or(pp_pg)
    bf16 = not args.fp32

    if is_nemotron_moe_vlm(args):
        vision_config = radio_vision_config(args, tp_size, pp_size)
        vision_encoder_spec = ModuleSpec(
            module=RADIOEncoderWrapper,
            params={
                "transformer_config": vision_config,
                "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
                "pg_collection": pg_collection,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "patch_dim": args.patch_dim,
                "class_token_len": args.class_token_len,
                "drop_class_token": True,
                "apply_pixel_shuffle": True,
                "force_eval_mode": args.freeze_vit,
                "dynamic_resolution": bool(getattr(args, "dynamic_resolution", False)),
            },
        )
        vision_projection_spec = ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": nemotron_projection_config(args, tp_size),
                "submodules": nemotron_projection_layer_spec().submodules,
                "projector_type": "mlp",
                "input_size": 5120,
                "tp_group": tp_pg if is_process_group_member(tp_pg) else None,
            },
        )
        return ModuleSpec(
            module=VisionModalitySubmodules,
            params={"pg_collection": pg_collection},
            submodules={
                "encoders": {NEMOTRON_VISION_ENCODER_KEY: vision_encoder_spec},
                "input_projections": [vision_projection_spec],
            },
        )

    vision_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        calculate_per_token_loss=True,
    )
    vision_encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": get_gpt_layer_with_transformer_engine_spec(),
            "pg_collection": pg_collection,
            "pre_process": pp_rank == 0,
            "post_process": pp_rank == pp_size - 1,
        },
    )

    projection_config = TransformerConfig(
        num_layers=1, hidden_size=args.hidden_size, num_attention_heads=1
    )
    projection_config.ffn_hidden_size = args.hidden_size
    projection_config.activation_func = torch.nn.functional.gelu

    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,
            "tp_group": tp_pg if is_process_group_member(tp_pg) else None,
        },
    )

    return ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {MOCK_VISION_ENCODER_KEY: vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )
