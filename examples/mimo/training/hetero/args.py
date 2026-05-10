# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Argument handling for the standalone heterogeneous MIMO training loop."""

from __future__ import annotations

import argparse

from examples.mimo.utils.hetero import (
    MOCK_MODEL_PRESET,
    NEMOTRON_20L_DEFAULT_STAGE,
    NEMOTRON_20L_IMAGE_SEQ_PER_TILE,
    NEMOTRON_20L_MAX_NUM_TILES,
    NEMOTRON_20L_MODEL_PRESET,
    is_nemotron_20l,
)


def parse_args() -> argparse.Namespace:
    """Parse standalone hetero MIMO loop arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Standalone heterogeneous MIMO mock training loop. "
            "This entrypoint owns one HyperCommGrid per MIMO module."
        )
    )

    grid = parser.add_argument_group("module grids")
    grid.add_argument("--encoder-offset", type=int, default=0)
    grid.add_argument("--encoder-tp", type=int, default=2)
    grid.add_argument("--encoder-cp", type=int, default=1)
    grid.add_argument("--encoder-pp", type=int, default=2)
    grid.add_argument("--encoder-dp", type=int, default=1)
    grid.add_argument("--encoder-ep", type=int, default=1)
    grid.add_argument("--encoder-expt-tp", type=int, default=None)
    grid.add_argument("--encoder-expt-dp", type=int, default=None)
    grid.add_argument("--llm-offset", type=int, default=4)
    grid.add_argument("--llm-tp", type=int, default=1)
    grid.add_argument("--llm-cp", type=int, default=1)
    grid.add_argument("--llm-pp", type=int, default=2)
    grid.add_argument("--llm-dp", type=int, default=2)
    grid.add_argument("--llm-ep", type=int, default=2)
    grid.add_argument("--llm-expt-tp", type=int, default=1)
    grid.add_argument("--llm-expt-dp", type=int, default=1)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-preset",
        choices=[MOCK_MODEL_PRESET, NEMOTRON_20L_MODEL_PRESET],
        default=MOCK_MODEL_PRESET,
        help="Model config preset. The Nemotron preset matches the 20L reference script.",
    )
    model.add_argument("--hidden-size", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--vocab-size", type=int, default=512)
    model.add_argument("--seq-length", type=int, default=32)
    model.add_argument("--image-seq-length", type=int, default=None)
    model.add_argument("--image-token-id", type=int, default=511)
    model.add_argument("--pad-token-id", type=int, default=0)
    model.add_argument("--image-token", type=str, default="<image>")
    model.add_argument("--tokenizer-model", type=str, default=None)
    model.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    model.add_argument("--image-tag-type", type=str, default="")
    model.add_argument("--force-system-message", action="store_true")
    model.add_argument("--num-moe-experts", type=int, default=4)
    model.add_argument("--moe-router-topk", type=int, default=1)
    model.add_argument("--moe-grouped-gemm", action="store_true")
    model.add_argument("--img-h", type=int, default=512)
    model.add_argument("--img-w", type=int, default=512)
    model.add_argument("--patch-dim", type=int, default=16)
    model.add_argument("--class-token-len", type=int, default=8)
    model.add_argument("--num-image-tiles", type=int, default=NEMOTRON_20L_MAX_NUM_TILES)
    model.add_argument("--freeze-lm", action="store_true", help="Freeze language model params")
    model.add_argument("--freeze-vit", action="store_true", help="Freeze vision encoder params")
    model.add_argument(
        "--freeze-projection", action="store_true", help="Freeze vision projection params"
    )
    model.add_argument(
        "--training-stage",
        choices=["stage1", "stage2", "stage3"],
        default=None,
        help="Nemotron VLM freeze stage. Defaults to stage2 for the 20L preset.",
    )
    model.add_argument(
        "--fp32", action="store_true", help="Build and train in fp32 instead of bf16"
    )

    train = parser.add_argument_group("training")
    train.add_argument("--micro-batch-size", type=int, default=2)
    train.add_argument("--global-batch-size", type=int, default=None)
    train.add_argument("--num-microbatches", type=int, default=2)
    train.add_argument("--train-iters", type=int, default=2)
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--min-lr", type=float, default=None)
    train.add_argument("--lr-decay-style", type=str, default="constant")
    train.add_argument("--lr-warmup-iters", type=int, default=0)
    train.add_argument("--lr-decay-iters", type=int, default=None)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--log-num-zeros-in-grad", action="store_true")
    train.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable DDP gradient-reduce overlap. Disable for parity with the 20L "
            "reference script."
        ),
    )
    train.add_argument(
        "--ddp-bucket-size",
        type=int,
        default=10000,
        help="DDP bucket size. Use 0 for a single unbounded bucket.",
    )
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--log-interval", type=int, default=1)

    return parser.parse_args()


def apply_model_preset(args: argparse.Namespace) -> None:
    """Apply architecture defaults for the selected model preset."""
    if not is_nemotron_20l(args):
        return

    args.num_layers = 20
    args.hidden_size = 2688
    args.num_attention_heads = 32
    args.num_moe_experts = 128
    args.moe_router_topk = 6
    args.moe_grouped_gemm = True
    args.seq_length = 8192
    args.image_seq_length = NEMOTRON_20L_IMAGE_SEQ_PER_TILE * args.num_image_tiles


def apply_training_stage(args: argparse.Namespace) -> None:
    """Apply the reference Nemotron VLM freeze stage defaults."""
    if not is_nemotron_20l(args):
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
    """Resolve the image token id from the reference MultimodalTokenizer when provided."""
    if not is_nemotron_20l(args) or not args.tokenizer_model:
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


def prepare_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Apply presets, resolve runtime args, and validate the hetero layout."""
    apply_model_preset(args)
    apply_training_stage(args)
    resolve_image_token_id(args)
    return validate_args(args, world_size)


def validate_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Validate the Phase 2 non-colocated 1F1B mock-training layout."""
    if args.encoder_cp != 1 or args.llm_cp != 1:
        raise ValueError("Phase 2 mock training currently supports CP=1 only")
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads")
    if args.num_moe_experts > 0 and args.num_moe_experts % args.llm_ep != 0:
        raise ValueError("--num-moe-experts must be divisible by --llm-ep")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1")
    if not 0 <= args.image_token_id < args.vocab_size:
        raise ValueError("--image-token-id must be within --vocab-size")
    if not 0 <= args.pad_token_id < args.vocab_size:
        raise ValueError("--pad-token-id must be within --vocab-size")

    image_seq_length = args.image_seq_length or args.seq_length // 2
    if image_seq_length >= args.seq_length:
        raise ValueError("--image-seq-length must be smaller than --seq-length")
    if args.seq_length - image_seq_length < 2:
        raise ValueError("mock next-token training needs at least two text tokens")
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("--micro-batch-size * --llm-dp must be divisible by --encoder-dp")

    encoder_size = args.encoder_tp * args.encoder_cp * args.encoder_pp * args.encoder_dp
    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    encoder_ranks = set(range(args.encoder_offset, args.encoder_offset + encoder_size))
    llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
    all_ranks = set(range(world_size))

    if not encoder_ranks.isdisjoint(llm_ranks):
        raise ValueError(
            "Phase 2 train_hetero.py supports non-colocated 1F1B only; "
            f"module rank spans overlap at {sorted(encoder_ranks & llm_ranks)}"
        )
    if encoder_ranks | llm_ranks != all_ranks:
        raise ValueError(
            "The non-colocated module grids must cover every torchrun rank exactly once; "
            f"covered={sorted(encoder_ranks | llm_ranks)}, world={sorted(all_ranks)}"
        )

    return encoder_size, llm_size
