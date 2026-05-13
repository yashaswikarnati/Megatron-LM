# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Argument handling for the standalone heterogeneous MIMO training loop."""

from __future__ import annotations

import argparse

from examples.mimo.data.hetero_mock import validate_mock_data_args
from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_20L_MODEL_PROVIDER,
    NEMOTRON_54L_MODEL_PROVIDER,
    add_model_provider_args,
    prepare_model_provider_args,
    validate_model_provider_args,
)


def parse_args() -> argparse.Namespace:
    """Parse standalone hetero MIMO loop arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Standalone heterogeneous MIMO training loop. "
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
    grid.add_argument("--llm-expt-dp", type=int, default=None)

    add_model_provider_args(parser)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--enable-experimental",
        action="store_true",
        help="Enable Megatron experimental kernels/features used by some MoE performance paths.",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset-provider", choices=["mock", "energon_multimodal"], default="mock")
    data.add_argument("--data-path", type=str, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--packing-buffer-size", type=int, default=None)
    data.add_argument("--shuffle-buffer-size", type=int, default=100)
    data.add_argument("--max-samples-per-sequence", type=int, default=100)
    data.add_argument(
        "--validate-energon-data-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Check that encoder and LLM Energon readers start from matching samples. "
            "This is disabled by default because the validation all-gather is expensive at scale."
        ),
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
        default=False,
        help=(
            "Enable DDP gradient-reduce overlap for the language module. Vision encoder DDP "
            "keeps overlap disabled because actual-data batches may be text-only."
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


def prepare_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Apply presets, resolve runtime args, and validate the hetero layout."""
    prepare_model_provider_args(args)
    return validate_args(args, world_size)


def validate_args(args: argparse.Namespace, world_size: int) -> tuple[int, int]:
    """Validate the current disjoint-grid training layout."""
    if args.encoder_cp != 1 or args.llm_cp != 1:
        raise ValueError("Phase 2 mock training currently supports CP=1 only")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1")

    validate_model_provider_args(args)
    if args.dataset_provider == "mock":
        validate_mock_data_args(args)
    else:
        validate_energon_data_args(args)
    if args.num_moe_experts > 0 and args.num_moe_experts % args.llm_ep != 0:
        raise ValueError("--num-moe-experts must be divisible by --llm-ep")
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("--micro-batch-size * --llm-dp must be divisible by --encoder-dp")

    encoder_size = args.encoder_tp * args.encoder_cp * args.encoder_pp * args.encoder_dp
    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    encoder_ranks = set(range(args.encoder_offset, args.encoder_offset + encoder_size))
    llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
    all_ranks = set(range(world_size))

    if not encoder_ranks.isdisjoint(llm_ranks):
        raise ValueError(
            "train_hetero.py currently expects disjoint module rank spans; "
            f"module rank spans overlap at {sorted(encoder_ranks & llm_ranks)}"
        )
    if encoder_ranks | llm_ranks != all_ranks:
        raise ValueError(
            "The non-colocated module grids must cover every torchrun rank exactly once; "
            f"covered={sorted(encoder_ranks | llm_ranks)}, world={sorted(all_ranks)}"
        )

    return encoder_size, llm_size


def validate_energon_data_args(args: argparse.Namespace) -> None:
    """Validate the actual-data non-colocated path."""
    if not args.data_path:
        raise ValueError("--data-path is required for --dataset-provider energon_multimodal")
    if not args.tokenizer_model:
        raise ValueError("--tokenizer-model is required for --dataset-provider energon_multimodal")
    if args.model_provider not in (NEMOTRON_20L_MODEL_PROVIDER, NEMOTRON_54L_MODEL_PROVIDER):
        raise ValueError(
            "energon_multimodal is currently wired for Nemotron MoE VLM providers"
        )
    if args.encoder_pp != 1 or args.llm_pp != 1:
        raise ValueError("energon_multimodal currently supports encoder and LLM PP size 1")
    if args.encoder_dp > args.llm_dp:
        raise ValueError(
            "energon_multimodal currently supports fan-out only: --encoder-dp must be "
            "<= --llm-dp"
        )
    if args.llm_dp % args.encoder_dp != 0:
        raise ValueError(
            "energon_multimodal fan-out requires --llm-dp to be divisible by --encoder-dp"
        )
    if args.encoder_dp != args.llm_dp and args.micro_batch_size != 1:
        raise ValueError(
            "energon_multimodal fan-out currently requires --micro-batch-size 1 so bridge "
            "splits map one encoder sample to one LLM DP lane"
        )
    if args.packing_buffer_size is not None and args.packing_buffer_size > 0:
        if args.micro_batch_size != 1:
            raise ValueError(
                "Energon packed multimodal batches currently require --micro-batch-size 1"
            )
