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
    grid.add_argument(
        "--llm-only",
        action="store_true",
        help=(
            "Run only the MIMO language module on the LLM grid. This keeps the MIMO "
            "training/data path but does not create encoder ranks or bridge communicators."
        ),
    )

    add_model_provider_args(parser)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--enable-experimental",
        action="store_true",
        help="Enable Megatron experimental kernels/features used by some MoE performance paths.",
    )
    runtime.add_argument(
        "--timeline-profile",
        action="store_true",
        help="Write rank-local 1F1B timeline JSONL traces for selected debug ranks.",
    )
    runtime.add_argument(
        "--timeline-dir",
        type=str,
        default=None,
        help="Directory for rank-local timeline JSONL traces.",
    )
    runtime.add_argument(
        "--timeline-ranks",
        type=str,
        default="dp-replica",
        help="'dp-replica', 'all', or comma-separated global ranks to trace.",
    )
    runtime.add_argument(
        "--timeline-dp-replica",
        type=int,
        default=0,
        help="Dense data-parallel replica to trace when --timeline-ranks=dp-replica.",
    )
    runtime.add_argument(
        "--timeline-cuda-events",
        action="store_true",
        help="Also record CUDA event elapsed time for compute events.",
    )
    runtime.add_argument(
        "--timeline-nvtx",
        action="store_true",
        help="Push NVTX ranges with timeline event names for Nsight Systems.",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset-provider", choices=["mock", "energon_multimodal"], default="mock")
    data.add_argument("--data-path", type=str, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--packing-buffer-size", type=int, default=None)
    data.add_argument("--shuffle-buffer-size", type=int, default=100)
    data.add_argument("--max-samples-per-sequence", type=int, default=100)
    data.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Directory where hetero Energon dataloader state is saved with checkpoints.",
    )
    data.add_argument(
        "--dataloader-load",
        type=str,
        default=None,
        help=(
            "Directory to load hetero Energon dataloader state from on resume. "
            "Defaults to --dataloader-save when omitted."
        ),
    )
    data.add_argument(
        "--validate-energon-data-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Check that encoder and LLM Energon readers start from matching samples. "
            "This is disabled by default because the validation all-gather is expensive at scale."
        ),
    )
    data.add_argument(
        "--energon-sample-trace-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for per-source-rank JSONL sample signatures used to validate "
            "checkpoint replay. Disabled when unset."
        ),
    )

    train = parser.add_argument_group("training")
    train.add_argument("--micro-batch-size", type=int, default=2)
    train.add_argument("--global-batch-size", type=int, default=None)
    train.add_argument("--num-microbatches", type=int, default=2)
    train.add_argument("--train-iters", type=int, default=2)
    train.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help=(
            "Total training budget in consumed samples. When set, --train-iters is "
            "re-derived as ceil(train_samples / global_batch_size)."
        ),
    )
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--min-lr", type=float, default=None)
    train.add_argument(
        "--lr-decay-style",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "inverse-square-root", "WSD"],
    )
    train.add_argument("--lr-warmup-iters", type=int, default=0)
    train.add_argument("--lr-decay-iters", type=int, default=None)
    train.add_argument(
        "--lr-warmup-samples",
        type=int,
        default=None,
        help="LR warmup duration in consumed samples. Overrides --lr-warmup-iters when set.",
    )
    train.add_argument(
        "--lr-decay-samples",
        type=int,
        default=None,
        help="LR decay duration in consumed samples. Overrides --lr-decay-iters when set.",
    )
    train.add_argument(
        "--lr-wsd-decay-samples",
        type=int,
        default=None,
        help=(
            "Length of the WSD decay tail in consumed samples. Required when "
            "--lr-decay-style=WSD."
        ),
    )
    train.add_argument(
        "--lr-wsd-decay-style",
        type=str,
        default=None,
        choices=["linear", "cosine", "exponential", "minus_sqrt"],
        help="Decay-style applied during the WSD tail.",
    )
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
        "--overlap-param-gather",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable distributed-optimizer param all-gather overlap with forward compute. "
            "Requires --use-distributed-optimizer (already on for the hetero loop)."
        ),
    )
    train.add_argument(
        "--ddp-bucket-size",
        type=int,
        default=10000,
        help="DDP bucket size in parameters. Use 0 for a single unbounded bucket.",
    )
    train.add_argument(
        "--ddp-num-buckets",
        type=int,
        default=None,
        help=(
            "If set, DDP bucket_size is derived from num_parameters // ddp_num_buckets "
            "(mutually exclusive with --ddp-bucket-size > 0)."
        ),
    )
    train.add_argument(
        "--ddp-pad-buckets-for-high-nccl-busbw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pad DDP bucket sizes to a multiple of 2^16 so NCCL collectives have high "
            "bus bandwidth at large DP counts."
        ),
    )
    train.add_argument(
        "--correct-encoder-grad-for-partial-participation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When some encoder DP ranks see text-only batches, scale vision "
            "grads post-DP-reduce by encoder_dp_size / participation_count so "
            "the vision encoder learns at full rate instead of being diluted. "
            "Default on; pass --no-correct-encoder-grad-for-partial-participation "
            "to disable."
        ),
    )
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--log-interval", type=int, default=1)
    train.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directory for tensorboard scalar logs. When set, the language "
        "logging rank writes lm_loss/grad-norm/learning-rate/etc. each log "
        "interval, matching the scalar keys used by Megatron's standard "
        "training_log so hetero and reference runs can be diffed in TB.",
    )

    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save distributed checkpoints into. Each save creates iter_NNNNNNN/.",
    )
    ckpt.add_argument(
        "--load",
        type=str,
        default=None,
        help=(
            "Directory to resume from. If the directory has no completed checkpoint, "
            "training starts from iteration 0."
        ),
    )
    ckpt.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help=(
            "Iteration interval between checkpoint saves. When unset, --save still "
            "produces exactly one checkpoint at --train-iters (the final iter). Set to "
            "an integer >=1 for periodic saves; the final iter is also always saved."
        ),
    )
    ckpt.add_argument("--no-save-optim", action="store_true", help="Skip optimizer state on save.")
    ckpt.add_argument(
        "--no-load-optim",
        action="store_true",
        help="Skip optimizer state on load (fresh optimizer at the loaded iteration).",
    )
    ckpt.add_argument(
        "--no-load-scheduler", action="store_true", help="Skip LR/WD scheduler state on load."
    )
    ckpt.add_argument(
        "--no-save-rng", action="store_true", help="Skip Python/NumPy/Torch/CUDA RNG state on save."
    )
    ckpt.add_argument(
        "--no-load-rng",
        action="store_true",
        help="Skip Python/NumPy/Torch/CUDA RNG state on load (start with fresh RNG).",
    )
    ckpt.add_argument(
        "--finetune",
        action="store_true",
        help=(
            "Treat the load directory as a pretrained checkpoint: restart from iteration 0 and "
            "skip optimizer + scheduler state regardless of the other flags."
        ),
    )
    ckpt.add_argument(
        "--load-nemotron-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a flat Nemotron-format VLM dist-ckpt. Loads weights and "
            "starts training at iter 0; mutually exclusive with --load."
        ),
    )
    ckpt.add_argument(
        "--load-vision-from",
        type=str,
        default=None,
        help=(
            "Path to a Megatron-Bridge RADIO encoder DCP (torch DCP layout with "
            "`model.vision_model.*` keys; e.g. post-c-radio-omni). Loads only "
            "the vision encoder; LLM and vision projection are random-init. "
            "Training starts at iter 0; mutually exclusive with --load and "
            "--load-nemotron-checkpoint."
        ),
    )
    ckpt.add_argument(
        "--dist-ckpt-optim-fully-reshardable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the 'fully_reshardable' DistributedOptimizer sharding type so a saved "
            "checkpoint can be reloaded under a different TP/EP layout. Defaults to False "
            "('dp_reshardable', DP-only reshardable, lower save-time memory). Enable when "
            "you intend to change --llm-tp / --llm-ep on resume. WARNING: this gathers "
            "the full per-DP optimizer state on DP rank 0 during save; on <80 GB GPUs "
            "(or when running near peak memory) the gather will OOM. Prefer leaving this "
            "False unless you actually need cross-TP/EP resharding."
        ),
    )

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
    if args.timeline_dp_replica < 0:
        raise ValueError("--timeline-dp-replica must be >= 0")

    validate_model_provider_args(args)
    if args.dataset_provider == "mock":
        validate_mock_data_args(args)
    else:
        validate_energon_data_args(args)
    if args.num_moe_experts > 0 and args.num_moe_experts % args.llm_ep != 0:
        raise ValueError("--num-moe-experts must be divisible by --llm-ep")
    if args.save_interval is not None and args.save_interval < 1:
        raise ValueError("--save-interval must be >= 1 when set")
    if args.save_interval is not None and args.save is None:
        raise ValueError("--save-interval requires --save")

    # Sample-based scheduler resolution: when --train-samples is set, derive
    # --train-iters from it using the (now-known) global batch size. The
    # OptimizerParamScheduler tracks "steps" in units of consumed samples, so
    # the sample-based knobs flow through unchanged downstream.
    if args.train_samples is not None:
        derived_gbs = args.micro_batch_size * args.num_microbatches * args.llm_dp
        gbs = args.global_batch_size if args.global_batch_size is not None else derived_gbs
        if gbs <= 0:
            raise ValueError(
                "--train-samples requires a positive derived/explicit --global-batch-size"
            )
        import math as _math

        derived_iters = _math.ceil(args.train_samples / gbs)
        args.train_iters = derived_iters

    if args.lr_decay_style == "WSD":
        if args.lr_wsd_decay_samples is None:
            raise ValueError("--lr-decay-style=WSD requires --lr-wsd-decay-samples")
        if args.lr_wsd_decay_style is None:
            raise ValueError("--lr-decay-style=WSD requires --lr-wsd-decay-style")

    llm_size = args.llm_tp * args.llm_cp * args.llm_pp * args.llm_dp
    if args.llm_only:
        if args.llm_offset != 0:
            raise ValueError(
                "--llm-only requires --llm-offset 0 so language ranks cover WORLD_SIZE"
            )
        llm_ranks = set(range(args.llm_offset, args.llm_offset + llm_size))
        all_ranks = set(range(world_size))
        if llm_ranks != all_ranks:
            raise ValueError(
                "--llm-only requires the language grid to cover every torchrun rank exactly "
                f"once; covered={sorted(llm_ranks)}, world={sorted(all_ranks)}"
            )
        return 0, llm_size

    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("--micro-batch-size * --llm-dp must be divisible by --encoder-dp")

    encoder_size = args.encoder_tp * args.encoder_cp * args.encoder_pp * args.encoder_dp
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
        raise ValueError("energon_multimodal is currently wired for Nemotron MoE VLM providers")
    if args.llm_pp != 1:
        raise ValueError("energon_multimodal currently supports LLM PP size 1")
    if args.llm_only:
        return
    if args.encoder_pp != 1:
        raise ValueError("energon_multimodal currently supports encoder PP size 1")
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
