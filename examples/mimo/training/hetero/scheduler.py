# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer scheduler helpers for heterogeneous MIMO training."""

from __future__ import annotations

import argparse

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def get_global_batch_size(args: argparse.Namespace) -> int:
    """Return the language-side global batch size for scheduler accounting."""
    derived_global_batch_size = args.micro_batch_size * args.num_microbatches * args.llm_dp
    if args.global_batch_size is None:
        return derived_global_batch_size
    if args.global_batch_size != derived_global_batch_size:
        raise ValueError(
            "--global-batch-size must equal "
            "--micro-batch-size * --num-microbatches * --llm-dp in this hetero loop "
            f"({derived_global_batch_size}); got {args.global_batch_size}"
        )
    return args.global_batch_size


def build_optimizer_param_scheduler(args: argparse.Namespace, optimizer) -> OptimizerParamScheduler:
    """Build the MCore optimizer parameter scheduler using Megatron train-iters semantics."""
    global_batch_size = get_global_batch_size(args)
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.train_iters
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=args.lr,
        min_lr=args.min_lr if args.min_lr is not None else 0.0,
        lr_warmup_steps=args.lr_warmup_iters * global_batch_size,
        lr_decay_steps=lr_decay_iters * global_batch_size,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.weight_decay,
        end_wd=args.weight_decay,
        wd_incr_steps=args.train_iters * global_batch_size,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )
