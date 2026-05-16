# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer and scheduler construction for heterogeneous MIMO training."""

from __future__ import annotations

import argparse

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def build_optimizer(args: argparse.Namespace, model: MimoModel):
    """Build the MIMO optimizer for active hetero module optimizers."""
    return get_mimo_optimizer(
        model,
        OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            clip_grad=args.clip_grad,
            bf16=not args.fp32,
            use_distributed_optimizer=True,
            log_num_zeros_in_grad=args.log_num_zeros_in_grad,
        ),
    )


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
    """Build the MCore optimizer parameter scheduler.

    The scheduler tracks "steps" in units of consumed samples (incremented by the
    global batch size per call). Sample-based knobs take precedence when set;
    iter-based knobs are converted via iter * global_batch_size for back-compat.
    """
    global_batch_size = get_global_batch_size(args)
    if args.lr_warmup_samples is not None:
        lr_warmup_steps = args.lr_warmup_samples
    else:
        lr_warmup_steps = args.lr_warmup_iters * global_batch_size
    if args.lr_decay_samples is not None:
        lr_decay_steps = args.lr_decay_samples
    else:
        lr_decay_iters = (
            args.lr_decay_iters if args.lr_decay_iters is not None else args.train_iters
        )
        lr_decay_steps = lr_decay_iters * global_batch_size
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=args.lr,
        min_lr=args.min_lr if args.min_lr is not None else 0.0,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.weight_decay,
        end_wd=args.weight_decay,
        wd_incr_steps=args.train_iters * global_batch_size,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=args.lr_wsd_decay_samples,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )
