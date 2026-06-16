# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero MIMO entry that runs on the *stock* Megatron ``train()`` loop.

This is PR-E5 of the NMFW-516 hetero-MIMO upstreaming effort: the end-to-end
integration that drives the Nemotron6-MoE VLM (20L) through Megatron's
production ``megatron/training/training.py::train()`` instead of the prototype's
bespoke loop. Scope: the NON-COLOCATED path (encoder grid and language grid on
disjoint rank spans), mock data.

Why not stock ``pretrain()``?
=============================
Stock ``pretrain(cfg_container, ...)`` does three things this entry cannot use
verbatim:

  1. It calls ``initialize_megatron`` -> ``_initialize_distributed`` ->
     ``mpu.initialize_model_parallel(...)``. MIMO must NOT initialize the MPU
     globals: each MIMO module owns its own ``HyperCommGrid`` with disjoint rank
     spans, so a single world-spanning MPU layout is wrong. We therefore
     replicate only the *non-MPU* pieces of ``initialize_megatron``
     (``set_global_variables`` minus the num-microbatches calculator wiring,
     random seeds, torch.distributed bring-up via MM1) and skip MPU init.
  2. It calls ``setup_model_and_optimizer`` -> ``get_model`` which re-wraps a
     single top-level DDP and builds a single optimizer over the MPU groups. The
     MIMO model needs PER-SUBMODULE DDP (one per module grid) and the
     ``MimoOptimizer`` (one inner optimizer per active module). ``E4
     build_mimo_runtime`` already assembles that; we feed its model list straight
     to ``train()``.
  3. It builds train/valid/test datasets via a dataset provider. MIMO uses the
     role-aware iterator from ``select_data_iterator`` (mock here).

What this entry replicates from ``initialize_megatron`` (and what it skips)
===========================================================================
Replicated (needed global state ``train()`` / the schedule read):

  * ``set_global_variables(args, build_tokenizer=False)`` MINUS its
    num-microbatches calculator call -- args global, timers, tensorboard/wandb,
    experimental flag. We build the calculator ourselves AFTER fixing
    ``args.data_parallel_size`` (see below), because ``set_global_variables``
    would key it on the stock (world-derived) DP size.
  * ``_set_random_seed`` equivalent: ``random`` / ``numpy`` / ``torch`` seeds
    (matches the prototype's ``loop.py`` so dataset-construction RNG agrees).
  * torch.distributed bring-up via MM1 ``initialize_distributed`` (NCCL +
    global memory buffer; NO MPU groups).

Skipped (and why):

  * ``mpu.initialize_model_parallel`` -- MIMO owns per-module grids; a global MPU
    layout is wrong for disjoint spans. See (1) above.
  * ``_compile_dependencies`` -- only compiles the C++ dataset index builder,
    which the mock iterator does not use.
  * ``_init_autoresume`` / ``_initialize_tp_communicators`` -- autoresume and TP
    comm-overlap user buffers are not exercised by this mock run.
  * tokenizer build -- mock data reads ``args.vocab_size`` directly and the
    provider's ``_vocab_size`` falls back to it, so no tokenizer is needed.

The ``data_parallel_size`` fix
==============================
Stock ``validate_args`` recomputes ``args.data_parallel_size = world_size //
(tp*pp*cp)`` from the *stock* (world-spanning) parallelism flags. For the
disjoint hetero layout that value is wrong: the language grid is only ``llm_dp``
wide. ``train()`` reads ``get_num_microbatches()`` (keyed on
``global_batch_size / (micro_batch_size * data_parallel_size)``) and
``mpu.get_data_parallel_world_size()`` for sample accounting. We therefore:

  * set ``args.data_parallel_size = args.llm_dp`` AFTER ``validate_args``;
  * build the num-microbatches calculator with that DP so it yields the
    script's ``--num-microbatches`` (gbs 8 / (mbs 1 * dp 2) = 4);
  * pin ``parallel_state._MPU_DATA_PARALLEL_WORLD_SIZE = llm_dp`` so train()'s
    four ``mpu.get_data_parallel_world_size()`` reads return the language DP size
    without a full MPU init (a bootstrap/MPU-materialization compatibility
    point, per CLAUDE.md).

The grad-finalization clobber
=============================
``E4 build_mimo_runtime`` installs the MIMO dual grad-finalization hook
(``configure_grad_sync`` sets ``model.config.finalize_model_grads_func`` to the
encoder+language finalizer). But stock ``train()`` unconditionally reassigns
``config.finalize_model_grads_func = finalize_model_grads`` (the module-level
import in ``megatron.training.training``). To keep the MIMO hook without editing
core, we monkeypatch that module-level symbol to the MIMO finalizer BEFORE
calling ``train()``. ``config.grad_scale_func`` is also reassigned by ``train()``
to ``optimizer.scale_loss``; for bf16 with no grad scaler the MimoOptimizer loss
scale is 1.0, so ``scale_loss(loss) == loss`` -- behaviorally identical to the
MIMO ``lambda loss: loss``, so no patch is needed there.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import megatron.training.training as training_module
from examples.mimo.model_providers.nemotron_moe_vlm import (
    add_model_provider_args,
    prepare_model_provider_args,
    validate_model_provider_args,
)
from examples.mimo.training.args import add_hetero_grid_args, validate_hetero_grid_args
from examples.mimo.training.bootstrap import build_mimo_runtime
from examples.mimo.training.data import add_data_args
from examples.mimo.training.distributed import (
    initialize_distributed,
    print_rank_0,
    shutdown_distributed,
)
from examples.mimo.training.step import mimo_forward_step
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables as _stock_set_global_variables

# args ``set_global_variables`` requires but which the script does not pass. We
# skip the tokenizer build (mock data reads args.vocab_size directly), so no
# tokenizer-type default is needed.
_ARGS_DEFAULTS = {
    # No dataset path / tokenizer for the mock run.
}


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Compose the model-provider arg group with the hetero grid arg group.

    Both register their own argparse groups onto the stock parser. We also add
    ``--num-microbatches`` here: stock Megatron derives the microbatch count from
    ``global_batch_size / (micro_batch_size * data_parallel_size)`` and has no
    such flag, but the prototype (and our run script) pass it explicitly.
    """
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_data_args(parser)
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=1,
        help="Explicit microbatch count for the hetero MIMO loop (stock derives this "
        "from gbs/mbs/dp; the hetero layout keys it on --llm-dp instead).",
    )
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Run the stock arg pipeline plus the MIMO preset/validation hooks.

    Sequence (handoff section 2):
      parse_args(extra) -> prepare_model_provider_args (preset, BEFORE validate)
      -> stock validate_args -> validate_model_provider_args
      -> validate_hetero_grid_args -> data_parallel_size fix.
    """
    args = parse_args(extra_args_provider)

    # Apply the Nemotron preset BEFORE stock validation so preset-derived sizes
    # (num_layers, hybrid pattern, seq_length, num_experts, image_seq_length, ...)
    # flow into validate_args.
    prepare_model_provider_args(args)

    # Stock validation. ``defaults`` fills only args the user left at default.
    validate_args(args, _ARGS_DEFAULTS)

    # MIMO-specific validation (runs after stock so padded_vocab_size / num_experts
    # are populated).
    validate_model_provider_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    validate_hetero_grid_args(args, world_size)

    # --- data_parallel_size fix (see module docstring). --------------------
    # Stock validate_args set args.data_parallel_size = world_size//(tp*pp*cp).
    # Re-key it on the language grid's DP so get_num_microbatches() and sample
    # accounting reflect the disjoint hetero layout.
    args.data_parallel_size = args.llm_dp

    # Stock throughput/FLOPs logging does ``1 + args.mtp_num_layers``; the Nemotron
    # preset leaves it None (MTP off), which TypeErrors. Default to 0 (no MTP).
    if getattr(args, "mtp_num_layers", None) is None:
        args.mtp_num_layers = 0

    return args


def _setup_globals(args: argparse.Namespace) -> None:
    """Set non-MPU global state, then build the num-microbatches calculator on llm_dp.

    Replicates the non-MPU portion of ``initialize_megatron`` (see module
    docstring). ``set_global_variables`` would build the num-microbatches
    calculator keyed on the (now-fixed) ``args.data_parallel_size``; we call it
    with ``build_tokenizer=False`` so no tokenizer is constructed. Because
    ``data_parallel_size`` is already ``llm_dp`` by this point, the calculator it
    builds is correct, so we do not rebuild it.
    """
    _stock_set_global_variables(args, build_tokenizer=False)
    if args.enable_experimental:
        set_experimental_flag(True)

    # Defensive: ensure the num-microbatches calculator reflects llm_dp even if a
    # future set_global_variables stops keying on args.data_parallel_size.
    from megatron.core.num_microbatches_calculator import get_num_microbatches as _gnmb

    expected = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
    if _gnmb() != expected:
        # Rebuild to the llm_dp-keyed value.
        init_num_microbatches_calculator(
            rank=args.rank,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=args.data_parallel_size,
            decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
        )


def _seed_everything(args: argparse.Namespace) -> None:
    """Seed python/numpy/torch RNG (matches the prototype loop + stock _set_random_seed)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def _build_optimizer(args: argparse.Namespace, model):
    """Build the MimoOptimizer over the per-submodule-DDP MimoModel.

    Mirrors the prototype's ``hetero/optimizer.py::build_optimizer``: a single
    ``OptimizerConfig`` shared by every active module optimizer, distributed
    optimizer on, bf16 unless --fp32.
    """
    return get_mimo_optimizer(
        model[0],
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


def _build_scheduler(args: argparse.Namespace, optimizer) -> OptimizerParamScheduler:
    """Build the stock OptimizerParamScheduler keyed on the llm_dp global batch.

    Ports ``hetero/optimizer.py::build_optimizer_param_scheduler``: scheduler
    "steps" are consumed samples (incremented by the global batch size per
    optimizer step), so iter-based knobs convert via ``iters * global_batch_size``.
    """
    global_batch_size = args.global_batch_size

    if getattr(args, "lr_warmup_samples", None) is not None:
        lr_warmup_steps = args.lr_warmup_samples
    else:
        lr_warmup_steps = args.lr_warmup_iters * global_batch_size

    if getattr(args, "lr_decay_samples", None) is not None:
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
        wsd_decay_steps=getattr(args, "lr_wsd_decay_samples", None),
        lr_wsd_decay_style=getattr(args, "lr_wsd_decay_style", None),
    )


def _install_mimo_grad_finalize(model) -> None:
    """Preserve the MIMO dual grad-finalization hook across stock train()'s clobber.

    stock ``train()`` reassigns ``config.finalize_model_grads_func`` to the
    module-level ``megatron.training.training.finalize_model_grads`` symbol. We
    repoint that symbol to the MIMO finalizer that ``configure_grad_sync`` already
    installed on ``model.config`` (in build_mimo_runtime). After train() runs its
    ``config.finalize_model_grads_func = finalize_model_grads`` line, the config
    therefore carries the MIMO hook.
    """
    mimo_finalize = model[0].config.finalize_model_grads_func
    assert mimo_finalize is not None, (
        "expected build_mimo_runtime -> configure_grad_sync to install a MIMO "
        "finalize_model_grads_func on the language config"
    )
    training_module.finalize_model_grads = mimo_finalize


def _install_safe_flops() -> None:
    """Neutralize stock throughput/FLOPs accounting for the hetero MIMO model.

    ``num_floating_point_operations`` derives a single homogeneous model's FLOPs
    from the global (language) args and runs on every rank. That is ill-defined
    for the disjoint encoder(RADIO)+language(hybrid-MoE) layout: encoder ranks
    IndexError parsing the language MoE/hybrid pattern, and the hybrid path trips
    on MTP. FLOPs/throughput is cosmetic for the smoke milestone, so wrap it to
    return 0 on failure. TODO(NMFW-516): proper per-module hetero FLOPs accounting.
    """
    _orig = training_module.num_floating_point_operations

    def _safe(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except Exception:
            return 0

    training_module.num_floating_point_operations = _safe


def _set_mpu_data_parallel_world_size(args: argparse.Namespace) -> None:
    """Pin the MPU DP world size to llm_dp for train()'s sample accounting.

    train() reads ``mpu.get_data_parallel_world_size()`` (4 sites) for consumed-
    sample / batch-size bookkeeping. That getter returns the module global
    ``_MPU_DATA_PARALLEL_WORLD_SIZE`` when set, before touching any MPU group, so
    pinning it lets train() run without a full MPU init. This is a bootstrap /
    MPU-materialization compatibility point (see CLAUDE.md "Megatron Core Process
    Groups").
    """
    parallel_state._MPU_DATA_PARALLEL_WORLD_SIZE = args.llm_dp


def main() -> None:
    """Program entrypoint: stock-args MIMO run on the stock train() loop."""
    if os.environ.get("MIMO_BRIDGE_DEBUG"):
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format=f"[rank{os.environ.get('RANK', '?')}] %(message)s",
            force=True,
        )
    args = _parse_and_validate()

    # Non-MPU global setup, then torch.distributed bring-up (no MPU init).
    _setup_globals(args)
    initialize_distributed()
    _seed_everything(args)
    _set_mpu_data_parallel_world_size(args)

    # Per-rank runtime: per-submodule-DDP MimoModel + topology + communicator +
    # role-aware data iterator + grad-sync hook (E4).
    rt = build_mimo_runtime(args)

    optimizer = _build_optimizer(args, rt.model)
    opt_param_scheduler = _build_scheduler(args, optimizer)

    # Keep the MIMO grad-finalization hook alive past stock train()'s reassign.
    _install_mimo_grad_finalize(rt.model)

    # Neutralize stock FLOPs/throughput accounting (ill-defined for the hetero model).
    _install_safe_flops()

    # Bookkeeping fields stock train() reads that setup_model_and_optimizer /
    # the dataset builder would normally set.
    args.iteration = 0
    args.num_floating_point_operations_so_far = 0
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.do_train = True
    args.do_valid = False
    args.do_test = False

    config = rt.model[0].config

    print_rank_0(
        "Starting hetero MIMO training on stock train(): "
        f"world_size={torch.distributed.get_world_size()}, "
        f"llm_dp={args.llm_dp}, num_microbatches={args.num_microbatches}, "
        f"global_batch_size={args.global_batch_size}, train_iters={args.train_iters}"
    )

    try:
        training_module.train(
            forward_step_func=mimo_forward_step,
            model=rt.model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            train_data_iterator=rt.data_iterator,
            valid_data_iterator=None,
            process_non_loss_data_func=None,
            config=config,
            checkpointing_context={},
            non_loss_data_func=None,
            p2p_communicator=rt.communicator,
            schedule_pg_collection=rt.topology.schedule_pg_collection,
        )
        torch.distributed.barrier()
        print_rank_0("Hetero MIMO training (stock loop) completed")
    finally:
        if hasattr(rt.model[0], "destroy"):
            rt.model[0].destroy()
        rt.topology.destroy()
        shutdown_distributed()


if __name__ == "__main__":
    main()
