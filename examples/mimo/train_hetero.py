# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone heterogeneous MIMO training entrypoint."""

import faulthandler
import os
import sys

import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.config import set_experimental_flag

from examples.mimo.training.hetero.args import parse_args
from examples.mimo.training.hetero.distributed import (
    initialize_distributed,
    print_rank_0,
    shutdown_distributed,
)
from examples.mimo.training.hetero.loop import run_train_loop


def main() -> None:
    """Program entrypoint."""
    # Dump every rank's python stack every 120 s. Hands-off diagnostic for
    # hetero MIMO hangs — output goes to each rank's stderr so cog/slurm log
    # capture works without code changes.
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

    args = parse_args()
    if args.enable_experimental:
        set_experimental_flag(True)
    initialize_distributed()
    try:
        run_train_loop(args)
        dist.barrier()
        print_rank_0("Heterogeneous MIMO training completed")
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
