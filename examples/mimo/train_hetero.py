# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone heterogeneous MIMO training entrypoint."""

import os
import sys

import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.mimo.training.hetero.args import parse_args
from examples.mimo.training.hetero.distributed import (
    initialize_distributed,
    print_rank_0,
    shutdown_distributed,
)
from examples.mimo.training.hetero.loop import run_train_loop


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    initialize_distributed()
    try:
        run_train_loop(args)
        dist.barrier()
        print_rank_0("Heterogeneous MIMO training completed")
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
