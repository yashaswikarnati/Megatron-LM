# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed checkpoint save/load for the heterogeneous MIMO training loop.

Wraps `megatron.core.dist_checkpointing` so the standalone hetero loop can
persist MimoModel + MimoOptimizer + LR scheduler state without depending on
`megatron.training.checkpointing` (which assumes the parallel_state singleton).

Stays intentionally close to the layout that `megatron/training/checkpointing.py`
produces so existing inspection tooling keeps working:

    <save>/
      latest_checkpointed_iteration.txt
      iter_0000010/
        common.pt              # args, checkpoint_version, iteration, scheduler
        metadata.json          # backend + version + sharding-type content_metadata
        ...torch_dist shards...
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from examples.mimo.training.hetero.distributed import print_rank_0
from examples.mimo.training.hetero.topology import HeteroTopology, is_rank_in_grid
from examples.mimo.utils.hetero import is_process_group_member
from megatron.core import dist_checkpointing, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.utils import _clean_metadata_for_serialization
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import MimoOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

_TRACKER_FILE = "latest_checkpointed_iteration.txt"
_CHECKPOINT_VERSION = 3.0
_VISION_DCP_PREFIX = "model.vision_model."


def _iter_directory(root: str, iteration: int) -> str:
    return os.path.join(root, f"iter_{iteration:07d}")


def _tracker_path(root: str) -> str:
    return os.path.join(root, _TRACKER_FILE)


def _build_optim_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """Optimizer-side metadata controlling DistributedOptimizer sharding format."""
    metadata: Dict[str, Any] = {"chained_optim_avoid_prefix": True, "singleton_local_shards": False}
    if args.dist_ckpt_optim_fully_reshardable:
        metadata["distrib_optim_sharding_type"] = "fully_reshardable"
    else:
        metadata["distrib_optim_sharding_type"] = "dp_reshardable"
    return metadata


def _pg_rank_size(pg: Optional[dist.ProcessGroup]) -> tuple[int, int]:
    """Return (rank, size) for a process group, or (0, 1) when this rank isn't a member."""
    if pg is not None and is_process_group_member(pg):
        return pg.rank(), pg.size()
    return 0, 1


def _collect_rng_state(topology: HeteroTopology) -> Optional[Dict[str, ShardedObject]]:
    """Collect this rank's Python/NumPy/Torch/CUDA RNG state, sharded by (pp, tp).

    Mirrors `megatron.training.checkpointing.get_rng_state` but reads pp/tp/dp
    groups from the active hetero branch's pg_collection instead of parallel_state.
    The returned dict has a single per-branch entry: encoder ranks publish
    ``mimo.<encoder_name>.rng_state`` and LLM ranks publish ``mimo.language.rng_state``
    so the two branches don't collide on the same ShardedObject key.
    Returns None when the rank is not in any branch (should not happen in
    non-colocated layouts, but defensive).
    """
    if is_rank_in_grid(topology.llm_grid):
        pg = topology.language_pg
        branch_name = "language"
    elif is_rank_in_grid(topology.encoder_grid):
        pg = topology.vision_pg
        branch_name = topology.encoder_name
    else:
        return None

    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    pp_rank, pp_size = _pg_rank_size(getattr(pg, "pp", None))
    tp_rank, tp_size = _pg_rank_size(getattr(pg, "tp", None))
    dp_rank, _ = _pg_rank_size(getattr(pg, "dp", None))

    key = f"mimo.{branch_name}.rng_state"
    # One RNG snapshot per (pp, tp) shard; replicated across DP within that shard.
    return {
        key: ShardedObject(
            key, [rng_state], (pp_size, tp_size), (pp_rank, tp_rank), replica_id=dp_rank
        )
    }


def _restore_rng_state(rng_state_obj) -> None:
    """Apply RNG state previously captured by `_collect_rng_state`."""
    if rng_state_obj is None:
        return
    rng_state_list = rng_state_obj
    if isinstance(rng_state_list, list) and rng_state_list and isinstance(rng_state_list[0], dict):
        rng_state = rng_state_list[0]
    elif isinstance(rng_state_list, dict):
        rng_state = rng_state_list
    else:
        # Unknown payload shape — skip silently rather than crash the run.
        return

    random.setstate(rng_state["random_rng_state"])
    np.random.set_state(rng_state["np_rng_state"])
    torch.set_rng_state(rng_state["torch_rng_state"])
    torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
    if rng_state.get("rng_tracker_states"):
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])


def _assemble_state_dict(
    model: MimoModel,
    optimizer: Optional[MimoOptimizer],
    opt_param_scheduler: Optional[OptimizerParamScheduler],
    iteration: Optional[int],
    args: argparse.Namespace,
    topology: HeteroTopology,
    include_optimizer: bool,
    include_scheduler: bool,
    include_rng: bool,
    include_args: bool,
    is_loading: bool,
) -> Dict[str, Any]:
    """Build the (sharded) state dict consumed by `dist_checkpointing.save`/`load`.

    The MimoModel and MimoOptimizer already inject per-submodule `dp_cp_group`
    from each module's pg_collection, so no global dp_cp_group needs to be set.
    """
    state_dict: Dict[str, Any] = {"checkpoint_version": _CHECKPOINT_VERSION}
    if iteration is not None:
        state_dict["iteration"] = iteration

    if include_args:
        # Stored as a plain dict (not vars(args) directly) so torch.save can pickle it
        # via the common-state path. argparse.Namespace round-trips fine; using a dict
        # gives us better forward-compat across argparse internals.
        state_dict["args"] = dict(vars(args))

    state_dict["model"] = model.sharded_state_dict()

    if include_optimizer and optimizer is not None and not optimizer.is_stub_optimizer:
        optim_kwargs = {"metadata": _build_optim_metadata(args)}
        state_dict["optimizer"] = optimizer.sharded_state_dict(
            state_dict, is_loading=is_loading, **optim_kwargs
        )

    if include_scheduler and opt_param_scheduler is not None:
        state_dict["opt_param_scheduler"] = opt_param_scheduler.state_dict()

    if include_rng:
        rng = _collect_rng_state(topology)
        if rng is not None:
            # The dict contains exactly one entry; merge it at top level so each
            # branch's ShardedObject lives under its own key (no cross-branch collision).
            state_dict.update(rng)

    return state_dict


def save_checkpoint(
    iteration: int,
    model: MimoModel,
    optimizer: Optional[MimoOptimizer],
    opt_param_scheduler: Optional[OptimizerParamScheduler],
    args: argparse.Namespace,
    topology: HeteroTopology,
) -> None:
    """Save a hetero MIMO checkpoint at iteration `iteration` under `args.save`."""
    if not args.save:
        return

    save_root = args.save
    target_dir = _iter_directory(save_root, iteration)

    # mkdir on every rank with exist_ok=True so a single rank's mkdir failure
    # doesn't strand peers behind a barrier.
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    print_rank_0(f"saving hetero checkpoint at iteration {iteration} to {target_dir}")

    state_dict = _assemble_state_dict(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
        args=args,
        topology=topology,
        include_optimizer=not args.no_save_optim,
        include_scheduler=True,
        include_rng=not args.no_save_rng,
        include_args=True,
        is_loading=False,
    )

    content_metadata = _clean_metadata_for_serialization(_build_optim_metadata(args))

    dist_checkpointing.save(state_dict, target_dir, content_metadata=content_metadata)

    if dist.get_rank() == 0:
        tracker_tmp = _tracker_path(save_root) + ".tmp"
        with open(tracker_tmp, "w") as f:
            f.write(str(iteration))
        os.replace(tracker_tmp, _tracker_path(save_root))
    dist.barrier()
    print_rank_0(f"hetero checkpoint at iteration {iteration} saved")


def _read_tracker(load_root: str) -> Optional[int]:
    """Return the iteration recorded in the tracker file (max-reduced across ranks).

    Mirrors `megatron.training.checkpointing.read_metadata`: each rank reads the
    local file and we agree on the largest value. None if no checkpoint exists
    at this path on any rank.
    """
    tracker = _tracker_path(load_root)
    local_iter = -1
    if os.path.isfile(tracker):
        with open(tracker) as f:
            contents = f.read().strip()
        if contents:
            try:
                local_iter = int(contents)
            except ValueError as e:
                raise RuntimeError(f"Tracker file {tracker} is corrupted: {contents!r}") from e

    if dist.is_available() and dist.is_initialized():
        iters_cuda = torch.tensor([local_iter], dtype=torch.long, device="cuda")
        dist.all_reduce(iters_cuda, op=dist.ReduceOp.MAX)
        max_iter = int(iters_cuda[0].item())
    else:
        max_iter = local_iter

    return max_iter if max_iter >= 0 else None


def load_checkpoint(
    model: MimoModel,
    optimizer: Optional[MimoOptimizer],
    opt_param_scheduler: Optional[OptimizerParamScheduler],
    args: argparse.Namespace,
    topology: HeteroTopology,
) -> int:
    """Restore a hetero MIMO checkpoint from `args.load` and return the resume iteration.

    Returns 0 if `--load` is not set or no completed checkpoint exists at that path.
    With `--finetune`, model state is loaded but iteration/optimizer/scheduler/rng
    are reset.
    """
    if not args.load:
        return 0

    load_root = args.load
    iteration = _read_tracker(load_root)
    if iteration is None:
        print_rank_0(f"no checkpoint found at {load_root}; starting from iteration 0")
        return 0

    source_dir = _iter_directory(load_root, iteration)
    if not os.path.isdir(source_dir):
        raise RuntimeError(
            f"Tracker at {load_root} points to iteration {iteration} but "
            f"{source_dir} is missing"
        )

    is_finetune = bool(args.finetune)
    include_optimizer = (not args.no_load_optim) and not is_finetune
    include_scheduler = (not args.no_load_scheduler) and not is_finetune
    include_rng = (not args.no_load_rng) and not is_finetune

    print_rank_0(
        f"loading hetero checkpoint from {source_dir}"
        f" (optimizer={'yes' if include_optimizer else 'no'},"
        f" scheduler={'yes' if include_scheduler else 'no'},"
        f" rng={'yes' if include_rng else 'no'},"
        f" finetune={is_finetune})"
    )

    sharded_state_dict = _assemble_state_dict(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
        args=args,
        topology=topology,
        include_optimizer=include_optimizer,
        include_scheduler=include_scheduler,
        include_rng=include_rng,
        include_args=False,  # args round-trips via common.pt, not via the request dict
        is_loading=True,
    )

    loaded = dist_checkpointing.load(sharded_state_dict, source_dir)

    model.load_state_dict(loaded["model"], strict=True)

    if include_optimizer and optimizer is not None and not optimizer.is_stub_optimizer:
        optimizer.load_state_dict(loaded["optimizer"])

    if include_scheduler and opt_param_scheduler is not None and "opt_param_scheduler" in loaded:
        opt_param_scheduler.load_state_dict(loaded["opt_param_scheduler"])

    if include_rng:
        # Find this rank's per-branch rng key in the loaded dict.
        for key, value in loaded.items():
            if key.startswith("mimo.") and key.endswith(".rng_state"):
                _restore_rng_state(value)
                break

    resume_iter = 0 if is_finetune else int(loaded.get("iteration", iteration))
    print_rank_0(f"resuming hetero training at iteration {resume_iter}")
    return resume_iter


def _tp_slice(tensor: torch.Tensor, param_shape, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Slice a full (TP=1) tensor to this rank's TP shard.

    Mirrors the helper in Sanjeev's `_load_vision_from_checkpoint`:
    handles column-parallel (first-dim split) and row-parallel (second-dim split).
    Returns the tensor unchanged when it already matches the param shape.
    """
    if tp_size == 1 or tuple(tensor.shape) == tuple(param_shape):
        return tensor
    if tensor.shape[0] != param_shape[0]:
        start = tp_rank * param_shape[0]
        return tensor[start : start + param_shape[0], ...]
    if len(tensor.shape) > 1 and tensor.shape[1] != param_shape[1]:
        start = tp_rank * param_shape[1]
        return tensor[:, start : start + param_shape[1]]
    return tensor


def _resolve_vision_dcp_dir(ckpt_dir: str) -> str:
    """Resolve a flat-DCP or `iter_NNNNNNN/` directory under `ckpt_dir`."""
    tracker = os.path.join(ckpt_dir, _TRACKER_FILE)
    if os.path.isfile(tracker):
        with open(tracker) as f:
            iteration = int(f.read().strip())
        return _iter_directory(ckpt_dir, iteration)
    return ckpt_dir


def _vision_submodule(model: MimoModel, encoder_name: str):
    """Return the modality-submodules container for the encoder branch, or None.

    Returns None when this rank is not part of the encoder branch (e.g. LLM-only
    ranks where the modality submodule was never instantiated).
    """
    submodules = getattr(model, "modality_submodules", None)
    if submodules is None or encoder_name not in submodules:
        return None
    return submodules[encoder_name]


def _radio_target_modules(vision_submodule, radio_encoder_key: str):
    """Return (radio_model, projector_or_None) for the RADIO encoder branch.

    `vision_submodule` is the `VisionModalitySubmodules` instance (or a DDP-wrapped
    variant). The RADIO encoder lives at `.encoders[radio_encoder_key].radio_model`
    and the (optional) projector at `.input_projections[0]`.
    """
    inner = getattr(vision_submodule, "module", vision_submodule)  # unwrap DDP
    encoders = getattr(inner, "encoders", None)
    if encoders is None or radio_encoder_key not in encoders:
        return None, None
    radio_wrapper = encoders[radio_encoder_key]
    radio_model = getattr(radio_wrapper, "radio_model", None)
    projector = None
    projections = getattr(inner, "input_projections", None)
    if projections is not None and len(projections) > 0:
        projector = projections[0]
    return radio_model, projector


def load_vision_from_checkpoint(
    model: MimoModel,
    args: argparse.Namespace,
    topology: HeteroTopology,
) -> None:
    """Load RADIO encoder (and best-effort projector) weights from a Bridge DCP.

    Active only on encoder-grid ranks. Reads keys with prefix
    `model.vision_model.` from `args.load_vision_from`, TP-slices each tensor
    against the per-rank parameter shape, and copies into
    `vision_submodule.encoders[<radio_key>].radio_model.<rel>`. Tensors that
    also match `vision_submodule.input_projections[0].<rel>` are copied into
    the projector; when `--allow-missing-vision-projection-checkpoint` is set,
    projector keys absent from the DCP are skipped silently.

    No-op on LLM-only ranks and on ranks whose vision pg is not a real member.
    """
    if not args.load_vision_from:
        return

    # Encoder-only: bail on LLM-only ranks and on ranks outside the encoder grid.
    if topology.encoder_grid is None or not is_rank_in_grid(topology.encoder_grid):
        return

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    encoder_name = topology.encoder_name
    radio_encoder_key = getattr(args, "vision_encoder_key", "radio_encoder")
    vision_submodule = _vision_submodule(model, encoder_name)
    if vision_submodule is None:
        return
    radio_model, projector = _radio_target_modules(vision_submodule, radio_encoder_key)
    if radio_model is None:
        return

    # Resolve `iter_NNNNNNN/` if a tracker exists; else treat the path as a flat DCP.
    iter_dir = _resolve_vision_dcp_dir(args.load_vision_from)
    print_rank_0(f"[load-vision-from] Loading ViT via DCP from {iter_dir}")

    reader = FileSystemReader(iter_dir)
    ckpt_metadata = reader.read_metadata().state_dict_metadata

    load_sd: Dict[str, torch.Tensor] = {
        k: torch.empty(meta.size, dtype=meta.properties.dtype)
        for k, meta in ckpt_metadata.items()
        if k.startswith(_VISION_DCP_PREFIX) and isinstance(meta, TensorStorageMetadata)
    }
    if not load_sd:
        print_rank_0(
            f"[load-vision-from] WARNING: no '{_VISION_DCP_PREFIX}*' keys in "
            f"{iter_dir} — encoder runs with random init"
        )
        return

    dcp.load(load_sd, storage_reader=reader)

    # TP slicing: read rank/size from the encoder pg_collection (vision_pg).
    tp_pg = getattr(topology.vision_pg, "tp", None) if topology.vision_pg is not None else None
    if is_process_group_member(tp_pg):
        tp_rank, tp_size = tp_pg.rank(), tp_pg.size()
    else:
        tp_rank, tp_size = 0, 1

    radio_targets = dict(radio_model.named_parameters())
    radio_targets.update(dict(radio_model.named_buffers()))

    projector_targets: Dict[str, torch.Tensor] = {}
    if projector is not None:
        projector_targets = dict(projector.named_parameters())
        projector_targets.update(dict(projector.named_buffers()))

    loaded = skipped = 0
    allow_missing_proj = bool(args.allow_missing_vision_projection_checkpoint)
    for ckpt_key, tensor in load_sd.items():
        rel_key = ckpt_key[len(_VISION_DCP_PREFIX) :]
        param = radio_targets.get(rel_key)
        if param is None and projector is not None:
            param = projector_targets.get(rel_key)
            if param is None and allow_missing_proj:
                # projector-like key not present in our projector module; skip silently.
                skipped += 1
                continue
        if param is None:
            skipped += 1
            continue
        tensor = _tp_slice(tensor, param.shape, tp_rank, tp_size)
        param.data.copy_(tensor.to(dtype=param.dtype))
        loaded += 1

    print_rank_0(
        f"[load-vision-from] ViT loaded ({loaded}/{len(load_sd)} tensors"
        f"{f', {skipped} skipped' if skipped else ''})"
    )
