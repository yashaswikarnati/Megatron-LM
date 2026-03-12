# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Utility helpers for mimo models.
"""

import os

import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.validation import StrictHandling


def _resolve_ckpt_dir(ckpt_dir: str) -> str:
    """Resolve a checkpoint path to the actual iteration directory.

    If *ckpt_dir* contains ``latest_checkpointed_iteration.txt``, read it
    and return the corresponding ``iter_NNNNNNN`` subdirectory.  Otherwise
    return *ckpt_dir* unchanged (assumed to already point at an iter dir).
    """
    tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        with open(tracker) as f:
            iteration = int(f.read().strip())
        iter_dir = os.path.join(ckpt_dir, f"iter_{iteration:07d}")
        if not os.path.isdir(iter_dir):
            raise FileNotFoundError(
                f"Checkpoint tracker points to iteration {iteration} but "
                f"{iter_dir} does not exist"
            )
        return iter_dir
    return ckpt_dir


def load_submodule_ckpt(module: torch.nn.Module, ckpt_dir: str):
    """Load *ckpt_dir* into *module* using Megatron distributed-checkpointing."""

    # 1) Ask for tensors using a `module.` prefix so they match checkpoint keys.
    sharded_sd_with_prefix = module.sharded_state_dict(prefix="module.")

    # Remove fp8 extra_state tensors – they may not exist in older checkpoints.
    for k in list(sharded_sd_with_prefix.keys()):
        if "extra_state" in k:
            del sharded_sd_with_prefix[k]

    # 2) Wrap it under a root key just as in user snippet; this becomes the state
    #    dict returned by `load` so we can easily strip the prefix afterwards.
    wrapper_sd = dict(state_dict=sharded_sd_with_prefix)
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.LOG_UNEXPECTED,
    )
    # 3) Remove the prefix and push into the module.
    cleaned = {k.removeprefix("module."): v for k, v in loaded["state_dict"].items()}

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load_state_dict had unexpected mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )


# Submodules of the MIMO model to load from a non-MIMO Nemotron VLM checkpoint.
# Each entry maps (MIMO submodule accessor, checkpoint prefix).
_NEMOTRON_SUBMODULE_MAP = [
    ("language_model", "language_model."),
    ("modality_submodules.images.encoders.radio_encoder.radio_model", "vision_model."),
    ("modality_submodules.images.input_projections.0", "vision_projection."),
]


def _get_nested_attr(obj, dotted_path: str):
    """Resolve ``a.b.c`` style attribute paths, supporting integer indices for ModuleList."""
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _load_submodule_from_ckpt(
    module: torch.nn.Module,
    ckpt_dir: str,
    ckpt_prefix: str,
    ignore_missing_extra_state: bool = True,
):
    """Load a single submodule from a checkpoint directory with a given key prefix.

    Each submodule produces its own ``sharded_state_dict`` with correct TP
    metadata (because the recursion goes through MegatronModule children
    that override ``sharded_state_dict``), avoiding the flat-state-dict
    problem that arises when calling ``sharded_state_dict`` on the whole
    MimoModel (whose ``nn.ModuleDict`` children lack that method).
    """
    sharded_sd = module.sharded_state_dict(prefix=ckpt_prefix)

    for k in list(sharded_sd.keys()):
        if "extra_state" in k:
            del sharded_sd[k]

    wrapper_sd = dict(state_dict=sharded_sd)
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.LOG_UNEXPECTED,
    )

    # Strip the checkpoint prefix so keys match the submodule's state_dict.
    cleaned = {k.removeprefix(ckpt_prefix): v for k, v in loaded["state_dict"].items()}

    # Validate shapes before loading to catch silent mismatches.
    model_sd = module.state_dict()
    shape_mismatches = []
    for k, v in cleaned.items():
        if k in model_sd and isinstance(v, torch.Tensor) and isinstance(model_sd[k], torch.Tensor):
            if v.shape != model_sd[k].shape:
                shape_mismatches.append(
                    f"  {k}: ckpt={list(v.shape)} vs model={list(model_sd[k].shape)}"
                )
    if shape_mismatches:
        raise RuntimeError(
            f"Shape mismatches loading prefix '{ckpt_prefix}':\n"
            + "\n".join(shape_mismatches)
        )

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load mismatch for prefix '{ckpt_prefix}'. "
            f"Missing: {missing}, Unexpected: {unexpected}"
        )

    # Report how many parameters were loaded.
    n_loaded = sum(1 for k in cleaned if k in model_sd and "extra_state" not in k)
    n_total = sum(1 for k in model_sd if "extra_state" not in k)
    print(f"    [{ckpt_prefix}] Loaded {n_loaded}/{n_total} parameter tensors.")


def load_nemotron_vlm_ckpt(
    mimo_model: torch.nn.Module,
    ckpt_dir: str,
    skip_projection: bool = False,
):
    """Load a non-MIMO Nemotron VLM checkpoint into *mimo_model*.

    The checkpoint uses flat prefixes (``vision_model.*``,
    ``vision_projection.*``, ``language_model.*``) while the MIMO model
    nests vision components under ``modality_submodules.*``.

    We load each submodule (language model, vision encoder, projection)
    separately so that each one produces correctly TP-sharded
    ``ShardedTensor`` metadata via its own ``sharded_state_dict``.

    If *skip_projection* is True, the vision_projection weights are not
    loaded (projection stays randomly initialized). Use for adapter-only
    training where the projection is trained from scratch.

    *ckpt_dir* may be either the parent ``checkpoints/`` directory
    (containing ``latest_checkpointed_iteration.txt``) or a specific
    ``iter_NNNNNNN`` dir.
    """
    ckpt_dir = _resolve_ckpt_dir(ckpt_dir)
    print(f"  Resolved checkpoint dir: {ckpt_dir}")

    for attr_path, ckpt_prefix in _NEMOTRON_SUBMODULE_MAP:
        if skip_projection and ckpt_prefix == "vision_projection.":
            print(f"  Skipping '{ckpt_prefix}*' (--skip-projection-checkpoint)")
            continue
        submodule = _get_nested_attr(mimo_model, attr_path)
        _load_submodule_from_ckpt(submodule, ckpt_dir, ckpt_prefix)
        print(f"  Loaded '{ckpt_prefix}*' → {attr_path}")
