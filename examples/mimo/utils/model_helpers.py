# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers to load non-MIMO Nemotron VLM checkpoints into hetero MIMO models.

Vision and language live on disjoint rank grids in hetero, so encoder ranks
load only ``vision_model.*`` / ``vision_projection.*`` and LLM ranks load
only ``language_model.*``.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from examples.mimo.utils.hetero import is_process_group_member
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.validation import StrictHandling


def _resolve_ckpt_dir(ckpt_dir: str) -> str:
    """Resolve a checkpoint path to the actual iteration directory.

    If ``ckpt_dir`` contains ``latest_checkpointed_iteration.txt``, read it
    and return the corresponding ``iter_NNNNNNN`` subdirectory. Otherwise
    return ``ckpt_dir`` unchanged (assumed to already point at an iter dir).
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
    """Load ``ckpt_dir`` into ``module`` using a flat ``module.*`` prefix.

    Retained from the original POC; not used by the hetero loader below.
    Kept so older inference scripts continue to import successfully.
    """
    sharded_sd_with_prefix = module.sharded_state_dict(prefix="module.")

    for k in list(sharded_sd_with_prefix.keys()):
        if "extra_state" in k:
            del sharded_sd_with_prefix[k]

    wrapper_sd = {"state_dict": sharded_sd_with_prefix}
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.LOG_UNEXPECTED,
    )
    cleaned = {k.removeprefix("module."): v for k, v in loaded["state_dict"].items()}

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load_state_dict had unexpected mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )


def _load_submodule_from_ckpt(
    module: torch.nn.Module,
    ckpt_dir: str,
    ckpt_prefix: str,
    dp_cp_group=None,
) -> tuple[int, int]:
    """Load one submodule from ``ckpt_dir`` under ``ckpt_prefix``. Returns
    ``(n_loaded, n_total)`` parameter-tensor counts. ``dp_cp_group`` must be
    passed when ``parallel_state`` isn't initialized."""
    metadata = {"dp_cp_group": dp_cp_group} if dp_cp_group is not None else None
    sharded_sd = module.sharded_state_dict(prefix=ckpt_prefix, metadata=metadata)

    for k in list(sharded_sd.keys()):
        if "extra_state" in k:
            del sharded_sd[k]

    wrapper_sd = {"state_dict": sharded_sd}
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.LOG_UNEXPECTED,
    )

    cleaned = {k.removeprefix(ckpt_prefix): v for k, v in loaded["state_dict"].items()}

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
            f"Shape mismatches loading prefix '{ckpt_prefix}':\n" + "\n".join(shape_mismatches)
        )

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load mismatch for prefix '{ckpt_prefix}'. "
            f"Missing: {missing}, Unexpected: {unexpected}"
        )

    n_loaded = sum(1 for k in cleaned if k in model_sd and "extra_state" not in k)
    n_total = sum(1 for k in model_sd if "extra_state" not in k)
    return n_loaded, n_total


def load_nemotron_vlm_ckpt_hetero(
    mimo_model,
    ckpt_dir: str,
    encoder_name: str,
    radio_encoder_key: str = "radio_encoder",
    *,
    has_encoder: bool,
    has_language: bool,
    language_dp_cp_group=None,
    encoder_dp_cp_group=None,
    skip_projection: bool = False,
) -> None:
    """Load a flat ``vision_model.* / vision_projection.* / language_model.*``
    ckpt into a hetero MIMO model. Each rank loads only the submodules its
    grid owns: encoder ranks load vision_model + vision_projection; LLM
    ranks load language_model."""
    ckpt_dir = _resolve_ckpt_dir(ckpt_dir)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[load-nemotron-vlm-ckpt] resolved iter_dir: {ckpt_dir}", flush=True)

    # Build a SINGLE combined sharded state dict containing all submodules
    # this rank participates in. We then issue ONE `dist_checkpointing.load`
    # call across all world ranks. This is required because mcore's
    # `dist_checkpointing.load` internally does world-collectives — splitting
    # the load into separate calls per grid deadlocks (encoder ranks finish
    # and hit a world barrier while LLM ranks are still inside the load).
    combined_sd: dict[str, Any] = {}
    targets: list[tuple[torch.nn.Module, str, str]] = []

    def _drill_through_ddp(mod):
        """Unwrap DDP / Float16Module wrappers so we hit the raw nn.Module."""
        try:
            from megatron.core.distributed import DistributedDataParallel as _DDP
        except Exception:  # pylint: disable=broad-except
            _DDP = ()
        seen = set()
        while True:
            inner = getattr(mod, "module", None)
            if inner is None or id(inner) in seen:
                return mod
            seen.add(id(inner))
            mod = inner

    if has_language:
        if not hasattr(mimo_model, "language_model") or mimo_model.language_model is None:
            raise RuntimeError(
                "has_language=True but mimo_model.language_model is None on this rank."
            )
        if language_dp_cp_group is None:
            raise RuntimeError(
                "has_language=True requires language_dp_cp_group (our hetero loop does "
                "not initialize megatron.core.parallel_state)."
            )
        # After wrap_active_modules_with_ddp, mimo_model.language_model is a
        # DistributedDataParallel wrapper; drill through to call
        # sharded_state_dict on the raw model.
        lm_raw = _drill_through_ddp(mimo_model.language_model)
        lm_sd = lm_raw.sharded_state_dict(
            prefix="language_model.", metadata={"dp_cp_group": language_dp_cp_group}
        )
        for k in list(lm_sd.keys()):
            if "extra_state" in k:
                del lm_sd[k]
        combined_sd.update(lm_sd)
        targets.append((lm_raw, "language_model.", "language_model"))

    if has_encoder:
        submodules = getattr(mimo_model, "modality_submodules", None)
        if submodules is None or encoder_name not in submodules:
            raise RuntimeError(
                f"has_encoder=True but mimo_model.modality_submodules[{encoder_name!r}] missing."
            )
        # Same DDP-unwrap dance for the encoder-side submodule.
        vision_submodule = _drill_through_ddp(submodules[encoder_name])
        encoders = getattr(vision_submodule, "encoders", None)
        if encoders is None or radio_encoder_key not in encoders:
            raise RuntimeError(f"vision submodule missing encoders[{radio_encoder_key!r}].")
        radio_wrapper = encoders[radio_encoder_key]
        radio_model = getattr(radio_wrapper, "radio_model", None)
        if radio_model is None:
            raise RuntimeError(
                f"encoders[{radio_encoder_key!r}].radio_model is None on this rank."
            )
        if encoder_dp_cp_group is None:
            raise RuntimeError(
                "has_encoder=True requires encoder_dp_cp_group (our hetero loop does "
                "not initialize megatron.core.parallel_state)."
            )
        radio_sd = radio_model.sharded_state_dict(
            prefix="vision_model.", metadata={"dp_cp_group": encoder_dp_cp_group}
        )
        for k in list(radio_sd.keys()):
            if "extra_state" in k:
                del radio_sd[k]
        combined_sd.update(radio_sd)
        targets.append(
            (radio_model, "vision_model.", f"encoders.{radio_encoder_key}.radio_model")
        )

        if not skip_projection:
            projectors = getattr(vision_submodule, "input_projections", None)
            if projectors is None or len(projectors) == 0:
                raise RuntimeError("vision submodule has no input_projections to load into.")
            proj_sd = projectors[0].sharded_state_dict(
                prefix="vision_projection.", metadata={"dp_cp_group": encoder_dp_cp_group}
            )
            for k in list(proj_sd.keys()):
                if "extra_state" in k:
                    del proj_sd[k]
            combined_sd.update(proj_sd)
            targets.append((projectors[0], "vision_projection.", "input_projections[0]"))

    # Even ranks with no local targets (shouldn't happen in non-colocated
    # hetero, but defensive) participate in the load so the world-collective
    # has the full barrier population.
    wrapper_sd = {"state_dict": combined_sd}
    if rank == 0:
        print(
            f"[load-nemotron-vlm-ckpt] rank=0 combined_sd has {len(combined_sd)} keys "
            f"across {len(targets)} target submodules",
            flush=True,
        )

    # RETURN_ALL semantics (megatron/core/dist_checkpointing/validation.py:267-274):
    #   third return = keys we requested but ckpt does NOT have
    #                  (DANGEROUS — those tensors would silently keep their
    #                  random-init values; PyTorch load_state_dict calls this
    #                  set "missing" but mcore returns it as "unexpected").
    # Weaker modes (LOG_UNEXPECTED, ASSUME_OK_UNEXPECTED) skip this check
    # entirely. Raise loudly on any non-extra_state mismatch.
    loaded, _ckpt_only_keys, request_only_keys = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
        strict=StrictHandling.RETURN_ALL,
    )
    missing_in_ckpt = sorted(k for k in request_only_keys if "extra_state" not in k)
    if missing_in_ckpt:
        raise RuntimeError(
            f"checkpoint is missing {len(missing_in_ckpt)} keys the model requested; "
            f"these would silently keep random-init values. "
            f"First 30: {missing_in_ckpt[:30]}"
        )
    loaded_sd = loaded.get("state_dict", {})

    # Apply loaded tensors back to each submodule by stripping its checkpoint prefix.
    for module, prefix, label in targets:
        cleaned = {
            k.removeprefix(prefix): v for k, v in loaded_sd.items() if k.startswith(prefix)
        }
        model_sd = module.state_dict()
        shape_mismatches = []
        for k, v in cleaned.items():
            if (
                k in model_sd
                and isinstance(v, torch.Tensor)
                and isinstance(model_sd[k], torch.Tensor)
            ):
                if v.shape != model_sd[k].shape:
                    shape_mismatches.append(
                        f"  {k}: ckpt={list(v.shape)} vs model={list(model_sd[k].shape)}"
                    )
        if shape_mismatches:
            raise RuntimeError(
                f"Shape mismatches for prefix '{prefix}':\n" + "\n".join(shape_mismatches)
            )

        incompatible = module.load_state_dict(cleaned, strict=False)
        unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
        missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
        if unexpected or missing:
            raise RuntimeError(
                f"load mismatch for prefix '{prefix}'. Missing: {missing}, "
                f"Unexpected: {unexpected}"
            )

        n_loaded = sum(1 for k in cleaned if k in model_sd and "extra_state" not in k)
        n_total = sum(1 for k in model_sd if "extra_state" not in k)
        if rank == 0 or has_encoder:
            print(
                f"[load-nemotron-vlm-ckpt] rank={rank} '{prefix}*' -> {label}"
                f" ({n_loaded}/{n_total} param tensors)",
                flush=True,
            )


def load_and_refresh_nemotron_checkpoint(model, optimizer, topology, args) -> None:
    """Load a Nemotron-format ckpt into a hetero MIMO model and resync the
    optimizer's FP32 main params. DistributedOptimizer is built before this
    custom load runs, so its shards otherwise hold the model-provider init
    weights; ``reload_model_params`` syncs them to the loaded weights."""
    from examples.mimo.model_providers.nemotron_moe_vlm import NEMOTRON_VISION_ENCODER_KEY

    if args.load:
        raise ValueError(
            "--load and --load-nemotron-checkpoint are mutually exclusive; pick one"
        )

    rank_in_llm = topology.language_pg is not None and is_process_group_member(
        getattr(topology.language_pg, "dp_cp", None)
    )
    rank_in_enc = topology.vision_pg is not None and is_process_group_member(
        getattr(topology.vision_pg, "dp_cp", None)
    )
    has_encoder = (
        rank_in_enc
        and topology.encoder_name in getattr(model, "modality_submodules", {})
        and model.modality_submodules[topology.encoder_name] is not None
    )
    has_language = rank_in_llm and getattr(model, "language_model", None) is not None

    load_nemotron_vlm_ckpt_hetero(
        model,
        args.load_nemotron_checkpoint,
        encoder_name=topology.encoder_name,
        radio_encoder_key=NEMOTRON_VISION_ENCODER_KEY,
        has_encoder=has_encoder,
        has_language=has_language,
        language_dp_cp_group=(
            getattr(topology.language_pg, "dp_cp", None) if has_language else None
        ),
        encoder_dp_cp_group=(
            getattr(topology.vision_pg, "dp_cp", None) if has_encoder else None
        ),
        skip_projection=False,
    )
    optimizer.reload_model_params()
