# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Role-aware external DataLoaders for heterogeneous MIMO mock training."""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

from examples.mimo.data.mock import get_mock_vlm_dataloader
from examples.mimo.training.topology import HeteroTopology
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_pg_rank

_ENCODER_SEED_OFFSET = 10_000
_LANGUAGE_SEED_OFFSET = 20_000
_SPLIT_SEED_OFFSETS = (0, 100_000, 200_000)


def build_train_valid_test_data_loaders(
    args: argparse.Namespace, topology: HeteroTopology
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build independent mock DataLoaders for the data-consuming rank role."""
    if getattr(args, "dataset_provider", "mock") != "mock":
        raise ValueError(f"unsupported dataset provider: {args.dataset_provider}")

    encoder_name = _encoder_name(topology)
    if encoder_name is not None and (args.micro_batch_size * args.llm_dp) % args.encoder_dp:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")

    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_pgc = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    language_needs_data = language_grid.is_current_rank_in_grid() and (
        is_pp_first_stage(language_pgc.pp) or is_pp_last_stage(language_pgc.pp)
    )

    encoder_needs_data = False
    encoder_pgc = None
    if encoder_name is not None:
        encoder_pgc = topology.module_pgs[encoder_name]
        rank_in_encoder = topology.grids[encoder_name].is_current_rank_in_grid()
        if rank_in_encoder and not getattr(args, "disable_vision_class_token", False):
            raise ValueError("RADIO mock data requires --disable-vision-class-token")
        encoder_needs_data = rank_in_encoder and is_pp_first_stage(encoder_pgc.pp)

    if encoder_needs_data and language_needs_data:
        raise ValueError("the external DataLoader adapter requires non-colocated module grids")
    if encoder_needs_data:
        encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp
        return _build_split_loaders(
            args,
            batch_size=encoder_mbs,
            pg_collection=encoder_pgc,
            module_seed_offset=_ENCODER_SEED_OFFSET,
            encoder_name=encoder_name,
        )
    if language_needs_data:
        return _build_split_loaders(
            args,
            batch_size=args.micro_batch_size,
            pg_collection=language_pgc,
            module_seed_offset=_LANGUAGE_SEED_OFFSET,
            encoder_name=None,
        )
    return (None, None, None)


def _build_split_loaders(
    args: argparse.Namespace,
    *,
    batch_size: int,
    pg_collection,
    module_seed_offset: int,
    encoder_name: Optional[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build split-local datasets with deterministic module/DP/split seeds."""
    base_seed = args.seed + module_seed_offset + get_pg_rank(pg_collection.dp)
    common = _mock_loader_kwargs(args, encoder_name)
    return tuple(
        get_mock_vlm_dataloader(
            batch_size=batch_size,
            dataset_size=getattr(args, "mock_dataset_size", 10_000),
            num_workers=0,
            shuffle=False,
            seed=base_seed + split_offset,
            **common,
        )
        for split_offset in _SPLIT_SEED_OFFSETS
    )


def _mock_loader_kwargs(args: argparse.Namespace, encoder_name: Optional[str]) -> dict:
    """Translate parsed training arguments to the reusable mock loader."""
    seq_len = args.seq_length
    dtype = getattr(args, "params_dtype", None)
    if dtype is None:
        dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float32

    image_size = getattr(args, "image_size", 224)
    img_h = getattr(args, "img_h", image_size)
    img_w = getattr(args, "img_w", image_size)
    patch_dim = getattr(args, "patch_dim", 16)
    num_image_tiles = getattr(args, "num_image_tiles", 1)
    pixel_shuffle = bool(getattr(args, "pixel_shuffle", False))
    dynamic_resolution = bool(getattr(args, "dynamic_resolution", False))
    image_seq_length = getattr(args, "image_seq_length", None)
    if image_seq_length is None:
        image_seq_length = (
            seq_len // 2
            if dynamic_resolution
            else _fixed_image_seq_length(
                img_h, img_w, patch_dim, num_image_tiles, pixel_shuffle
            )
        )

    return {
        "image_size": image_size,
        "seq_len": seq_len,
        "image_seq_length": image_seq_length,
        "vocab_size": args.vocab_size,
        "pad_token_id": getattr(args, "pad_token_id", 0),
        "image_token_id": args.image_token_id,
        "modality_module_name": encoder_name or "images",
        "encoder_name": encoder_name or "clip_encoder",
        "include_modality_inputs": encoder_name is not None,
        "dtype": dtype,
        "dynamic_resolution": dynamic_resolution,
        "patch_dim": patch_dim,
        "img_h": img_h,
        "img_w": img_w,
        "pixel_shuffle": pixel_shuffle,
        "num_image_tiles": num_image_tiles,
        "validate_image_token_count": encoder_name is not None,
    }


def _fixed_image_seq_length(
    img_h: int, img_w: int, patch_dim: int, num_image_tiles: int, pixel_shuffle: bool
) -> int:
    """Derive fixed-resolution RADIO output tokens from image geometry."""
    if patch_dim <= 0 or img_h % patch_dim or img_w % patch_dim:
        raise ValueError("fixed RADIO image dimensions must be divisible by patch_dim")
    patches = num_image_tiles * (img_h // patch_dim) * (img_w // patch_dim)
    return patches // 4 if pixel_shuffle else patches


def _encoder_name(topology: HeteroTopology) -> Optional[str]:
    """Return the example's optional single encoder module name."""
    names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    if len(names) > 1:
        raise ValueError("the heterogeneous MIMO example supports at most one encoder module")
    return names[0] if names else None
