# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# pylint: disable=bad-builtin

"""Compare current MIMO Energon batches with the previous branch provider.

This is a dataloader-only parity check. It instantiates the previous branch's
``MimoMultiModalPackingEncoder`` and the current branch's encoder in the same
process, feeds both through Megatron-Energon with identical loader settings, and
requires exact equality for the emitted batch tensors and packed-sequence
metadata. Defaults favor deterministic sample identity; override workers and
shuffle settings when intentionally stress-testing training-like loader behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from examples.mimo.data import energon_multimodal_provider as current_provider

OLD_PROVIDER_REPO_PATH = "examples/mimo/data/energon_multimodal_provider.py"
OLD_PROVIDER_BUNDLED_PATH = REPO_ROOT / "examples/mimo/vendor/old_energon_multimodal_provider.py"


def parse_args() -> argparse.Namespace:
    """Parse dataloader parity options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--old-provider-path", type=str, default=None)
    parser.add_argument(
        "--old-provider-ref",
        type=str,
        default="origin/feat/nemotron-moe-vlm-mimo",
        help="Git ref used when --old-provider-path is not supplied.",
    )
    parser.add_argument("--image-token", type=str, default="<image>")
    parser.add_argument("--tokenizer-prompt-format", type=str, default="nemotron6-moe")
    parser.add_argument("--image-tag-type", type=str, default="")
    parser.add_argument("--force-system-message", action="store_true")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--packing-buffer-size", type=int, default=128)
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--max-samples-per-sequence", type=int, default=100)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--patch-dim", type=int, default=16)
    parser.add_argument("--class-token-len", type=int, default=8)
    parser.add_argument("--max-num-tiles", type=int, default=12)
    parser.add_argument("--vision-model-type", type=str, default="radio")
    parser.add_argument("--pixel-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-vision-class-token", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-thumbnail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-name", type=str, default="radio_encoder")
    parser.add_argument("--encoder-input-key", type=str, default="x")
    return parser.parse_args()


def main() -> None:
    """Run the parity comparison."""
    args = parse_args()
    set_seed(args.seed)

    old_provider = load_old_provider(args)
    tokenizer = build_tokenizer(args)
    image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
    if image_token_id is None:
        raise RuntimeError(f"Tokenizer did not produce an id for {args.image_token!r}")
    pad_id = int(tokenizer.pad)

    old_loader = build_loader(old_provider, tokenizer, image_token_id, pad_id, args)
    set_seed(args.seed)
    current_loader = build_loader(current_provider, tokenizer, image_token_id, pad_id, args)

    for batch_idx in range(args.num_batches):
        batch_seed = args.seed + batch_idx
        set_seed(batch_seed)
        old_batch = next(old_loader)
        set_seed(batch_seed)
        current_batch = next(current_loader)
        mismatches = compare_values("batch", old_batch, current_batch)
        if mismatches:
            print(f"Batch {batch_idx} mismatch")
            for mismatch in mismatches[:20]:
                print(f"  - {mismatch}")
            print(f"old:     {batch_summary(old_batch)}")
            print(f"current: {batch_summary(current_batch)}")
            raise SystemExit(1)
        print(f"batch {batch_idx}: OK {batch_summary(current_batch)}")

    print(f"Parity OK for {args.num_batches} batches")


def set_seed(seed: int = 12345) -> None:
    """Set process-local RNG state before loader construction."""
    random.seed(seed)
    torch.manual_seed(seed)


def load_old_provider(args: argparse.Namespace) -> ModuleType:
    """Load the previous branch provider from a path or git ref."""
    if args.old_provider_path is not None:
        provider_path = Path(args.old_provider_path)
    else:
        try:
            provider_path = materialize_old_provider_from_git(args.old_provider_ref)
        except RuntimeError:
            if not OLD_PROVIDER_BUNDLED_PATH.is_file():
                raise
            provider_path = OLD_PROVIDER_BUNDLED_PATH
    return import_module_from_path("old_energon_multimodal_provider", provider_path)


def materialize_old_provider_from_git(ref: str) -> Path:
    """Write the provider from a git ref to a temporary importable file."""
    provider_source = git_show(ref, OLD_PROVIDER_REPO_PATH)
    temp_dir = Path(tempfile.mkdtemp(prefix="old_energon_provider_"))
    provider_path = temp_dir / "energon_multimodal_provider.py"
    provider_path.write_text(provider_source)
    return provider_path


def git_show(ref: str, repo_path: str) -> str:
    """Return file content from a git ref, with a local-branch fallback."""
    refs_to_try = [ref]
    if ref.startswith("origin/"):
        refs_to_try.append(ref.removeprefix("origin/"))

    errors = []
    for candidate in refs_to_try:
        command = ["git", "show", f"{candidate}:{repo_path}"]
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        errors.append(result.stderr.strip())
    raise RuntimeError("Unable to load old provider from git:\n" + "\n".join(errors))


def import_module_from_path(name: str, path: Path) -> ModuleType:
    """Import a Python module from an explicit path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_tokenizer(args: argparse.Namespace):
    """Build the Megatron multimodal tokenizer used by both providers."""
    from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
        MegatronMultimodalTokenizer,
    )

    return MegatronMultimodalTokenizer(
        path=args.tokenizer_model,
        prompt_format=args.tokenizer_prompt_format,
        special_tokens=[args.image_token],
        image_tag_type=args.image_tag_type,
        force_system_message=args.force_system_message,
    )


def build_loader(
    provider: ModuleType, tokenizer, image_token_id: int, pad_id: int, args: argparse.Namespace
):
    """Build one Energon dataloader using a provider module's encoder."""
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    encoder = build_encoder(provider, tokenizer, image_token_id, pad_id, args)
    worker_config = WorkerConfig(
        rank=args.dp_rank,
        world_size=args.dp_world_size,
        num_workers=args.num_workers,
        seed_offset=args.seed_offset,
        data_parallel_group=None,
    )
    dataset = get_train_dataset(
        args.data_path,
        batch_size=args.batch_size,
        task_encoder=encoder,
        worker_config=worker_config,
        packing_buffer_size=args.packing_buffer_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples_per_sequence=args.max_samples_per_sequence,
    )
    return iter(get_loader(dataset))


def build_encoder(
    provider: ModuleType, tokenizer, image_token_id: int, pad_id: int, args: argparse.Namespace
):
    """Build a provider-specific MIMO multimodal packing encoder."""
    if provider is current_provider:
        return provider.build_multimodal_encoder(
            args,
            tokenizer,
            encoder_name=args.encoder_name,
            encoder_input_key=args.encoder_input_key,
        )

    vision_config = provider.VisionConfig(
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        vision_model_type=args.vision_model_type,
        disable_vision_class_token=args.disable_vision_class_token,
        pixel_shuffle=args.pixel_shuffle,
        max_num_tiles=args.max_num_tiles,
        use_tiling=args.use_tiling,
        use_thumbnail=args.use_thumbnail,
        class_token_len=args.class_token_len,
        conv_merging=False,
        use_tile_tags=False,
        use_image_break_token=False,
        use_area_weighted_aspect_ratio=False,
        dynamic_resolution=False,
    )
    packing_config = provider.PackingConfig(
        seq_length=args.seq_length, pad_id=pad_id, image_token_id=image_token_id
    )
    adapter_cls = getattr(provider, "TokenizerAdapter", None)
    if adapter_cls is None:
        adapter_cls = getattr(provider, "_TokenizerAdapter")
    return provider.MimoMultiModalPackingEncoder(
        vision_config=vision_config,
        packing_config=packing_config,
        tokenizer=adapter_cls(tokenizer),
        encoder_name=args.encoder_name,
        encoder_input_key=args.encoder_input_key,
        target_seq_length=args.seq_length,
    )


def compare_values(path: str, old_value: Any, current_value: Any) -> list[str]:
    """Return exact mismatches between nested batch values."""
    if isinstance(old_value, dict) or isinstance(current_value, dict):
        if not isinstance(old_value, dict) or not isinstance(current_value, dict):
            old_type = type(old_value).__name__
            current_type = type(current_value).__name__
            return [f"{path}: type mismatch {old_type} != {current_type}"]
        mismatches = []
        old_keys = set(old_value)
        current_keys = set(current_value)
        if old_keys != current_keys:
            mismatches.append(
                f"{path}: keys differ old={sorted(old_keys)} current={sorted(current_keys)}"
            )
        for key in sorted(old_keys & current_keys):
            mismatches.extend(compare_values(f"{path}.{key}", old_value[key], current_value[key]))
        return mismatches

    if isinstance(old_value, torch.Tensor) or isinstance(current_value, torch.Tensor):
        if not isinstance(old_value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
            return [f"{path}: tensor/type mismatch"]
        if old_value.shape != current_value.shape:
            return [
                f"{path}: shape mismatch {tuple(old_value.shape)} != {tuple(current_value.shape)}"
            ]
        if old_value.dtype != current_value.dtype:
            return [f"{path}: dtype mismatch {old_value.dtype} != {current_value.dtype}"]
        if torch.equal(old_value, current_value):
            return []
        detail = f"checksum {tensor_checksum(old_value)} != {tensor_checksum(current_value)}"
        if old_value.is_floating_point():
            max_abs = (old_value - current_value).abs().max().item()
            detail += f", max_abs={max_abs}"
        return [f"{path}: tensor mismatch ({detail})"]

    if old_value != current_value:
        return [f"{path}: value mismatch {old_value!r} != {current_value!r}"]
    return []


def batch_summary(batch: dict) -> str:
    """Return a compact human-readable batch summary."""
    image_tensor = first_tensor(nested_get(batch, ("modality_inputs", "images")))
    packing_kwargs = batch.get("packing_kwargs")
    cu_seqlens = None
    if packing_kwargs is not None:
        cu_seqlens = packing_kwargs["cu_seqlens_q"]
    return (
        f"input={tuple(batch['input_ids'].shape)}:{tensor_checksum(batch['input_ids'])} "
        f"labels={tensor_checksum(batch['labels'])} "
        f"loss_tokens={int(batch['loss_mask'].sum().item())} "
        f"images={None if image_tensor is None else tuple(image_tensor.shape)}:"
        f"{tensor_checksum(image_tensor)} "
        f"cu={None if cu_seqlens is None else cu_seqlens.tolist()[:8]}"
    )


def nested_get(value: dict, keys: tuple[str, ...]):
    """Return a nested value if every key exists."""
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_tensor(value):
    """Return the first tensor in a nested mapping."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_checksum(tensor: Optional[torch.Tensor]) -> int:
    """Return a deterministic bounded checksum for a tensor."""
    if tensor is None or tensor.numel() == 0:
        return 0
    values = tensor.detach().reshape(-1)
    stride = max(values.numel() // 4096, 1)
    values = values[::stride]
    if values.is_floating_point():
        values = (values.float() * 1024).to(dtype=torch.long)
    else:
        values = values.to(dtype=torch.long)
    positions = torch.arange(1, values.numel() + 1, dtype=torch.long, device=values.device)
    return int(((values * positions).sum() % 2_147_483_647).item())


if __name__ == "__main__":
    main()
