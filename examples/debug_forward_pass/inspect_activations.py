#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inspect activation files produced by debug_forward_pass.py."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved Megatron-LM activation tensors.")
    parser.add_argument("activation_file", type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        help="Optional second activation file to compare layer-by-layer.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="If set, write per-layer histogram PNGs to this directory.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Histogram bin count used with --plot-dir.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        help="Restrict output to a layer key such as layer_001. Can be repeated.",
    )
    return parser.parse_args()


def torch_load(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected {path} to contain a dict payload, got {type(payload)}")
    return payload


def get_activations(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    activations = payload.get("activations", payload)
    if not isinstance(activations, Mapping):
        raise TypeError("Activation payload does not contain a mapping under 'activations'.")
    for name, tensor in activations.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"Activation {name} is {type(tensor)}, expected torch.Tensor")
    return activations


def select_layers(
    activations: Mapping[str, torch.Tensor],
    requested_layers: Optional[list[str]],
) -> "OrderedDict[str, torch.Tensor]":
    if not requested_layers:
        return OrderedDict(sorted(activations.items()))

    selected: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for layer in requested_layers:
        if layer not in activations:
            available = ", ".join(sorted(activations.keys()))
            raise KeyError(f"Layer {layer} not found. Available layers: {available}")
        selected[layer] = activations[layer]
    return selected


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def print_stats(title: str, activations: Mapping[str, torch.Tensor]) -> None:
    print(title)
    for name, tensor in activations.items():
        stats = tensor_stats(tensor)
        shape = "x".join(str(dim) for dim in stats["shape"])
        print(
            f"  {name} shape={shape} dtype={stats['dtype']} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )


def compare_activations(
    lhs: Mapping[str, torch.Tensor],
    rhs: Mapping[str, torch.Tensor],
    rhs_path: Path,
) -> None:
    print(f"Comparison against {rhs_path}:")
    all_layers = sorted(set(lhs.keys()) | set(rhs.keys()))
    for name in all_layers:
        if name not in lhs:
            print(f"  {name} only present in comparison file")
            continue
        if name not in rhs:
            print(f"  {name} only present in primary file")
            continue

        left = lhs[name].detach().float()
        right = rhs[name].detach().float()
        if left.shape != right.shape:
            print(f"  {name} shape mismatch: {list(left.shape)} vs {list(right.shape)}")
            continue

        diff = left - right
        mean_abs = diff.abs().mean().item()
        max_abs = diff.abs().max().item()
        rms = diff.square().mean().sqrt().item()
        cosine = torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0)
        print(
            f"  {name} mean_abs={mean_abs:.6e} max_abs={max_abs:.6e} "
            f"rms={rms:.6e} cosine={cosine.item():.8f}"
        )


def plot_histograms(
    activations: Mapping[str, torch.Tensor],
    plot_dir: Path,
    bins: int,
    compare_activations_map: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot-dir") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in activations.items():
        values = tensor.detach().float().flatten().numpy()
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=bins, alpha=0.70, label="primary")

        if compare_activations_map is not None and name in compare_activations_map:
            other = compare_activations_map[name].detach().float()
            if other.shape == tensor.shape:
                plt.hist(other.flatten().numpy(), bins=bins, alpha=0.45, label="compare")
                plt.legend()

        plt.title(name)
        plt.xlabel("Activation value")
        plt.ylabel("Count")
        plt.tight_layout()
        output_path = plot_dir / f"{name}_hist.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    payload = torch_load(args.activation_file)
    activations = select_layers(get_activations(payload), args.layer)

    print_stats(f"Activation statistics for {args.activation_file}:", activations)

    comparison = None
    if args.compare:
        comparison_payload = torch_load(args.compare)
        comparison = select_layers(get_activations(comparison_payload), args.layer)
        compare_activations(activations, comparison, args.compare)

    if args.plot_dir:
        plot_histograms(activations, args.plot_dir, args.bins, comparison)


if __name__ == "__main__":
    main()
