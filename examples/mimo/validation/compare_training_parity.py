# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare artifacts produced by ``training_parity.py``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    """Parse comparison arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homo-dir", type=Path, required=True)
    parser.add_argument("--hetero-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--max-loss-diff", type=float, default=1.0e-5)
    parser.add_argument("--require-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    summary = compare_runs(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def compare_runs(args: argparse.Namespace) -> dict[str, Any]:
    """Compare loss curves, sample ids, and optional state snapshots."""
    homo_metrics = load_metrics(args.homo_dir)
    hetero_metrics = load_metrics(args.hetero_dir)
    loss_summary = compare_loss_curves(homo_metrics, hetero_metrics, args.max_loss_diff)
    sample_summary = compare_sample_ids(homo_metrics, hetero_metrics)
    state_summary = compare_state_dirs(args)
    return {
        "loss": loss_summary,
        "samples": sample_summary,
        "state": state_summary,
        "status": "pass",
    }


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Load all rank-local JSONL metric records."""
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("metrics_rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise AssertionError(f"no metrics found under {run_dir}")
    return records


def compare_loss_curves(
    homo_metrics: list[dict[str, Any]],
    hetero_metrics: list[dict[str, Any]],
    max_loss_diff: float,
) -> dict[str, Any]:
    """Compare scalar loss records emitted by the logging rank."""
    homo = losses_by_iteration(homo_metrics)
    hetero = losses_by_iteration(hetero_metrics)
    if set(homo) != set(hetero):
        raise AssertionError(
            f"loss iterations differ: homo={sorted(homo)}, hetero={sorted(hetero)}"
        )
    max_abs_diff = 0.0
    worst_iteration = None
    for iteration in sorted(homo):
        diff = abs(homo[iteration] - hetero[iteration])
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_iteration = iteration
    if max_abs_diff > max_loss_diff:
        pairs = [
            {
                "iteration": iteration,
                "homo": homo[iteration],
                "hetero": hetero[iteration],
                "abs_diff": abs(homo[iteration] - hetero[iteration]),
            }
            for iteration in sorted(homo)
        ]
        raise AssertionError(
            f"loss mismatch: max_abs_diff={max_abs_diff} at iter={worst_iteration}, "
            f"threshold={max_loss_diff}; losses={pairs[:8]}"
        )
    return {
        "iterations": len(homo),
        "max_abs_diff": max_abs_diff,
        "worst_iteration": worst_iteration,
    }


def losses_by_iteration(metrics: list[dict[str, Any]]) -> dict[int, float]:
    """Return one scalar loss per iteration."""
    losses: dict[int, float] = {}
    for record in metrics:
        loss = record.get("loss")
        if loss is None:
            continue
        iteration = int(record["iteration"])
        if iteration in losses:
            raise AssertionError(f"multiple loss records for iteration {iteration}")
        losses[iteration] = float(loss)
    if not losses:
        raise AssertionError("no scalar loss records found")
    return losses


def compare_sample_ids(
    homo_metrics: list[dict[str, Any]], hetero_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate both runs consumed the same global sample ids per iteration."""
    homo = sample_sets_by_iteration(homo_metrics)
    hetero = sample_sets_by_iteration(hetero_metrics)
    if set(homo) != set(hetero):
        raise AssertionError(
            f"sample iterations differ: homo={sorted(homo)}, hetero={sorted(hetero)}"
        )
    for iteration in sorted(homo):
        if homo[iteration] != hetero[iteration]:
            raise AssertionError(
                f"sample ids differ at iter={iteration}: "
                f"homo={sorted(homo[iteration])}, hetero={sorted(hetero[iteration])}"
            )
    return {"iterations": len(homo)}


def sample_sets_by_iteration(metrics: list[dict[str, Any]]) -> dict[int, set[int]]:
    """Return union of rank-local sample ids per iteration."""
    samples: dict[int, set[int]] = {}
    for record in metrics:
        if "iteration" not in record:
            continue
        iteration = int(record["iteration"])
        samples.setdefault(iteration, set()).update(int(v) for v in record.get("sample_ids", []))
    return samples


def compare_state_dirs(args: argparse.Namespace) -> dict[str, Any]:
    """Compare per-iteration tensor snapshots when present."""
    homo_iters = {path.name: path for path in args.homo_dir.glob("iter_*") if path.is_dir()}
    hetero_iters = {path.name: path for path in args.hetero_dir.glob("iter_*") if path.is_dir()}
    common_iters = sorted(set(homo_iters) & set(hetero_iters))
    if args.require_state and not common_iters:
        raise AssertionError("state snapshots were required but none were found")
    if set(homo_iters) != set(hetero_iters):
        raise AssertionError(
            f"state iterations differ: homo={sorted(homo_iters)}, hetero={sorted(hetero_iters)}"
        )

    compared_tensors = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    worst_key = None
    worst_iter = None
    mismatches: list[dict[str, Any]] = []
    for iter_name in common_iters:
        homo = load_snapshot(homo_iters[iter_name])
        hetero = load_snapshot(hetero_iters[iter_name])
        if set(homo) != set(hetero):
            missing = sorted(set(homo) - set(hetero))
            extra = sorted(set(hetero) - set(homo))
            raise AssertionError(
                f"state keys differ at {iter_name}: missing={missing[:10]}, extra={extra[:10]}"
            )
        for key in sorted(homo):
            diff, rel = tensor_diff(homo[key], hetero[key])
            compared_tensors += 1
            if diff > max_abs_diff or rel > max_rel_diff:
                max_abs_diff = max(max_abs_diff, diff)
                max_rel_diff = max(max_rel_diff, rel)
                worst_key = key
                worst_iter = iter_name
            if not torch.allclose(homo[key], hetero[key], atol=args.atol, rtol=args.rtol):
                mismatches.append(
                    {
                        "iteration": iter_name,
                        "key": key,
                        "max_abs_diff": diff,
                        "max_rel_diff": rel,
                    }
                )
    if mismatches:
        top = sorted(
            mismatches, key=lambda item: (item["max_abs_diff"], item["max_rel_diff"]), reverse=True
        )[:12]
        raise AssertionError(
            f"state mismatches={len(mismatches)}, top={top}, atol={args.atol}, rtol={args.rtol}"
        )
    return {
        "iterations": len(common_iters),
        "tensors": compared_tensors,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "worst_iteration": worst_iter,
        "worst_key": worst_key,
    }


def load_snapshot(iter_dir: Path) -> dict[str, torch.Tensor]:
    """Load and merge all rank-local snapshot shards from one iteration."""
    entries: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(iter_dir.glob("rank_*.pt")):
        snapshot = torch.load(path, map_location="cpu")
        for name, entry in snapshot.get("params", {}).items():
            entries.setdefault(f"params::{name}", []).append(entry)
        for name, entry in snapshot.get("grads", {}).items():
            entries.setdefault(f"grads::{name}", []).append(entry)
        for param_name, states in snapshot.get("optimizer", {}).items():
            for state_name, entry in states.items():
                entries.setdefault(f"optimizer::{param_name}::{state_name}", []).append(entry)
    if not entries:
        raise AssertionError(f"no tensor entries found under {iter_dir}")
    return {key: merge_entries(key, value) for key, value in entries.items()}


def merge_entries(key: str, entries: list[dict[str, Any]]) -> torch.Tensor:
    """Merge duplicated full tensors or distributed-optimizer shards."""
    first = entries[0]
    shape = tuple(first["shape"])
    numel = int(first["numel"])
    full = torch.empty(numel, dtype=torch.float32)
    filled = torch.zeros(numel, dtype=torch.bool)

    for entry in entries:
        if tuple(entry["shape"]) != shape or int(entry["numel"]) != numel:
            raise AssertionError(f"inconsistent metadata for {key}")
        tensor = entry["tensor"].float().view(-1)
        start, end = (int(v) for v in entry["range"])
        if end - start != tensor.numel():
            if tensor.numel() == numel and start == 0 and end == numel:
                pass
            else:
                raise AssertionError(
                    f"bad tensor range for {key}: range=({start}, {end}), "
                    f"tensor_numel={tensor.numel()}"
                )
        existing = filled[start:end]
        if existing.any():
            old = full[start:end][existing]
            new = tensor[existing]
            if not torch.equal(old, new):
                diff = (old - new).abs().max().item()
                raise AssertionError(f"duplicate shard mismatch for {key}: max_abs_diff={diff}")
        missing = ~existing
        if missing.any():
            full[start:end][missing] = tensor[missing]
            filled[start:end][missing] = True

    if not filled.all():
        missing_count = int((~filled).sum().item())
        raise AssertionError(f"incomplete tensor reconstruction for {key}: missing={missing_count}")
    return full.view(shape)


def tensor_diff(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    """Return max absolute and relative differences."""
    if left.numel() == 0 and right.numel() == 0:
        return 0.0, 0.0
    abs_diff = (left - right).abs()
    max_abs = abs_diff.max().item()
    denom = torch.maximum(left.abs(), right.abs()).clamp_min(1.0e-12)
    max_rel = (abs_diff / denom).max().item()
    if math.isnan(max_abs) or math.isnan(max_rel):
        raise AssertionError("NaN encountered while comparing tensors")
    return max_abs, max_rel


if __name__ == "__main__":
    main()
