# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Analyze torch.profiler output for MIMO colocated training.

Produces a structured report with timeline, per-phase breakdown, and memory
analysis that the optimization agent can parse.
"""

from __future__ import annotations

import re
from typing import Optional


# Phase ordering for timeline display.
_MIMO_PHASE_ORDER = [
    "mimo::forward_backward",
    "mimo::encoder_forward",
    "mimo::bridge_communicate",
    "mimo::text_embedding",
    "mimo::align_embeddings",
    "mimo::llm_forward",
    "mimo::optimizer_step",
]

# Op classification patterns.  Checked in order; first match wins.
_OP_CATEGORIES = [
    (
        "Compute",
        re.compile(
            r"gemm|nvjet|cutlass|cublas|linear|attn|fused|layernorm",
            re.IGNORECASE,
        ),
    ),
    (
        "Communication",
        re.compile(
            r"nccl|allreduce|all_reduce|allgather|all_gather"
            r"|reduce_scatter|broadcast|c10d::",
            re.IGNORECASE,
        ),
    ),
    (
        "Memory",
        re.compile(r"memcpy|memset|empty|resize", re.IGNORECASE),
    ),
    (
        "Elementwise",
        re.compile(
            r"aten::add|aten::mul|dropout|gelu|elementwise|aten::copy_",
            re.IGNORECASE,
        ),
    ),
]

# NCCL collective patterns for the communication breakdown.
_NCCL_COLLECTIVES = [
    ("AllReduce", re.compile(r"allreduce|all_reduce", re.IGNORECASE)),
    ("AllGather", re.compile(r"allgather|all_gather", re.IGNORECASE)),
    ("ReduceScatter", re.compile(r"reduce_scatter", re.IGNORECASE)),
    ("Broadcast", re.compile(r"broadcast", re.IGNORECASE)),
]


def _classify_op(name: str) -> str:
    """Return the category string for a profiler event name."""
    for category, pattern in _OP_CATEGORIES:
        if pattern.search(name):
            return category
    return "Other"


def _us_to_ms(us: float) -> float:
    return us / 1000.0


def _bytes_to_mb(b: float) -> float:
    return b / (1024.0 * 1024.0)


def _bytes_to_gb(b: float) -> float:
    return b / (1024.0 * 1024.0 * 1024.0)


def _pct(part: float, total: float) -> float:
    if total == 0:
        return 0.0
    return 100.0 * part / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_profile(prof, config_description: str = "") -> str:
    """Analyze profiler output and return structured text report.

    Args:
        prof: torch.profiler.profile object (after profiling is complete).
        config_description: Optional string describing the config
            (e.g., "1B+7B enc_TP2/DP4 llm_TP4/DP2").

    Returns:
        Structured text report string.
    """
    key_averages = prof.key_averages()

    # ------------------------------------------------------------------
    # 1. Extract mimo:: phase events
    # ------------------------------------------------------------------
    mimo_phases: dict[str, dict] = {}
    for evt in key_averages:
        if evt.key.startswith("mimo::"):
            mimo_phases[evt.key] = {
                "cuda_time_us": evt.cuda_time * evt.count,
                "self_device_memory_bytes": getattr(
                    evt, "self_device_memory_usage", 0
                ),
                "count": evt.count,
            }

    has_mimo = bool(mimo_phases)

    # ------------------------------------------------------------------
    # 2. Categorize ALL ops
    # ------------------------------------------------------------------
    category_totals: dict[str, float] = {}   # category -> total cuda_time_us
    category_counts: dict[str, int] = {}
    comm_breakdown: dict[str, dict] = {}     # collective -> {time_us, count}

    for evt in key_averages:
        # Skip mimo:: wrapper events for the category analysis to avoid
        # double-counting.
        if evt.key.startswith("mimo::"):
            continue

        cat = _classify_op(evt.key)
        cuda_us = evt.cuda_time * evt.count  # total, not average
        category_totals[cat] = category_totals.get(cat, 0.0) + cuda_us
        category_counts[cat] = category_counts.get(cat, 0) + evt.count

        # Communication sub-breakdown
        if cat == "Communication":
            for coll_name, coll_pat in _NCCL_COLLECTIVES:
                if coll_pat.search(evt.key):
                    entry = comm_breakdown.setdefault(
                        coll_name, {"time_us": 0.0, "count": 0}
                    )
                    entry["time_us"] += cuda_us
                    entry["count"] += evt.count
                    break

    total_cuda_us = sum(category_totals.values())

    # ------------------------------------------------------------------
    # 3. Top memory-allocating ops
    # ------------------------------------------------------------------
    mem_ops = []
    for evt in key_averages:
        mem_bytes = getattr(evt, "self_device_memory_usage", 0)
        if mem_bytes > 0:
            mem_ops.append((evt.key, mem_bytes, evt.count))
    mem_ops.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # 4. Try to get peak memory from torch.cuda
    # ------------------------------------------------------------------
    peak_memory_gb: Optional[float] = None
    try:
        import torch

        peak_memory_gb = _bytes_to_gb(torch.cuda.max_memory_allocated())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    lines: list[str] = []

    def section(title: str):
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  {title}")
        lines.append("=" * 72)

    def subsection(title: str):
        lines.append("")
        lines.append(f"--- {title} ---")

    # Header
    lines.append("MIMO Profiler Analysis Report")
    if config_description:
        lines.append(f"Config: {config_description}")

    # ------------------------------------------------------------------
    # VIEW 1: Timeline
    # ------------------------------------------------------------------
    section("VIEW 1: Phase Timeline")

    if has_mimo:
        # Order phases: known order first, then any extras alphabetically.
        ordered_keys = [p for p in _MIMO_PHASE_ORDER if p in mimo_phases]
        extras = sorted(set(mimo_phases) - set(_MIMO_PHASE_ORDER))
        ordered_keys.extend(extras)

        # Determine the outer phase for percentage calculations.
        outer_key = "mimo::forward_backward"
        if outer_key in mimo_phases:
            total_iter_us = mimo_phases[outer_key]["cuda_time_us"]
        else:
            total_iter_us = max(
                (v["cuda_time_us"] for v in mimo_phases.values()), default=1
            )

        longest_key = max(
            mimo_phases, key=lambda k: mimo_phases[k]["cuda_time_us"]
        )

        lines.append("")
        lines.append(
            f"{'Phase':<35} {'CUDA ms':>10} {'% iter':>8} {'Calls':>7}"
        )
        lines.append("-" * 62)

        for key in ordered_keys:
            info = mimo_phases[key]
            ms = _us_to_ms(info["cuda_time_us"])
            pct = _pct(info["cuda_time_us"], total_iter_us)
            marker = "  <<<< longest" if key == longest_key else ""
            lines.append(
                f"{key:<35} {ms:>10.2f} {pct:>7.1f}% {info['count']:>7d}"
                f"{marker}"
            )

        lines.append("")
        lines.append(
            f"Total iteration CUDA time: {_us_to_ms(total_iter_us):.2f} ms"
        )
    else:
        lines.append("")
        lines.append(
            "No mimo:: annotated events found.  Showing raw key_averages table"
            " (top 30 by CUDA time):"
        )
        lines.append("")

        sorted_evts = sorted(
            key_averages, key=lambda e: e.cuda_time * e.count, reverse=True
        )
        lines.append(
            f"{'Op':<50} {'CUDA ms':>10} {'Calls':>7}"
        )
        lines.append("-" * 69)
        for evt in sorted_evts[:30]:
            ms = _us_to_ms(evt.cuda_time * evt.count)
            lines.append(f"{evt.key[:50]:<50} {ms:>10.2f} {evt.count:>7d}")

    # ------------------------------------------------------------------
    # VIEW 2: Op-level breakdown
    # ------------------------------------------------------------------
    section("VIEW 2: Op-Level Breakdown (all ops, excluding mimo:: wrappers)")

    lines.append("")
    lines.append(
        f"{'Category':<20} {'CUDA ms':>10} {'% total':>9} {'Op count':>10}"
    )
    lines.append("-" * 51)

    for cat in ["Compute", "Communication", "Elementwise", "Memory", "Other"]:
        t = category_totals.get(cat, 0.0)
        c = category_counts.get(cat, 0)
        ms = _us_to_ms(t)
        pct = _pct(t, total_cuda_us)
        lines.append(f"{cat:<20} {ms:>10.2f} {pct:>8.1f}% {c:>10d}")

    lines.append("-" * 51)
    lines.append(
        f"{'TOTAL':<20} {_us_to_ms(total_cuda_us):>10.2f} {'100.0%':>9}"
    )

    # Communication sub-breakdown
    if comm_breakdown:
        subsection("Communication Breakdown by Collective")
        lines.append("")
        lines.append(
            f"{'Collective':<20} {'Total ms':>10} {'Calls':>8} {'Avg ms':>10}"
        )
        lines.append("-" * 50)

        for coll_name in ["AllReduce", "AllGather", "ReduceScatter", "Broadcast"]:
            if coll_name not in comm_breakdown:
                continue
            entry = comm_breakdown[coll_name]
            total_ms = _us_to_ms(entry["time_us"])
            count = entry["count"]
            avg_ms = total_ms / count if count > 0 else 0.0
            lines.append(
                f"{coll_name:<20} {total_ms:>10.2f} {count:>8d} {avg_ms:>10.3f}"
            )

    # ------------------------------------------------------------------
    # VIEW 3: Memory
    # ------------------------------------------------------------------
    section("VIEW 3: Memory Analysis")

    if peak_memory_gb is not None:
        lines.append("")
        lines.append(f"Peak GPU memory allocated: {peak_memory_gb:.2f} GB")
        utilization = _pct(peak_memory_gb, 80.0)
        lines.append(f"GPU memory utilization:    {utilization:.1f}% of 80 GB")

    if mem_ops:
        subsection("Top Memory-Allocating Ops")
        lines.append("")
        lines.append(f"{'Op':<50} {'MB':>10} {'Calls':>7}")
        lines.append("-" * 69)
        for name, mem_bytes, count in mem_ops[:15]:
            mb = _bytes_to_mb(mem_bytes)
            lines.append(f"{name[:50]:<50} {mb:>10.2f} {count:>7d}")

        # Flag encoder activations as offload candidates
        encoder_mem = sum(
            mb
            for name, mb, _ in mem_ops
            if "encoder" in name.lower() or "vit" in name.lower()
        )
        if encoder_mem > 0:
            lines.append("")
            lines.append(
                f"* Encoder-related allocations: {_bytes_to_mb(encoder_mem):.2f} MB"
                " (candidate for activation offloading)"
            )
    else:
        lines.append("")
        lines.append("No device memory allocation data available in profile.")

    # ------------------------------------------------------------------
    # Bottleneck Assessment
    # ------------------------------------------------------------------
    section("Bottleneck Assessment")
    lines.append("")

    assessments: list[str] = []

    comm_us = category_totals.get("Communication", 0.0)
    compute_us = category_totals.get("Compute", 0.0)

    if total_cuda_us > 0:
        comm_pct = _pct(comm_us, total_cuda_us)
        compute_pct = _pct(compute_us, total_cuda_us)

        if comm_pct > 25.0:
            assessments.append(
                f"COMMUNICATION-BOUND: Communication is {comm_pct:.1f}% of "
                f"total CUDA time ({_us_to_ms(comm_us):.1f} ms). Consider "
                "reducing TP degree or overlapping communication with compute."
            )
        if compute_pct < 50.0:
            assessments.append(
                f"UNDERUTILIZED COMPUTE: Compute is only {compute_pct:.1f}% of "
                f"total CUDA time. Non-compute overhead is significant."
            )

    if peak_memory_gb is not None and peak_memory_gb > 0.7 * 80.0:
        assessments.append(
            f"MEMORY-CONSTRAINED: Peak memory {peak_memory_gb:.1f} GB "
            f"exceeds 70% of 80 GB. Consider activation offloading or "
            "reducing micro-batch size."
        )

    if not assessments:
        assessments.append(
            "No major bottleneck detected. Profile looks balanced."
        )

    for assessment in assessments:
        lines.append(f"  * {assessment}")

    # ------------------------------------------------------------------
    # Cross-reference hint (only when both views are available)
    # ------------------------------------------------------------------
    if has_mimo and comm_breakdown:
        lines.append("")
        lines.append("Hint: Cross-reference phase durations (View 1) with")
        lines.append("communication totals (View 2) to identify which phase")
        lines.append("contains the communication bottleneck.")

    lines.append("")
    return "\n".join(lines)


def save_analysis(report: str, filepath: str):
    """Save the analysis report to a file.

    Args:
        report: The text report produced by :func:`analyze_profile`.
        filepath: Destination file path.
    """
    with open(filepath, "w") as f:
        f.write(report)
