# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Analyze torch.profiler output for MIMO colocated training.

Uses prof.events() (individual FunctionEvent objects with inclusive CUDA times)
rather than key_averages() (which gives self-time only). Produces a structured
report with timeline, per-phase op breakdown, memory waterfall, and bottleneck
assessment.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

# Phase ordering for timeline display.
_MIMO_PHASE_ORDER = [
    "mimo::encoder_forward",
    "mimo::bridge_communicate",
    "mimo::text_embedding",
    "mimo::align_embeddings",
    "mimo::llm_forward",
    "mimo::optimizer_step",
]

# Op classification patterns. Checked in order; first match wins.
_OP_CATEGORIES = [
    ("Compute", re.compile(
        r"gemm|nvjet|cutlass|cublas|_Linear|_LayerNormLinear|FusedAttn|layernorm",
        re.IGNORECASE,
    )),
    ("Communication", re.compile(
        r"nccl|allreduce|all_reduce|allgather|all_gather"
        r"|reduce_scatter|broadcast|c10d::|record_param_comms",
        re.IGNORECASE,
    )),
    ("Elementwise", re.compile(
        r"aten::add|aten::mul|dropout|gelu|elementwise|aten::copy_",
        re.IGNORECASE,
    )),
]

_NCCL_COLLECTIVES = [
    ("AllReduce", re.compile(r"allreduce|all_reduce", re.IGNORECASE)),
    ("AllGather", re.compile(r"allgather|all_gather", re.IGNORECASE)),
    ("ReduceScatter", re.compile(r"reduce_scatter", re.IGNORECASE)),
    ("Broadcast", re.compile(r"broadcast", re.IGNORECASE)),
]


def _classify_op(name: str) -> str:
    for category, pattern in _OP_CATEGORIES:
        if pattern.search(name):
            return category
    return "Other"


def _us_to_ms(us: float) -> float:
    return us / 1000.0


def _pct(part: float, total: float) -> float:
    return 100.0 * part / total if total > 0 else 0.0


def analyze_profile(
    prof,
    config_description: str = "",
    memory_waterfall: Optional[dict] = None,
) -> str:
    """Analyze profiler output and return structured text report.

    Args:
        prof: torch.profiler.profile object (after profiling is complete).
        config_description: String describing the config.
        memory_waterfall: Optional dict mapping phase names to memory_allocated
            in bytes at the END of that phase. Recorded by the training loop.

    Returns:
        Structured text report string.
    """
    # Use prof.events() for accurate inclusive CUDA time (not key_averages).
    all_events = prof.events()

    # ── 1. Extract mimo:: phase events (inclusive CUDA time) ──────────
    # FunctionEvent objects from record_function have .cuda_time_total
    # which includes all child kernel time.
    mimo_phases: dict[str, dict] = {}
    for evt in all_events:
        if evt.name.startswith("mimo::") and evt.name != "mimo::forward_backward":
            key = evt.name
            if key not in mimo_phases:
                mimo_phases[key] = {"cuda_time_us": 0, "count": 0}
            mimo_phases[key]["cuda_time_us"] += evt.cuda_time_total
            mimo_phases[key]["count"] += 1

    has_mimo = bool(mimo_phases)

    # ── 2. Categorize leaf ops (using key_averages for aggregation) ───
    key_averages = prof.key_averages()
    category_totals: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    comm_breakdown: dict[str, dict] = {}

    for evt in key_averages:
        if evt.key.startswith("mimo::"):
            continue
        cat = _classify_op(evt.key)
        # self_cuda_time avoids double counting in nested events
        cuda_us = evt.cuda_time * evt.count
        category_totals[cat] = category_totals.get(cat, 0.0) + cuda_us
        category_counts[cat] = category_counts.get(cat, 0) + evt.count

        if cat == "Communication":
            for coll_name, coll_pat in _NCCL_COLLECTIVES:
                if coll_pat.search(evt.key):
                    entry = comm_breakdown.setdefault(coll_name, {"time_us": 0.0, "count": 0})
                    entry["time_us"] += cuda_us
                    entry["count"] += evt.count
                    break

    total_leaf_cuda_us = sum(category_totals.values())

    # ── 3. Build report ──────────────────────────────────────────────
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("MIMO PROFILE ANALYSIS")
    lines.append("=" * 72)
    if config_description:
        lines.append(f"Config: {config_description}")
    lines.append("")

    # ── VIEW 1: Phase Timeline (inclusive CUDA time) ─────────────────
    lines.append("─── PHASE TIMELINE (inclusive CUDA time) " + "─" * 31)
    lines.append("")

    if has_mimo:
        ordered = [p for p in _MIMO_PHASE_ORDER if p in mimo_phases]
        ordered += sorted(set(mimo_phases) - set(_MIMO_PHASE_ORDER))

        total_phase_us = sum(v["cuda_time_us"] for v in mimo_phases.values()
                             if not v.get("_is_outer"))
        longest = max(mimo_phases, key=lambda k: mimo_phases[k]["cuda_time_us"])

        lines.append(f"{'Phase':<35} {'CUDA ms':>10} {'% iter':>8} {'Calls':>7}  {'Avg ms':>8}")
        lines.append("─" * 72)

        for key in ordered:
            info = mimo_phases[key]
            total_ms = _us_to_ms(info["cuda_time_us"])
            pct = _pct(info["cuda_time_us"], total_phase_us)
            avg_ms = total_ms / max(info["count"], 1)
            marker = " ◄ longest" if key == longest else ""
            short = key.replace("mimo::", "")
            lines.append(
                f"{short:<35} {total_ms:>10.1f} {pct:>7.1f}% {info['count']:>7d}  {avg_ms:>8.2f}{marker}"
            )

        lines.append("─" * 72)
        lines.append(f"{'SUM':<35} {_us_to_ms(total_phase_us):>10.1f}")
        lines.append("")

        # Sanity check vs wall-clock
        lines.append(f"(Wall-clock fwd_bwd from harness timers is the ground truth.")
        lines.append(f" CUDA inclusive times may exceed wall-clock due to overlap.)")
    else:
        lines.append("No mimo:: annotated events found. Run with record_function annotations.")

    lines.append("")

    # ── VIEW 2: Op-Level Breakdown ───────────────────────────────────
    lines.append("─── OP-LEVEL BREAKDOWN (leaf ops, self CUDA time) " + "─" * 22)
    lines.append("")
    lines.append(f"{'Category':<20} {'CUDA ms':>10} {'% total':>9} {'Op count':>10}")
    lines.append("─" * 51)

    for cat in ["Compute", "Communication", "Elementwise", "Other"]:
        t = category_totals.get(cat, 0.0)
        c = category_counts.get(cat, 0)
        lines.append(f"{cat:<20} {_us_to_ms(t):>10.1f} {_pct(t, total_leaf_cuda_us):>8.1f}% {c:>10d}")

    lines.append("─" * 51)
    lines.append(f"{'TOTAL':<20} {_us_to_ms(total_leaf_cuda_us):>10.1f} {'100.0%':>9}")
    lines.append("")

    if comm_breakdown:
        lines.append(f"{'Collective':<20} {'Total ms':>10} {'Calls':>8} {'Avg ms':>10}")
        lines.append("─" * 50)
        for coll_name in ["AllReduce", "AllGather", "ReduceScatter", "Broadcast"]:
            if coll_name not in comm_breakdown:
                continue
            entry = comm_breakdown[coll_name]
            total_ms = _us_to_ms(entry["time_us"])
            count = entry["count"]
            avg_ms = total_ms / count if count > 0 else 0.0
            lines.append(f"{coll_name:<20} {total_ms:>10.2f} {count:>8d} {avg_ms:>10.3f}")
        lines.append("")

    # ── VIEW 3: Memory Waterfall ─────────────────────────────────────
    lines.append("─── MEMORY " + "─" * 61)
    lines.append("")

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    lines.append(f"Peak GPU memory:  {peak_gb:.2f} GB  ({_pct(peak_gb, 80.0):.0f}% of 80 GB)")
    lines.append(f"Headroom:         {80.0 - peak_gb:.1f} GB")
    lines.append("")

    if memory_waterfall:
        lines.append(f"{'Phase':<35} {'Mem (GB)':>10} {'Delta':>10}")
        lines.append("─" * 57)
        prev_gb = 0.0
        for phase_name in _MIMO_PHASE_ORDER:
            if phase_name not in memory_waterfall:
                continue
            mem_gb = memory_waterfall[phase_name] / (1024**3)
            delta = mem_gb - prev_gb
            short = phase_name.replace("mimo::", "")
            peak_marker = " ◄ PEAK" if abs(mem_gb - peak_gb) < 0.5 else ""
            lines.append(f"{short:<35} {mem_gb:>10.2f} {delta:>+9.2f}{peak_marker}")
            prev_gb = mem_gb
        lines.append("")
    else:
        lines.append("(No memory waterfall data. Pass memory_waterfall dict to analyze_profile.)")
        lines.append("")

    # Top allocators from profiler
    mem_ops = []
    for evt in key_averages:
        mem_bytes = getattr(evt, "self_device_memory_usage", 0)
        if mem_bytes > 0:
            mem_ops.append((evt.key, mem_bytes, evt.count))
    mem_ops.sort(key=lambda x: -x[1])

    if mem_ops:
        lines.append(f"{'Top allocators':<50} {'MB':>10} {'Calls':>7}")
        lines.append("─" * 69)
        for name, mem_bytes, count in mem_ops[:10]:
            lines.append(f"{name[:50]:<50} {mem_bytes / 1e6:>10.1f} {count:>7d}")
        lines.append("")

    # ── BOTTLENECK ASSESSMENT ────────────────────────────────────────
    lines.append("─── BOTTLENECK ASSESSMENT " + "─" * 46)
    lines.append("")

    comm_us = category_totals.get("Communication", 0.0)
    compute_us = category_totals.get("Compute", 0.0)

    if total_leaf_cuda_us > 0:
        comm_pct = _pct(comm_us, total_leaf_cuda_us)
        compute_pct = _pct(compute_us, total_leaf_cuda_us)

        if comm_pct > 20.0:
            lines.append(
                f"COMMUNICATION-BOUND: {comm_pct:.0f}% of CUDA time in collectives "
                f"({_us_to_ms(comm_us):.0f}ms). TP all-reduce is the dominant cost."
            )
        if compute_pct < 50.0:
            lines.append(
                f"COMPUTE UNDERUTILIZED: Only {compute_pct:.0f}% of CUDA time in compute. "
                f"Increase batch size or reduce communication."
            )

    if peak_gb > 56.0:  # 70% of 80
        lines.append(
            f"MEMORY PRESSURE: {peak_gb:.1f} GB peak. "
            f"Consider activation recompute/offload to enable larger batch."
        )

    if has_mimo:
        # Check if encoder time is significant
        enc_us = mimo_phases.get("mimo::encoder_forward", {}).get("cuda_time_us", 0)
        llm_us = mimo_phases.get("mimo::llm_forward", {}).get("cuda_time_us", 0)
        if enc_us > 0 and llm_us > 0:
            enc_frac = _pct(enc_us, enc_us + llm_us)
            lines.append(f"ENCODER/LLM SPLIT: encoder={enc_frac:.0f}%, llm={100-enc_frac:.0f}% of model compute")

        comm_us_in_bridge = mimo_phases.get("mimo::bridge_communicate", {}).get("cuda_time_us", 0)
        if comm_us_in_bridge < 1000:  # < 1ms
            lines.append(f"BRIDGE OVERHEAD: {_us_to_ms(comm_us_in_bridge):.2f}ms — negligible")

    if memory_waterfall:
        enc_mem = memory_waterfall.get("mimo::encoder_forward", 0) / (1024**3)
        llm_mem = memory_waterfall.get("mimo::llm_forward", 0) / (1024**3)
        if enc_mem > 5.0 and llm_mem > enc_mem:
            lines.append(
                f"OFFLOAD OPPORTUNITY: {enc_mem:.1f} GB encoder activations held through "
                f"LLM forward (peaks at {llm_mem:.1f} GB)"
            )

    lines.append("")
    return "\n".join(lines)


def save_analysis(report: str, filepath: str):
    """Save the analysis report to a file."""
    with open(filepath, "w") as f:
        f.write(report)
