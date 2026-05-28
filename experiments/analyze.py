#!/usr/bin/env python3
"""Standardized per-experiment analysis of cw-dfw timeline JSONL data.

Reads the timeline JSONL files for an LLM rank, computes iter.total
distribution, per-microbatch fwd/bwd medians/p90s, the boundary ratios
(mb0.fwd / mb1.fwd, mb_last.bwd / mb(last-1).bwd), and a per-iter
stall-budget attribution for the 8 slowest iters.

Writes results.json with all numbers and a short human summary.

Usage:
    python analyze.py <timeline_dir> [--rank 16] [--from-iter 20]
                      [--to-iter 100] [--nmb 6] [--out results.json]

The timeline dir is expected to contain rank{00000..00xxx}.jsonl files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FWD_FLOOR_MS = 167.0  # LLM per-mb forward compute floor (measured baseline)
BWD_FLOOR_MS = 213.0  # LLM per-mb backward compute floor (measured baseline)


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def load_rank(timeline_dir: Path, rank: int, from_iter: int, to_iter: int):
    """Return dict[iter_idx] -> dict[event_key] -> aggregated_ms."""
    path = timeline_dir / f"rank{rank:05d}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    per_iter = defaultdict(lambda: defaultdict(float))
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            it = e.get("iteration")
            if it is None or it < from_iter or it > to_iter:
                continue
            name = e.get("event", "")
            dur = e.get("duration_us")
            if dur is None:
                continue
            mb = e.get("microbatch", e.get("mb"))
            key = name + (f".mb{mb}" if mb is not None else "")
            per_iter[it][key] += dur / 1000.0
    return per_iter


def distribution(values):
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "p10": pct(values, 0.1),
        "p50": statistics.median(values),
        "p90": pct(values, 0.9),
        "p99": pct(values, 0.99),
        "max": max(values),
        "stdev": statistics.pstdev(values),
        "sum": sum(values),
    }


def stall_budget_for_iter(L, nmb: int):
    """Decompose one iter's stall budget into named contributors."""
    tot = L.get("iter.total", 0)
    br_rf = L.get("bridge.recv_forward.mb0", 0)
    lm_dn = sum(L.get(f"data.next.mb{m}", 0) for m in range(nmb))
    dtoh = sum(L.get(f"moe.dtoh_sync.mb{m}", 0) for m in range(nmb))
    mb0_fwd = L.get("schedule.forward.mb0", 0)
    mb_last_bwd = L.get(f"schedule.backward.mb{nmb-1}", 0)
    mid_fwd = sum(L.get(f"schedule.forward.mb{m}", 0) for m in range(1, nmb))
    mid_bwd = sum(L.get(f"schedule.backward.mb{m}", 0) for m in range(nmb - 1))
    dn_mb0 = L.get("data.next.mb0", 0)
    sum_mb = mb0_fwd + mid_fwd + mid_bwd + mb_last_bwd
    return {
        "iter_total": tot,
        "stall_budget": tot - (nmb * FWD_FLOOR_MS + nmb * BWD_FLOOR_MS),
        "encoder_bridge": br_rf,
        "llm_data_next": lm_dn,
        "moe_dtoh_sync": dtoh,
        "mb0_fwd_jitter": mb0_fwd - FWD_FLOOR_MS - br_rf - dn_mb0,
        "mb_last_bwd_jitter": mb_last_bwd - BWD_FLOOR_MS,
        "middle_fwd_jitter": mid_fwd - (nmb - 1) * FWD_FLOOR_MS - (lm_dn - dn_mb0),
        "middle_bwd_jitter": mid_bwd - (nmb - 1) * BWD_FLOOR_MS,
        "outside_mb": tot - sum_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline_dir", type=Path)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--from-iter", type=int, default=20)
    ap.add_argument("--to-iter", type=int, default=100)
    ap.add_argument("--nmb", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    args = ap.parse_args()

    per_iter = load_rank(args.timeline_dir, args.rank, args.from_iter, args.to_iter)
    iters = sorted(per_iter.keys())

    iter_totals = [per_iter[it].get("iter.total", 0) for it in iters]
    iter_dist = distribution([v for v in iter_totals if v > 0])

    per_mb = {}
    for kind in ("schedule.forward", "schedule.backward"):
        per_mb[kind] = {}
        for m in range(args.nmb):
            key = f"{kind}.mb{m}"
            vs = [per_iter[it].get(key, 0) for it in iters if per_iter[it].get(key, 0) > 0]
            per_mb[kind][f"mb{m}"] = distribution(vs)

    last_mb = args.nmb - 1
    boundary = {
        "mb0_fwd_p50": per_mb["schedule.forward"]["mb0"].get("p50", 0),
        "mb1_fwd_p50": per_mb["schedule.forward"]["mb1"].get("p50", 0),
        "mb0_fwd_over_mb1_fwd_p50": (
            per_mb["schedule.forward"]["mb0"].get("p50", 0)
            / max(1, per_mb["schedule.forward"]["mb1"].get("p50", 0))
        ),
        "mb_last_bwd_p50": per_mb["schedule.backward"][f"mb{last_mb}"].get("p50", 0),
        "mb_last_minus_1_bwd_p50": per_mb["schedule.backward"][f"mb{last_mb-1}"].get("p50", 0),
        "mb_last_bwd_over_mb_last_minus_1_bwd_p50": (
            per_mb["schedule.backward"][f"mb{last_mb}"].get("p50", 0)
            / max(1, per_mb["schedule.backward"][f"mb{last_mb-1}"].get("p50", 0))
        ),
    }

    moe_dtoh_sum = [
        sum(per_iter[it].get(f"moe.dtoh_sync.mb{m}", 0) for m in range(args.nmb)) for it in iters
    ]
    outside_mb = [
        per_iter[it].get("iter.total", 0)
        - sum(per_iter[it].get(f"schedule.forward.mb{m}", 0) for m in range(args.nmb))
        - sum(per_iter[it].get(f"schedule.backward.mb{m}", 0) for m in range(args.nmb))
        for it in iters
    ]

    # 8-slowest stall budget
    iter_rows = [(it, per_iter[it].get("iter.total", 0)) for it in iters]
    iter_rows.sort(key=lambda x: x[1])
    slowest = [stall_budget_for_iter(per_iter[it], args.nmb) for it, _ in iter_rows[-8:]]
    fastest = [stall_budget_for_iter(per_iter[it], args.nmb) for it, _ in iter_rows[:5]]

    def avg_attribution(rows):
        out = {}
        for key in (
            "iter_total",
            "stall_budget",
            "encoder_bridge",
            "llm_data_next",
            "moe_dtoh_sync",
            "mb0_fwd_jitter",
            "mb_last_bwd_jitter",
            "middle_fwd_jitter",
            "middle_bwd_jitter",
            "outside_mb",
        ):
            out[key] = sum(r[key] for r in rows) / max(1, len(rows))
        return out

    results = {
        "timeline_dir": str(args.timeline_dir),
        "rank": args.rank,
        "iter_window": [args.from_iter, args.to_iter],
        "nmb": args.nmb,
        "iter_total": iter_dist,
        "per_mb": per_mb,
        "boundary_ratios": boundary,
        "moe_dtoh_sync_sum": distribution([v for v in moe_dtoh_sum if v > 0]),
        "outside_mb": distribution([v for v in outside_mb if v >= 0]),
        "stall_attribution_8_slowest_avg": avg_attribution(slowest),
        "stall_attribution_5_fastest_avg": avg_attribution(fastest),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Print a short human summary to stdout.
    print(f"iter.total p50={iter_dist.get('p50',0):.0f}ms p90={iter_dist.get('p90',0):.0f}ms "
          f"p99={iter_dist.get('p99',0):.0f}ms stdev={iter_dist.get('stdev',0):.0f}ms")
    print(f"mb0.fwd p50={boundary['mb0_fwd_p50']:.0f}ms  "
          f"(vs mb1.fwd {boundary['mb1_fwd_p50']:.0f}ms = "
          f"{boundary['mb0_fwd_over_mb1_fwd_p50']:.2f}x)")
    print(f"mb{last_mb}.bwd p50={boundary['mb_last_bwd_p50']:.0f}ms  "
          f"(vs mb{last_mb-1}.bwd {boundary['mb_last_minus_1_bwd_p50']:.0f}ms = "
          f"{boundary['mb_last_bwd_over_mb_last_minus_1_bwd_p50']:.2f}x)")
    print(f"moe.dtoh sum p50={results['moe_dtoh_sync_sum'].get('p50',0):.0f}ms")
    print(f"outside-mb p50={results['outside_mb'].get('p50',0):.0f}ms")
    print()
    print("8-slowest avg stall-budget attribution:")
    s = results["stall_attribution_8_slowest_avg"]
    print(f"  iter_total={s['iter_total']:.0f}  stall_budget={s['stall_budget']:.0f}")
    for k in ("mb_last_bwd_jitter", "mb0_fwd_jitter", "moe_dtoh_sync", "outside_mb",
              "middle_fwd_jitter", "middle_bwd_jitter", "encoder_bridge", "llm_data_next"):
        print(f"  {k:>22s} = {s[k]:+8.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
