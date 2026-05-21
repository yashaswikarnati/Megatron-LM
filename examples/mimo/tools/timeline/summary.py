"""Summarize rank-local JSONL timeline traces into steady-state stats.

Reads every ``rank*.jsonl`` file in the given directory, filters by iteration
window (default: drop first 50), and prints markdown tables:

  1. Per-event distribution (median / p90 / max) across all ranks in window
  2. Per-rank encoder.forward stats (host vs CUDA event time if available)
  3. Per-rank image-count distribution (data-imbalance signal)
  4. Per-rank encoder.data_next vs encoder.forward — stall ratio

Stdlib only. No pandas / numpy / plotly required.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_records(timeline_dir, from_iter, to_iter):
    rank_records = defaultdict(list)
    files = sorted(glob.glob(str(Path(timeline_dir) / "rank*.jsonl")))
    if not files:
        raise SystemExit(f"no rank*.jsonl files in {timeline_dir}")
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                it = rec.get("iteration")
                if it is None:
                    continue
                if from_iter is not None and it < from_iter:
                    continue
                if to_iter is not None and it > to_iter:
                    continue
                rank_records[int(rec["rank"])].append(rec)
    return rank_records, files


def _table(rows, headers):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline_dir", help="Directory containing rank*.jsonl")
    ap.add_argument("--from-iter", type=int, default=50,
                    help="Drop iters < N (steady-state window start, default 50)")
    ap.add_argument("--to-iter", type=int, default=None,
                    help="Drop iters > N (default: no cap)")
    ap.add_argument("--top-events", type=int, default=20,
                    help="How many event names to include in distribution table")
    args = ap.parse_args()

    rank_records, files = _load_records(args.timeline_dir, args.from_iter, args.to_iter)
    if not rank_records:
        raise SystemExit(
            f"no events in iter window [{args.from_iter}, {args.to_iter}]"
        )

    ranks = sorted(rank_records.keys())
    all_records = [r for rank in ranks for r in rank_records[rank]]
    sample = all_records[0]
    iters = sorted({r["iteration"] for r in all_records})

    print(f"# Timeline summary — {args.timeline_dir}")
    print()
    print(f"- ranks: {len(ranks)} (rank ids: {ranks[0]}..{ranks[-1]})")
    print(f"- iter window: [{iters[0]}, {iters[-1]}] ({len(iters)} iters)")
    print(f"- total events: {len(all_records)}")
    md = {k: sample.get(k) for k in ("role", "encoder_dp", "llm_dp", "lanes_per_encoder")
          if k in sample}
    if md:
        print(f"- run metadata: {md}")
    print()

    # 1) Per-event distribution
    by_event_host_us = defaultdict(list)
    by_event_cuda_ms = defaultdict(list)
    for rec in all_records:
        ev = rec["event"]
        if "duration_us" in rec:
            by_event_host_us[ev].append(rec["duration_us"])
        if "cuda_ms" in rec:
            by_event_cuda_ms[ev].append(rec["cuda_ms"])

    event_order = sorted(by_event_host_us.keys(),
                         key=lambda e: -statistics.median(by_event_host_us[e]))
    print("## Per-event host wall-time (ms) — across all ranks × iters in window")
    rows = []
    for ev in event_order[:args.top_events]:
        vs = [v / 1000.0 for v in by_event_host_us[ev]]
        cuda_med = ""
        if ev in by_event_cuda_ms and by_event_cuda_ms[ev]:
            cuda_med = f"{statistics.median(by_event_cuda_ms[ev]):.1f}"
        rows.append([
            ev, len(vs),
            f"{statistics.median(vs):.1f}",
            f"{_percentile(vs, 50):.1f}",
            f"{_percentile(vs, 90):.1f}",
            f"{max(vs):.1f}",
            cuda_med,
        ])
    print(_table(rows, ["event", "n", "host_med_ms", "p50", "p90", "max", "cuda_med_ms"]))
    print()

    # 2) Per-rank encoder.forward
    print("## Per-rank encoder.forward (steady-state)")
    rows = []
    for rank in ranks:
        rs = [r for r in rank_records[rank] if r["event"] == "encoder.forward"]
        if not rs:
            continue
        host_ms = [r["duration_us"] / 1000.0 for r in rs]
        cuda_ms = [r["cuda_ms"] for r in rs if "cuda_ms" in r]
        rows.append([
            rank, len(rs),
            f"{statistics.median(host_ms):.1f}",
            f"{_percentile(host_ms, 90):.1f}",
            f"{statistics.median(cuda_ms):.1f}" if cuda_ms else "—",
            f"{_percentile(cuda_ms, 90):.1f}" if cuda_ms else "—",
        ])
    print(_table(rows, ["rank", "n", "host_med", "host_p90", "cuda_med", "cuda_p90"]))
    print()

    # 3) Per-rank data-next vs forward (stall ratio)
    print("## Per-rank data_next vs forward (data-stall ratio)")
    rows = []
    for rank in ranks:
        dnext = [r["duration_us"] / 1000.0 for r in rank_records[rank]
                 if r["event"] == "encoder.data_next"]
        fwd = [r["duration_us"] / 1000.0 for r in rank_records[rank]
               if r["event"] == "encoder.forward"]
        if not dnext or not fwd:
            continue
        d_med = statistics.median(dnext)
        f_med = statistics.median(fwd)
        ratio = d_med / f_med if f_med > 0 else float("inf")
        rows.append([
            rank,
            f"{d_med:.1f}",
            f"{f_med:.1f}",
            f"{ratio:.2f}",
            f"{_percentile(dnext, 90):.1f}",
        ])
    print(_table(rows, ["rank", "data_next_med_ms", "forward_med_ms",
                        "stall_ratio", "data_next_p90_ms"]))
    print()

    # 4) Per-rank image-count distribution
    print("## Per-rank image-count per iter (data-imbalance signal)")
    rows = []
    for rank in ranks:
        imgs = [r.get("image_count", 0) for r in rank_records[rank]
                if r["event"] == "encoder.batch_stats"]
        if not imgs:
            continue
        rows.append([
            rank, len(imgs),
            f"{statistics.mean(imgs):.1f}",
            f"{statistics.median(imgs):.0f}",
            f"{min(imgs)}",
            f"{max(imgs)}",
        ])
    print(_table(rows, ["rank", "n_iters", "mean_imgs", "median_imgs",
                        "min_imgs", "max_imgs"]))
    print()

    # 5) Per-lane fan-out (from encoder.lane_combine)
    lane_records = [r for r in all_records if r["event"] == "encoder.lane_combine"]
    if lane_records:
        print("## Lane-combine per iter (lane fanout)")
        lwi_all = [r["lanes_with_images"] for r in lane_records]
        lc_all = [r["lane_count"] for r in lane_records]
        print(f"- lane_count (configured): {lc_all[0] if lc_all else 'n/a'}")
        print(f"- lanes_with_images mean / median / min / max: "
              f"{statistics.mean(lwi_all):.1f} / "
              f"{statistics.median(lwi_all):.0f} / "
              f"{min(lwi_all)} / {max(lwi_all)}")
        print()


if __name__ == "__main__":
    main()
