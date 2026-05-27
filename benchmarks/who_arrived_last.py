"""Identify the slow peer in rank 16's 489 ms tp_ep AllGather.

Strategy:
- For each of the 16 LLM ranks, find their AllGather kernel that's part of
  the SAME collective as rank 16's slow one. The collective is identified by:
  (a) end times within a tight window of the reference end
  (b) the previous AG (one layer back) end times should also be tightly aligned
      (because the prior layer's collective ended together).

Then for the slow peer:
- Look at the time between the previous AG (last sync point) and this AG launch
  — this gap is the compute that happened on that rank between two consecutive
  layers. The rank with the LONGEST gap is the drifter.
- Sample NVTX and kernel coverage in that gap to see what was slow.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
R16_K_E = 17223943823
R16_K_S = 16734877290
R16_LAUNCH = 16734868238

# fwd window of iter=34 mb=1 on rank 16: 16273135594 - 17354890584
FWD_S = 16273135594
FWD_E = 17354890584


def all_ags_in_window(sqlite_path, ws, we):
    """Return all AllGather kernels on stream=7 within window [ws, we]."""
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE sid.value LIKE '%AllGather%' AND k.streamId = 7 AND k.start >= ? AND k.end <= ?
        ORDER BY k.start
        """,
        (ws, we),
    ).fetchall()
    con.close()
    return rows


# Step 1: for each rank, find AGs on stream 7 in the fwd window and pick
# the one closest in end time to R16_K_E (must be >= 100 ms to be the same big collective).
print("Step 1: For each LLM rank, find the long AG kernel closest in end-time to rank 16's reference (489 ms ending at 17223943823 ns).")
print(f"{'rank':>4} {'k_start_ns':>16} {'k_end_ns':>16} {'dur_ms':>8} {'host_launch':>16} {'end_skew_ms':>12} {'start_skew_ms':>14}")
print("-" * 100)
per_rank = {}
for rank in range(8, 24):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    ags = all_ags_in_window(p, FWD_S, FWD_E)
    # filter to long AGs (>= 100 ms)
    long_ags = [r for r in ags if r[1] - r[0] >= 100_000_000]
    if not long_ags:
        # try wider — any AG that's part of a long collective
        long_ags = [r for r in ags if r[1] - r[0] >= 50_000_000]
    if not long_ags:
        print(f"{rank:>4}  no long AG found")
        continue
    # Pick the one whose end is closest to R16_K_E
    best = min(long_ags, key=lambda r: abs(r[1] - R16_K_E))
    k_s, k_e, corr = best
    # get host launch
    con = sqlite3.connect(p)
    cur = con.cursor()
    rt = cur.execute("SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?", (corr,)).fetchone()
    launch = rt[0] if rt else k_s
    con.close()
    end_skew = (k_e - R16_K_E) / 1e6
    start_skew = (k_s - R16_K_S) / 1e6
    print(f"{rank:>4} {k_s:>16} {k_e:>16} {(k_e - k_s) / 1e6:>8.1f} {launch:>16} {end_skew:>+12.2f} {start_skew:>+14.2f}")
    per_rank[rank] = {
        "k_s": k_s, "k_e": k_e, "launch": launch, "corr": corr, "dur": k_e - k_s,
    }

# Step 2: among ranks whose end is within +/- 50 ms of rank 16, identify the latest launcher
print()
co_collective = {r: d for r, d in per_rank.items() if abs(d["k_e"] - R16_K_E) <= 50_000_000}
print(f"Same-collective candidates (end skew <= 50 ms): {sorted(co_collective.keys())}")
if co_collective:
    latest_rank, latest_data = max(co_collective.items(), key=lambda kv: kv[1]["launch"])
    earliest_rank, earliest_data = min(co_collective.items(), key=lambda kv: kv[1]["launch"])
    print(f"\nEarliest launcher: rank {earliest_rank} at {earliest_data['launch']}")
    print(f"Latest launcher (= slowest peer): rank {latest_rank} at {latest_data['launch']}")
    gap_ms = (latest_data["launch"] - earliest_data["launch"]) / 1e6
    print(f"Skew between earliest and latest launcher: {gap_ms:.1f} ms")

    # Step 3: find the previous AG on the slow rank (one MoE layer back) to bracket
    # the compute that caused the drift.
    if latest_rank != earliest_rank:
        slow_p = NSYS_DIR / f"rank{latest_rank:05d}.sqlite"
        scon = sqlite3.connect(slow_p)
        scur = scon.cursor()
        prev_ag = scur.execute(
            """
            SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
            WHERE sid.value LIKE '%AllGather%' AND k.streamId = 7 AND k.end < ?
            ORDER BY k.end DESC LIMIT 1
            """,
            (latest_data["k_s"],),
        ).fetchone()
        if prev_ag:
            prev_s, prev_e = prev_ag
            inter_layer_gap = (latest_data["k_s"] - prev_e) / 1e6
            print(f"\nSlow rank {latest_rank}'s prior AG ended at {prev_e}; current AG started at {latest_data['k_s']}")
            print(f"Inter-AG gap on slow rank (compute between layers): {inter_layer_gap:.1f} ms")
            # Same gap on the earliest rank
            fast_p = NSYS_DIR / f"rank{earliest_rank:05d}.sqlite"
            fcon = sqlite3.connect(fast_p)
            fcur = fcon.cursor()
            prev_ag_fast = fcur.execute(
                """
                SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
                WHERE sid.value LIKE '%AllGather%' AND k.streamId = 7 AND k.end < ?
                ORDER BY k.end DESC LIMIT 1
                """,
                (earliest_data["k_s"],),
            ).fetchone()
            if prev_ag_fast:
                fast_inter = (earliest_data["k_s"] - prev_ag_fast[1]) / 1e6
                print(f"Inter-AG gap on earliest rank {earliest_rank}: {fast_inter:.1f} ms")
                print(f"Drift accumulated this layer: {inter_layer_gap - fast_inter:.1f} ms")

            # Step 4: NVTX stack at sampled points in the slow rank's inter-AG gap
            def nvtx_stack(cur, t):
                rows = cur.execute(
                    "SELECT start, end, text FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND text IS NOT NULL ORDER BY (end - start) ASC",
                    (t, t),
                ).fetchall()
                rows = [r for r in rows if r[0] <= t <= r[1]]
                return rows

            print(f"\nWhat slow rank {latest_rank} was doing between prev AG (t={prev_e}) and this AG launch (t={latest_data['launch']}):")
            for off_frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                t = int(prev_e + (latest_data["launch"] - prev_e) * off_frac)
                st = nvtx_stack(scur, t)
                if not st:
                    continue
                # pick innermost non-NCCL-plumbing frame
                ufr = None
                for s_, e_, txt in st[:30]:
                    if any(x in txt for x in ("nccl:_", "record_param_comms", "c10d::", "NCCL", "CCCL")):
                        continue
                    ufr = (e_ - s_, txt)
                    break
                if ufr:
                    rel = (t - prev_e) / (latest_data["launch"] - prev_e) * 100
                    print(f"  t={t} ({rel:5.1f}%): [{ufr[0] / 1e6:7.2f} ms] {ufr[1][:120]}")

            # Step 5: kernel coverage in the inter-AG gap on slow rank
            krows = scur.execute(
                """
                SELECT k.start, k.end, k.streamId, sid.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
                WHERE k.start >= ? AND k.end <= ?
                """,
                (prev_e, latest_data["k_s"]),
            ).fetchall()
            print(f"\nKernels on slow rank {latest_rank} in inter-AG gap ({inter_layer_gap:.1f} ms): {len(krows)} kernels")
            fam = defaultdict(int)
            by_stream = defaultdict(int)
            for k_s2, k_e2, stream2, name in krows:
                dur = k_e2 - k_s2
                if "AllGather" in name: f = "AllGather"
                elif "SendRecv" in name: f = "SendRecv"
                elif "ReduceScatter" in name: f = "ReduceScatter"
                elif "AllReduce" in name: f = "AllReduce"
                elif "flash" in name.lower(): f = "attention"
                elif "gemm" in name.lower() or "nvjet" in name: f = "gemm"
                elif "mamba" in name.lower() or "ssm" in name.lower(): f = "mamba"
                elif "elementwise" in name.lower(): f = "elementwise"
                elif "_row_id_map" in name: f = "moe_routing"
                elif "_permute" in name: f = "moe_permute"
                elif "_unpermute" in name: f = "moe_unpermute"
                elif "softmax" in name.lower(): f = "softmax"
                elif "layer_norm" in name.lower() or "rmsnorm" in name.lower(): f = "norm"
                elif "scan" in name.lower(): f = "mamba_scan"
                elif "causal_conv" in name.lower(): f = "mamba_conv"
                else: f = name[:40]
                fam[f] += dur
                by_stream[stream2] += dur
            print("kernel time by stream (ms):")
            for s, ns in sorted(by_stream.items(), key=lambda x: -x[1])[:8]:
                print(f"  stream={s:>4}: {ns / 1e6:7.1f} ms")
            print("kernel time by family (top 15, ms):")
            for f, ns in sorted(fam.items(), key=lambda x: -x[1])[:15]:
                print(f"  {f:>20}: {ns / 1e6:7.1f} ms")
