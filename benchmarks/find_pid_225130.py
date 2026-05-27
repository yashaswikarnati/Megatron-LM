"""Find the rank with PID 225130, then inspect what it was doing during
the time other ranks were already in the 489 ms tp_ep AllGather of mb=1."""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
TARGET_PID = 225130
R16_LAUNCH = 16734868238
R16_K_E = 17223943823

# Step 1: map PID -> rank
print("Step 1: PID -> rank mapping")
print(f"{'rank':>4} {'pid':>10}")
print("-" * 20)
matched_rank = None
for rank in range(8, 24):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    procs = cur.execute("SELECT DISTINCT pid FROM PROCESSES").fetchall()
    pids = [r[0] for r in procs]
    print(f"{rank:>4} {pids}")
    if TARGET_PID in pids:
        matched_rank = rank
    con.close()

if matched_rank is None:
    print(f"\nPID {TARGET_PID} not found in any LLM rank!")
    # Try thread IDs instead — PID in nsys-ui might be a thread tid
    print("\nSearching globalTid for the target...")
    for rank in range(8, 24):
        p = NSYS_DIR / f"rank{rank:05d}.sqlite"
        if not p.exists():
            continue
        con = sqlite3.connect(p)
        cur = con.cursor()
        # NVTX_EVENTS has globalTid; PROCESSES has pid; ThreadNames has tid+name
        rows = cur.execute("SELECT DISTINCT globalTid FROM ThreadNames LIMIT 10").fetchall()
        # encode/decode: globalTid = (pid << 24) | tid in some systems
        # check ThreadNames where the name suggests forward thread
        named = cur.execute("SELECT nameId, globalTid FROM ThreadNames LIMIT 20").fetchall()
        if any(g[1] >> 24 == TARGET_PID >> 24 for g in rows):
            print(f"  rank {rank}: candidate global tid match")
        con.close()
else:
    print(f"\n=> PID {TARGET_PID} corresponds to RANK {matched_rank}")
    p = NSYS_DIR / f"rank{matched_rank:05d}.sqlite"
    con = sqlite3.connect(p)
    cur = con.cursor()

    # Step 2: find what NVTX was active on this rank at the time rank 16 launched the slow AG
    def nvtx_stack(cur, t):
        rows = cur.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND text IS NOT NULL ORDER BY (end - start) ASC",
            (t, t),
        ).fetchall()
        rows = [r for r in rows if r[0] <= t <= r[1]]
        return rows

    print(f"\nNVTX stack on rank {matched_rank} at t=R16_LAUNCH={R16_LAUNCH}:")
    stack = nvtx_stack(cur, R16_LAUNCH)
    for st, en, text in stack[:20]:
        print(f"  [{(en - st) / 1e6:8.2f} ms] {text[:130]}")

    # Step 3: find rank's own AG kernel that matches the 489 ms collective
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE sid.value LIKE '%AllGather%' AND k.streamId = 7
              AND k.end >= ? AND k.end <= ?
        ORDER BY (k.end - k.start) DESC LIMIT 5
        """,
        (R16_K_E - 500_000_000, R16_K_E + 500_000_000),
    ).fetchall()
    print(f"\nRank {matched_rank}'s top AGs ending near rank 16's reference:")
    for k_s, k_e, stream, corr in rows:
        rt = cur.execute("SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?", (corr,)).fetchone()
        launch = rt[0] if rt else k_s
        print(f"  dur={(k_e - k_s) / 1e6:7.1f} ms  start={k_s}  end={k_e}  launch={launch}  end_skew={(k_e - R16_K_E) / 1e6:+.1f} ms")

    # Step 4: best matching AG (close in end time AND long duration) → assume part of the same collective
    # Pick the one whose end is closest to R16_K_E among long AGs
    matching = [r for r in rows if r[1] - r[0] >= 300_000_000]
    if matching:
        best = min(matching, key=lambda r: abs(r[1] - R16_K_E))
        k_s, k_e, stream, corr = best
        rt = cur.execute("SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?", (corr,)).fetchone()
        launch = rt[0]
        print(f"\nThis rank's matching AG: launched at {launch} (rank 16 launched at {R16_LAUNCH})")
        gap_ms = (launch - R16_LAUNCH) / 1e6
        print(f"Launch skew vs rank 16: {gap_ms:+.1f} ms")

        # NVTX stack at sampled points between rank 16's launch and this rank's launch
        print(f"\nWhat rank {matched_rank} was doing in the gap [{R16_LAUNCH}, {launch}]:")
        if launch > R16_LAUNCH:
            for off in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                t = int(R16_LAUNCH + (launch - R16_LAUNCH) * off)
                st = nvtx_stack(cur, t)
                ufr = None
                for s_, e_, txt in st[:30]:
                    if any(x in txt for x in ("nccl:_", "record_param_comms", "c10d::", "NCCL", "CCCL")):
                        continue
                    ufr = (e_ - s_, txt)
                    break
                if ufr:
                    print(f"  t=+{(t - R16_LAUNCH) / 1e6:6.1f} ms: [{ufr[0] / 1e6:7.2f} ms] {ufr[1][:130]}")

            # Kernel coverage on compute stream in gap
            krows = cur.execute(
                """
                SELECT k.start, k.end, sid.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
                WHERE k.start >= ? AND k.end <= ? AND k.streamId = 7
                ORDER BY (k.end - k.start) DESC LIMIT 15
                """,
                (R16_LAUNCH, launch),
            ).fetchall()
            print(f"\nTop kernels on stream 7 (rank {matched_rank}) in this gap:")
            for s_, e_, name in krows:
                print(f"  dur={(e_ - s_) / 1e6:7.2f} ms  {name[:100]}")
    con.close()
