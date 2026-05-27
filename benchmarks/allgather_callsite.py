"""Identify the call site that launches the slow AllGather kernel in iter=34 mb=1
on LLM rank 16. Approach:

1. Find the biggest ncclDevKernel_AllGather_RING_LL kernel in the slow forward.
2. Look up its CUPTI correlation_id → CUPTI_ACTIVITY_KIND_RUNTIME row (the
   cudaLaunchKernel call on the host).
3. Get the wall-clock time of that host launch.
4. Find the NVTX stack active on the host at that moment (PyTorch emit_nvtx
   + our timeline_event + Megatron internal NVTX).
"""

import sqlite3
import sys
from pathlib import Path

SQLITE = sys.argv[1] if len(sys.argv) > 1 else "rank00016.sqlite"
con = sqlite3.connect(SQLITE)
cur = con.cursor()

# 1. Get bounds of schedule.forward/iter=34/mb=1
s, e = cur.execute(
    "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
    ("%schedule.forward/iter=34/mb=1%",),
).fetchone()
print(f"schedule.forward/iter=34/mb=1: [{s}, {e}] dur={(e - s) / 1e6:.1f} ms\n")

# 2. Biggest AllGather kernel inside
ag = cur.execute(
    """
    SELECT k.start, k.end, k.streamId, k.correlationId, sid.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
    WHERE k.start >= ? AND k.end <= ? AND sid.value LIKE '%AllGather%'
    ORDER BY (k.end - k.start) DESC LIMIT 1
    """,
    (s, e),
).fetchone()
ag_s, ag_e, ag_stream, ag_corr, ag_name = ag
print(f"slowest AllGather kernel: stream={ag_stream}  dur={(ag_e - ag_s) / 1e6:.1f} ms  corr_id={ag_corr}")
print(f"  start_ns={ag_s}  end_ns={ag_e}")
print(f"  name={ag_name}\n")

# 3. Find the host-side cudaLaunchKernel that produced it (same correlationId)
runtime = cur.execute(
    """
    SELECT start, end, globalTid, nameId
    FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?
    """,
    (ag_corr,),
).fetchone()
if runtime:
    rt_s, rt_e, rt_tid, rt_nameid = runtime
    rt_name = cur.execute("SELECT value FROM StringIds WHERE id=?", (rt_nameid,)).fetchone()
    print(f"host launch call: {rt_name[0]}  at {rt_s}..{rt_e} ({(rt_e - rt_s) / 1e6:.3f} ms host)")
    print(f"  host launch globalTid={rt_tid}")
    print(f"  kernel-to-host launch lag = {(ag_s - rt_s) / 1e6:.1f} ms (kernel ran this much after launch)")

    # 4. Find the NVTX ranges active at the launch time
    nvtx = cur.execute(
        """
        SELECT start, end, text
        FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND eventType IN (59, 60, 71, 72) AND text IS NOT NULL
        ORDER BY start DESC
        """,
        (rt_s, rt_s),
    ).fetchall()
    # Filter to ranges that actually surround
    surrounding = [r for r in nvtx if r[1] is not None and r[0] <= rt_s <= r[1]]
    print(f"\nNVTX stack at launch time ({rt_s} ns) — innermost first:")
    # Sort by smallest (innermost) first
    surrounding.sort(key=lambda r: r[1] - r[0])
    for st, en, text in surrounding[:30]:
        print(f"  [{(en - st) / 1e6:8.2f} ms]  {text}")
else:
    print("no host runtime record for that correlationId")

# Bonus: count how many AllGather kernels are launched in this fwd window
ag_count = cur.execute(
    """
    SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
    WHERE k.start >= ? AND k.end <= ? AND sid.value LIKE '%AllGather%'
    """,
    (s, e),
).fetchone()[0]
print(f"\ntotal AllGather kernels in this fwd window: {ag_count}")

# Show ALL AllGather kernels with their durations to see distribution
all_ag = cur.execute(
    """
    SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
    WHERE k.start >= ? AND k.end <= ? AND sid.value LIKE '%AllGather%'
    ORDER BY (k.end - k.start) DESC LIMIT 10
    """,
    (s, e),
).fetchall()
print(f"top 10 AllGather durations (ms): {[f'{(e-s)/1e6:.1f}' for s, e in all_ag]}")
