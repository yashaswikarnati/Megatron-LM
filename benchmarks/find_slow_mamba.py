"""Find which rank has the slow MambaSplitConv1dScanCombined kernel in
iter=34 mb=1 (and what its duration is). Also report Python process PID
per rank to map PID 225143 -> rank.

In nsys's NVTX records the Python autograd Function class names appear via
emit_nvtx. MambaSplitConv1dScanCombinedFn is the autograd Function for the
fused Mamba scan kernel.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")

# iter=34 mb=1 schedule.forward window on rank 16: [16273135594, 17354890584]
# Across ranks, fwd windows should differ slightly but we want all mamba NVTX
# events inside the iter=34 mb=1 boundaries on each rank.


def analyze_rank(rank):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        return None
    con = sqlite3.connect(p)
    cur = con.cursor()
    # Find the iter=34 mb=1 fwd window on this rank
    win = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        ("%schedule.forward/iter=34/mb=1%",),
    ).fetchone()
    if not win:
        con.close()
        return None
    s, e = win
    # Find all MambaSplitConv1dScanCombined NVTX ranges in this window
    mamba = cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE ? AND start >= ? AND end <= ? ORDER BY (end - start) DESC LIMIT 5",
        ("%MambaSplitConv1dScanCombined%", s, e),
    ).fetchall()
    # Find globalTid for the iter.total emission (training process)
    main_tid = cur.execute(
        "SELECT globalTid FROM NVTX_EVENTS WHERE text LIKE 'iter.total%' LIMIT 1"
    ).fetchone()
    # Try to find OS pid via TARGET_INFO_SYSTEM_ENV or similar
    pid_row = cur.execute(
        "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='HOSTNAME' OR name LIKE '%PID%' OR name LIKE '%pid%' LIMIT 5"
    ).fetchall()
    # Try ThreadNames
    threads = cur.execute(
        "SELECT DISTINCT globalTid, nameId FROM ThreadNames LIMIT 20"
    ).fetchall()
    con.close()
    return {
        "fwd_window": (s, e),
        "fwd_ms": (e - s) / 1e6,
        "top_mamba": mamba,
        "main_tid": main_tid[0] if main_tid else None,
        "thread_count": len(threads),
    }


print(f"iter=34 mb=1 — MambaSplitConv1dScanCombined max duration per LLM rank:")
print(f"{'rank':>4} {'fwd_ms':>7} {'main_globalTid':>16} {'mamba_max_ms':>13}  mamba_op_info")
print("-" * 110)
for rank in range(8, 24):
    r = analyze_rank(rank)
    if r is None:
        continue
    if r["top_mamba"]:
        top_s, top_e, top_text = r["top_mamba"][0]
        dur_ms = (top_e - top_s) / 1e6
        # extract seq= and sizes from text
        op_short = top_text[:90]
        flag = "  <-- SPIKE" if dur_ms > 50 else ""
        print(f"{rank:>4} {r['fwd_ms']:>7.0f} {str(r['main_tid']):>16} {dur_ms:>13.1f}  {op_short}{flag}")
    else:
        print(f"{rank:>4} {r['fwd_ms']:>7.0f}  no mamba ranges in mb=1")
