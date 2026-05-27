"""For every AllGather kernel >= 20 ms in iter=34 on rank 16, find the
NVTX stack at its host launch to identify the call site."""

import sqlite3
import sys
from pathlib import Path

SQLITE = sys.argv[1] if len(sys.argv) > 1 else "rank00016.sqlite"
THRESHOLD_MS = 20.0
ITER = 34

con = sqlite3.connect(SQLITE)
cur = con.cursor()


def innermost_user_nvtx(launch_t):
    """Get top of host NVTX stack at the launch time."""
    rows = cur.execute(
        """
        SELECT start, end, text FROM NVTX_EVENTS
        WHERE start <= ? AND end >= ? AND text IS NOT NULL
        ORDER BY (end - start) ASC
        """,
        (launch_t, launch_t),
    ).fetchall()
    # Filter to actually-bracketing
    rows = [r for r in rows if r[0] <= launch_t <= r[1]]
    return rows


for mb in range(4):
    win = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={ITER}/mb={mb}%",),
    ).fetchone()
    if not win:
        continue
    s, e = win
    print(f"\n=========== iter={ITER} mb={mb}  (fwd dur={(e - s) / 1e6:.0f} ms) ===========")
    slow = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, k.correlationId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE k.start >= ? AND k.end <= ? AND sid.value LIKE '%AllGather%' AND (k.end - k.start) >= ?
        ORDER BY (k.end - k.start) DESC
        """,
        (s, e, int(THRESHOLD_MS * 1e6)),
    ).fetchall()
    print(f"slow AllGather kernels (>= {THRESHOLD_MS} ms): {len(slow)}")
    for k_s, k_e, stream, corr, name in slow:
        dur = (k_e - k_s) / 1e6
        rt = cur.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?",
            (corr,),
        ).fetchone()
        if not rt:
            print(f"  dur={dur:.1f} ms  (no host launch record)")
            continue
        rt_s, rt_e = rt
        stack = innermost_user_nvtx(rt_s)
        # Pick first interesting frames (skip pure NCCL plumbing)
        user_frames = []
        for st, en, text in stack[:20]:
            if any(s in text for s in ("nccl:_", "record_param_comms", "c10d::")):
                continue
            user_frames.append((en - st, text))
            if len(user_frames) >= 6:
                break
        print(f"  dur={dur:.1f} ms  stream={stream}  kernel_s_ns={k_s} kernel_e_ns={k_e}")
        for span_ns, text in user_frames:
            short = text[:120]
            print(f"      [{span_ns / 1e6:8.2f} ms] {short}")
