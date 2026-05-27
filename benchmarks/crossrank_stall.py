"""For iter=38 mb=1, get the biggest dtoh stall and the AllGather kernel
start/end times. Compare across ranks to identify the slow peer."""

import sqlite3
import sys
from pathlib import Path

NSYS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TARGETS = ["iter=38/mb=1", "iter=34/mb=1", "iter=39/mb=0", "iter=30/mb=2"]


def analyze_rank(sqlite_path: Path, target_label: str):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    # 1) Find the schedule.forward window for this iter/mb
    fwd = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL ORDER BY start",
        (f"%schedule.forward/{target_label}%",),
    ).fetchone()
    if not fwd:
        return None
    s, e = fwd
    fwd_ms = (e - s) / 1e6
    # 2) Biggest sync (host stall) in the window
    big = cur.execute(
        """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        WHERE start >= ? AND end <= ? ORDER BY (end - start) DESC LIMIT 1
        """,
        (s, e),
    ).fetchone()
    if not big:
        con.close()
        return {"fwd_ms": fwd_ms, "stall_ms": 0.0}
    bs, be = big
    stall_ms = (be - bs) / 1e6
    # 3) The AllGather kernel overlapping this sync window
    ag = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
        WHERE k.start < ? AND k.end > ? AND s.value LIKE '%AllGather%'
        ORDER BY (k.end - k.start) DESC LIMIT 1
        """,
        (be, bs),
    ).fetchone()
    con.close()
    out = {"fwd_ms": fwd_ms, "stall_ms": stall_ms, "stall_start_ns": bs, "stall_end_ns": be}
    if ag:
        ag_s, ag_e, ag_stream, ag_name = ag
        out.update(
            {"ag_start_ns": ag_s, "ag_end_ns": ag_e, "ag_ms": (ag_e - ag_s) / 1e6,
             "ag_stream": ag_stream, "ag_name": ag_name[:40]}
        )
    return out


for target in TARGETS:
    print(f"\n========= {target} =========")
    print(f"{'rank':>4} {'fwd_ms':>8} {'stall_ms':>10} {'ag_ms':>8} {'ag_start_ns':>16} {'ag_end_ns':>16}  ag_name")
    rows = []
    for p in sorted(NSYS_DIR.glob("rank000*.sqlite")):
        rank = int(p.stem.replace("rank", ""))
        r = analyze_rank(p, target)
        if r is None:
            continue
        rows.append((rank, r))
        ag_s = r.get("ag_start_ns", 0)
        ag_e = r.get("ag_end_ns", 0)
        ag_ms = r.get("ag_ms", 0)
        ag_name = r.get("ag_name", "")
        print(f"{rank:>4} {r['fwd_ms']:>8.0f} {r['stall_ms']:>10.1f} {ag_ms:>8.1f} {ag_s:>16} {ag_e:>16}  {ag_name}")
    # cross-rank skew analysis
    if rows:
        ag_starts = [r["ag_start_ns"] for _, r in rows if "ag_start_ns" in r]
        ag_ends = [r["ag_end_ns"] for _, r in rows if "ag_end_ns" in r]
        if ag_starts and ag_ends:
            print(f"  AG_start skew: max-min = {(max(ag_starts) - min(ag_starts)) / 1e6:.1f} ms")
            print(f"  AG_end alignment: max-min = {(max(ag_ends) - min(ag_ends)) / 1e6:.1f} ms (expect ~0 if collective)")
            slow_rank = max(rows, key=lambda r: r[1].get("ag_start_ns", 0))[0]
            fast_rank = min(rows, key=lambda r: r[1].get("ag_start_ns", 0))[0]
            print(f"  slowest peer arriving at AG: rank {slow_rank}")
            print(f"  fastest peer at AG:           rank {fast_rank}")
