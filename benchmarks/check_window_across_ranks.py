"""For a given (lo,hi) ns window, query each rank's sqlite for:
  - GPU kernel time overlapping the window
  - Long cuda*Sync* calls overlapping
  - NVTX context

Confirms whether the "GPU busy" answer is rank-dependent.
"""
import sqlite3
import sys

LO = int(sys.argv[1])
HI = int(sys.argv[2])
for path in sys.argv[3:]:
    print(f"--- {path} ---")
    con = sqlite3.connect(path)
    cur = con.cursor()
    row = cur.execute(
        "SELECT COUNT(*), SUM(MIN(end,?)-MAX(start,?)) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? AND end>?",
        (HI, LO, HI, LO),
    ).fetchone()
    n_k = row[0]
    busy_ns = row[1] or 0
    print(f"  GPU kernels in window: {n_k}  busy {busy_ns/1e6:.1f}ms / {(HI-LO)/1e6:.0f}ms")
    syncs = cur.execute(
        """
        SELECT sid.value, (r.end-r.start), r.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds sid ON r.nameId=sid.id
        WHERE r.start<? AND r.end>? AND sid.value LIKE 'cuda%Sync%' AND (r.end-r.start)>50000000
        ORDER BY (r.end-r.start) DESC LIMIT 3
        """,
        (HI, LO),
    ).fetchall()
    for name, dur, start in syncs:
        print(f"  long sync: {name:<35} dur={dur/1e6:>6.1f}ms  start={start}")
    nv = cur.execute(
        """
        SELECT text FROM NVTX_EVENTS WHERE end IS NOT NULL AND start<? AND end>?
          AND (text LIKE 'schedule.%' OR text LIKE 'mamba.%')
        ORDER BY (end-start) DESC LIMIT 1
        """,
        (HI, LO),
    ).fetchone()
    nv_txt = nv[0] if nv else "(none)"
    print(f"  NVTX context: {nv_txt}")
    print()
