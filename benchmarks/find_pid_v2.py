"""Find rank with main training process PID 225130 via NVTX events.

In nsys, globalTid = (pid << 24) | (tid_low_24bits) on Linux. The pid is the
top bits. Find the rank whose iter.total NVTX events have globalTid with
pid == 225130."""

import sqlite3
from pathlib import Path

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
TARGET_PID = 225130


def decode_pid(globalTid):
    """nsys packs (pid, tid) into globalTid. Heuristic: high 40 bits = pid << 24, low 24 = tid."""
    return globalTid >> 24


for rank in range(8, 24):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE text LIKE 'iter.total%' LIMIT 5"
    ).fetchall()
    pids = set(decode_pid(r[0]) for r in rows)
    matches = TARGET_PID in pids
    print(f"rank {rank:>2}  iter.total globalTids -> pids: {pids}  {'<-- MATCH' if matches else ''}")
    con.close()
