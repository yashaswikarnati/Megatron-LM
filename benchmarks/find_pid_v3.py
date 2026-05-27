"""Look at PROCESSES schema and find the actual training process PID per rank."""

import sqlite3
from pathlib import Path

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")

# Step 0: Check the PROCESSES schema
con = sqlite3.connect(NSYS_DIR / "rank00016.sqlite")
cur = con.cursor()
for r in cur.execute("PRAGMA table_info(PROCESSES)").fetchall():
    print("PROCESSES col:", r)
print()
# Show a few rows
print("Sample PROCESSES rows:")
for r in cur.execute("SELECT * FROM PROCESSES LIMIT 5"):
    print(r)
con.close()
print()

# Step 1: find rank where PROCESSES has only ONE pid that matches the main training pid pattern
# Try filtering PROCESSES to rows where the pid generated NVTX events (training process)
for rank in range(8, 24):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    # The training process has many NVTX events. Find the pid whose globalTid emits NVTX
    rows = cur.execute(
        """
        SELECT DISTINCT p.pid, COUNT(n.text) as evt_count
        FROM PROCESSES p
        LEFT JOIN NVTX_EVENTS n ON (n.globalTid >> 24) = (p.pid)
        GROUP BY p.pid
        HAVING evt_count > 1000
        ORDER BY evt_count DESC LIMIT 5
        """
    ).fetchall()
    print(f"rank {rank:>2}: top NVTX-emitting pids: {rows}")
    con.close()
