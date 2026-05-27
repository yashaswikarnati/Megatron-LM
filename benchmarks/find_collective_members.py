"""Find ALL LLM ranks (8..23) that have a long AllGather kernel ending
near rank 16's 489 ms stall. Use a wider window to catch the real collective."""

import sqlite3
from pathlib import Path

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
R16_K_E = 17223943823
R16_K_S = 16734877290


def closest_long_ag(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    # find the kernel >= 100 ms ending closest to R16_K_E
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, (k.end - k.start) AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE sid.value LIKE '%AllGather%' AND (k.end - k.start) >= 100000000
        ORDER BY ABS(k.end - ?) ASC LIMIT 1
        """,
        (R16_K_E,),
    ).fetchall()
    con.close()
    return rows[0] if rows else None


print(f"rank 16 reference: kernel start={R16_K_S}, end={R16_K_E}, dur={(R16_K_E - R16_K_S) / 1e6:.1f} ms\n")
print(f"{'rank':>4} {'k_start_ns':>16} {'k_end_ns':>16} {'dur_ms':>8} {'stream':>6} {'end_offset_ms':>14}")
print("-" * 90)
for rank in range(8, 24):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    r = closest_long_ag(p)
    if not r:
        print(f"{rank:>4}  no long AG found")
        continue
    k_s, k_e, stream, dur = r
    delta = (k_e - R16_K_E) / 1e6
    flag = "" if abs(delta) < 50 else "  <-- different collective"
    print(f"{rank:>4} {k_s:>16} {k_e:>16} {dur / 1e6:>8.1f} {stream:>6} {delta:>14.2f}{flag}")
