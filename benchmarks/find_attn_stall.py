"""For iter=37 mb=3, find the rank with the biggest attention.forward
NVTX stall and verify it's a real GPU-idle stall."""

import sqlite3
from pathlib import Path

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")

IT = 37
MB = 3


def analyze(rank):
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        return None
    con = sqlite3.connect(p)
    cur = con.cursor()
    win = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={IT}/mb={MB}%",),
    ).fetchone()
    if not win:
        con.close()
        return None
    s, e = win
    fwd_ms = (e - s) / 1e6
    # find longest attention-related NVTX in this window
    attns = cur.execute(
        """
        SELECT start, end, text FROM NVTX_EVENTS
        WHERE start >= ? AND end <= ? AND text IS NOT NULL
              AND (text LIKE '%forward_attention%' OR text LIKE '%self_attention%' OR text LIKE '%core_attention%' OR text LIKE '%self_attn%')
        ORDER BY (end - start) DESC LIMIT 3
        """,
        (s, e),
    ).fetchall()
    if not attns:
        return {"fwd_ms": fwd_ms, "attn_ms": 0, "attn_text": None}
    a_s, a_e, a_text = attns[0]
    a_ms = (a_e - a_s) / 1e6
    # compute GPU coverage on stream 7 inside this attention range
    krow = cur.execute(
        """
        SELECT COALESCE(SUM(MIN(k.end, ?) - MAX(k.start, ?)), 0)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE k.start < ? AND k.end > ? AND k.streamId = 7
        """,
        (a_e, a_s, a_e, a_s),
    ).fetchone()
    coverage_ms = (krow[0] or 0) / 1e6
    idle_ratio = 1 - coverage_ms / a_ms if a_ms > 0 else 0
    # Also map PID for this rank — query main NVTX-emitting process
    pids = cur.execute(
        "SELECT DISTINCT p.pid FROM PROCESSES p JOIN NVTX_EVENTS n ON (n.globalTid >> 24) = p.pid WHERE n.text LIKE 'iter.total%'"
    ).fetchall()
    con.close()
    return {
        "fwd_ms": fwd_ms,
        "attn_ms": a_ms,
        "attn_text": a_text[:80],
        "gpu_cov_ms": coverage_ms,
        "idle_pct": idle_ratio * 100,
        "attn_start_ns": a_s,
        "attn_end_ns": a_e,
        "pids": [r[0] for r in pids],
    }


print(f"iter={IT} mb={MB} — attention.forward stall scan across all 16 LLM ranks")
print(f"{'rank':>4} {'fwd_ms':>7} {'attn_ms':>8} {'gpu_cov':>8} {'idle%':>7}  attn_text  (pids)")
print("-" * 130)
results = []
for rank in range(8, 24):
    r = analyze(rank)
    if r is None:
        continue
    results.append((rank, r))
    flag = ""
    if r["attn_ms"] > 100 and r["idle_pct"] > 50:
        flag = "  <-- STALL"
    print(f"{rank:>4} {r['fwd_ms']:>7.0f} {r['attn_ms']:>8.0f} {r['gpu_cov_ms']:>8.1f} {r['idle_pct']:>7.0f}  {r['attn_text']}  pids={r['pids']}{flag}")

# Report rank with biggest stall
if results:
    worst = max(results, key=lambda kv: kv[1]["attn_ms"])
    print(f"\nWorst attention stall: rank {worst[0]}  attn_ms={worst[1]['attn_ms']:.0f}  idle={worst[1]['idle_pct']:.0f}%")
    print(f"  attn_start_ns={worst[1]['attn_start_ns']}  attn_end_ns={worst[1]['attn_end_ns']}")
    print(f"  text: {worst[1]['attn_text']}")
