"""Improved GPU-idle scan. For each gap, sample NVTX at multiple points
and report ALL layer/op-level NVTX that surround the gap (just before,
inside, just after). Also broadens the "interesting" NVTX list to include
GroupedLinear, MoE permute/unpermute, all autograd Function classes."""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
GAP_THRESHOLD_MS = 40
LLM_RANKS = list(range(8, 24))


def nvtx_at(cur, t):
    rows = cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND text IS NOT NULL ORDER BY (end - start) ASC",
        (t, t),
    ).fetchall()
    return [r for r in rows if r[0] <= t <= r[1]]


def interesting(text):
    INTERESTING_PARTS = (
        "_GroupedLinear", "_Linear", "_LayerNormLinear",
        "_forward_mlp", "_forward_attention", "self_attention",
        "core_attention", "MambaSplit", "transformer_layer.forward",
        "_GatherFromSequenceParallel", "_ReduceFromTensorParallel",
        "_ScatterToSequenceParallel", "_AllToAll", "_permute", "_unpermute",
        "linear_with_grad_accumulation", "language_model_embedding",
        "RotaryEmbedding", "moe_layer", "TopK", "Router",
        "RandomSTE", "moe.dtoh", "moe.a2a", "bridge.recv", "bridge.send",
        "aten::item", "aten::_local_scalar_dense",
    )
    return any(x in text for x in INTERESTING_PARTS)


def innermost_interesting(stack):
    for s, e, text in stack[:50]:
        # skip NCCL plumbing
        if any(x in text for x in ("nccl:_", "record_param_comms", "c10d::", "NCCL", "CCCL")):
            continue
        if interesting(text):
            return (e - s) / 1e6, text
    return None


def pid_of_rank(cur):
    rows = cur.execute(
        "SELECT DISTINCT p.pid FROM PROCESSES p JOIN NVTX_EVENTS n ON (n.globalTid >> 24) = p.pid WHERE n.text LIKE 'iter.total%' LIMIT 1"
    ).fetchall()
    return rows[0][0] if rows else None


# Find which rank PID 225062 corresponds to
print("PID 225062 -> rank mapping:")
for rank in LLM_RANKS:
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    pid = pid_of_rank(cur)
    flag = "  <-- MATCH" if pid == 225062 else ""
    print(f"  rank {rank}: pid {pid}{flag}")
    con.close()
print()

# For iter=38 mb=1, dump ALL gaps on ALL ranks with full NVTX attribution at three sample points
IT, MB = 38, 1
print(f"=== iter={IT} mb={MB} — all GPU-idle gaps >= {GAP_THRESHOLD_MS} ms per rank ===")
print(f"{'rank':>4} {'gap_ms':>7} {'gap_start_ns':>14}  nvtx_at_start | nvtx_at_mid | nvtx_at_end")
print("-" * 180)
for rank in LLM_RANKS:
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    win = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={IT}/mb={MB}%",),
    ).fetchone()
    if not win:
        con.close()
        continue
    ws, we = win
    ks = cur.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = 7 AND start >= ? AND end <= ? ORDER BY start",
        (ws, we),
    ).fetchall()
    prev_end = ws
    for k_s, k_e in ks:
        gap_ms = (k_s - prev_end) / 1e6
        if gap_ms >= GAP_THRESHOLD_MS:
            # sample at 3 points in the gap
            samples = []
            for off in (0.05, 0.5, 0.95):
                t = int(prev_end + (k_s - prev_end) * off)
                st = nvtx_at(cur, t)
                lbl = innermost_interesting(st)
                samples.append(lbl[1][:55] if lbl else "(no NVTX)")
            print(f"{rank:>4} {gap_ms:>7.1f} {prev_end:>14}  {samples[0]:<55} | {samples[1]:<55} | {samples[2]}")
        prev_end = max(prev_end, k_e)
    con.close()
