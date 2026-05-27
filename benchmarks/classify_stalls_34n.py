"""Comprehensive 34n stall classifier — same logic as classify_stalls_3n.py
but on the 34n trace (ranks 16-23, 8 ranks profiled out of 256 LLM ranks).
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs768-34n-PG1-GR1/297782/nsys")
TIMELINE_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs768-34n-PG1-GR1/297782/timeline")
GAP_THRESHOLD_MS = 40
LLM_RANKS = list(range(16, 24))  # only profiled ranks
PROFILE_ITERS = list(range(30, 41))


def nvtx_at(cur, t):
    rows = cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND text IS NOT NULL ORDER BY (end - start) ASC",
        (t, t),
    ).fetchall()
    return [r for r in rows if r[0] <= t <= r[1]]


def innermost_useful(stack):
    INTERESTING = (
        "_GroupedLinear", "_Linear", "_LayerNormLinear",
        "_forward_mlp", "_forward_attention", "self_attention",
        "core_attention", "MambaSplit", "transformer_layer.forward",
        "_GatherFromSequenceParallel", "_ReduceFromTensorParallel",
        "_ScatterToSequenceParallel", "_AllToAll", "_permute", "_unpermute",
        "language_model_embedding", "RotaryEmbedding",
        "moe_layer", "TopK", "Router", "RandomSTE",
        "moe.dtoh", "moe.a2a", "bridge.recv", "bridge.send",
        "aten::item", "aten::_local_scalar_dense",
    )
    for s, e, text in stack[:50]:
        if any(x in text for x in ("nccl:_", "record_param_comms", "c10d::", "NCCL", "CCCL")):
            continue
        if any(x in text for x in INTERESTING):
            return text
    return None


def short(text):
    if text is None:
        return "?"
    return text.split(",")[0].split("(")[0].split("/")[0][:50]


print("Collecting gaps from all profiled ranks (16-23)…")
gaps_per_imb = defaultdict(list)
fwd_per_imb = defaultdict(list)  # (iter, mb) -> [(rank, fwd_ms)]
ag_per_imb = defaultdict(list)   # (iter, mb) -> [(rank, max_AG_ms)]
for rank in LLM_RANKS:
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    for it in PROFILE_ITERS:
        for mb in range(6):  # 34n has 6 microbatches (GBS=768 / DP=128)
            win = cur.execute(
                "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
                (f"%schedule.forward/iter={it}/mb={mb}%",),
            ).fetchone()
            if not win:
                continue
            ws, we = win
            fwd_per_imb[(it, mb)].append((rank, (we - ws) / 1e6))
            # max AG kernel
            ag = cur.execute(
                """
                SELECT MAX(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds sid ON k.demangledName = sid.id
                WHERE sid.value LIKE '%AllGather%' AND k.start >= ? AND k.end <= ?
                """,
                (ws, we),
            ).fetchone()
            ag_per_imb[(it, mb)].append((rank, (ag[0] or 0) / 1e6))
            ks = cur.execute(
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = 7 AND start >= ? AND end <= ? ORDER BY start",
                (ws, we),
            ).fetchall()
            prev_end = ws
            for k_s, k_e in ks:
                gap_ms = (k_s - prev_end) / 1e6
                if gap_ms >= GAP_THRESHOLD_MS:
                    mid = (prev_end + k_s) // 2
                    label = innermost_useful(nvtx_at(cur, mid))
                    gaps_per_imb[(it, mb)].append((rank, prev_end, k_s, gap_ms, label))
                prev_end = max(prev_end, k_e)
    con.close()


# Cluster by overlap
def cluster_gaps(gaps):
    if not gaps:
        return []
    gaps = sorted(gaps, key=lambda g: g[1])
    clusters = [[gaps[0]]]
    for g in gaps[1:]:
        c_min = min(c[1] for c in clusters[-1])
        c_max = max(c[2] for c in clusters[-1])
        if g[1] <= c_max and g[2] >= c_min:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    return clusters


print("\n=== 34n STALL CATALOG (iters 30-40) — profiled ranks 16-23 ===")
print(f"{'iter':>4} {'mb':>2} {'type':>11} {'#ranks':>7} {'max_ms':>7}  ranks (NVTX)")
print("-" * 130)

stall_summary = defaultdict(int)
true_stalls = []
cascades = []
for (it, mb), gaps in sorted(gaps_per_imb.items()):
    clusters = cluster_gaps(gaps)
    for cluster in clusters:
        ranks_involved = sorted(set(g[0] for g in cluster))
        max_gap = max(g[3] for g in cluster)
        labels = [g[4] for g in cluster if g[4]]
        label_summary = {}
        for label in labels:
            s = short(label)
            label_summary[s] = label_summary.get(s, 0) + 1
        label_str = ", ".join(f"{k}({v})" for k, v in sorted(label_summary.items(), key=lambda x: -x[1])[:3])
        n = len(ranks_involved)
        # at 34n we only see 8 of 256 ranks; treat 6+/8 as CASCADE proxy
        if n >= 6:
            cls = "CASCADE"
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {label_str}")
            cascades.append((it, mb, n, max_gap, label_str))
        elif n <= 2:
            cls = "TRUE_STALL"
            rs = [f"r{g[0]}={g[3]:.0f}[{short(g[4])}]" for g in cluster]
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {' '.join(rs)}")
            true_stalls.append((it, mb, ranks_involved, max_gap, label_str))
        else:
            cls = "MEDIUM"
            rs = [f"r{g[0]}={g[3]:.0f}" for g in cluster]
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {' '.join(rs)} [{label_str}]")
        stall_summary[cls] += 1

print("\n=== Histogram ===")
for k, v in sorted(stall_summary.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")


# Per-(iter, mb) summary: fwd, AG, gap signatures
print("\n=== 34n per-(iter, mb) summary ===")
print(f"{'iter':>4} {'mb':>2} {'fwd_min':>8} {'fwd_max':>8} {'fwd_spread':>11} {'AG_max':>8}")
for (it, mb) in sorted(fwd_per_imb):
    fwds = [f for _, f in fwd_per_imb[(it, mb)]]
    ags = [a for _, a in ag_per_imb[(it, mb)]]
    if not fwds:
        continue
    print(f"{it:>4} {mb:>2} {min(fwds):>8.0f} {max(fwds):>8.0f} {max(fwds)-min(fwds):>11.0f} {max(ags) if ags else 0:>8.0f}")


print("\n=== TRUE_STALL details ===")
for it, mb, ranks, dur, ctx in true_stalls:
    print(f"  iter={it} mb={mb}  ranks={ranks}  max_gap={dur:.0f} ms  ctx={ctx}")

print("\n=== CASCADE details ===")
for it, mb, n, dur, ctx in cascades:
    print(f"  iter={it} mb={mb}  {n}/8 ranks  max_gap={dur:.0f} ms  ctx={ctx}")
