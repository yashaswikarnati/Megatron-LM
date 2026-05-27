"""Comprehensive 3n stall classifier.

For each (iter, mb) in iters 30-40, find all GPU-idle gaps > 40 ms on stream 7
across all 16 LLM ranks, then bucket them by wall-clock window:
- TRUE_STALL: 1-2 ranks idle while others are NOT idle in that window
              (= real compute jitter on those ranks)
- CASCADE:    >=10 ranks idle simultaneously
              (= synchronization point absorbing upstream slowdown)
- MEDIUM:     3-9 ranks (could be partial cascade or multi-rank stall)
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
GAP_THRESHOLD_MS = 40
LLM_RANKS = list(range(8, 24))
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
    # strip details
    s = text.split(",")[0].split("(")[0].split("/")[0]
    return s[:50]


# Gather all gaps per (iter, mb)
print("Collecting gaps from all ranks…")
gaps_per_imb = defaultdict(list)  # (iter, mb) -> list of (rank, gap_start, gap_end, gap_ms, nvtx_label)
for rank in LLM_RANKS:
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    con = sqlite3.connect(p)
    cur = con.cursor()
    for it in PROFILE_ITERS:
        for mb in range(4):
            win = cur.execute(
                "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
                (f"%schedule.forward/iter={it}/mb={mb}%",),
            ).fetchone()
            if not win:
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
                    mid = (prev_end + k_s) // 2
                    label = innermost_useful(nvtx_at(cur, mid))
                    gaps_per_imb[(it, mb)].append((rank, prev_end, k_s, gap_ms, label))
                prev_end = max(prev_end, k_e)
    con.close()


# For each (iter, mb), cluster gaps by overlapping wall-clock window
def cluster_gaps(gaps):
    """Return list of clusters; each cluster is a list of gap tuples whose
    [start, end] intervals overlap with at least one other in the cluster."""
    if not gaps:
        return []
    # sort by gap_start
    gaps = sorted(gaps, key=lambda g: g[1])
    clusters = [[gaps[0]]]
    for g in gaps[1:]:
        # if g overlaps any in the last cluster, add it
        c_min = min(c[1] for c in clusters[-1])
        c_max = max(c[2] for c in clusters[-1])
        if g[1] <= c_max and g[2] >= c_min:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    return clusters


# Classify each cluster
print("\n=== STALL CATALOG (iters 30-40) ===")
print(f"{'iter':>4} {'mb':>2} {'type':>11} {'#ranks':>7} {'max_ms':>7}  ranks (NVTX context)")
print("-" * 130)

stall_summary = defaultdict(int)
true_stalls_detailed = []
cascades_detailed = []

for (it, mb), gaps in sorted(gaps_per_imb.items()):
    clusters = cluster_gaps(gaps)
    for cluster in clusters:
        ranks_involved = sorted(set(g[0] for g in cluster))
        max_gap = max(g[3] for g in cluster)
        # Most common NVTX label
        labels = [g[4] for g in cluster if g[4]]
        label_summary = {}
        for label in labels:
            s = short(label)
            label_summary[s] = label_summary.get(s, 0) + 1
        label_str = ", ".join(f"{k}({v})" for k, v in sorted(label_summary.items(), key=lambda x: -x[1])[:3])
        n = len(ranks_involved)
        if n <= 2:
            cls = "TRUE_STALL"
            # detail of which rank
            rank_strs = []
            for g in cluster:
                rank_strs.append(f"r{g[0]}={g[3]:.0f}ms[{short(g[4])}]")
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {' '.join(rank_strs)}")
            true_stalls_detailed.append((it, mb, ranks_involved, max_gap, label_str))
        elif n >= 10:
            cls = "CASCADE"
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {label_str}")
            cascades_detailed.append((it, mb, n, max_gap, label_str))
        else:
            cls = "MEDIUM"
            rank_strs = []
            for g in cluster:
                rank_strs.append(f"r{g[0]}={g[3]:.0f}")
            print(f"{it:>4} {mb:>2} {cls:>11} {n:>7} {max_gap:>7.0f}  {' '.join(rank_strs)} [{label_str}]")
        stall_summary[cls] += 1


print(f"\n=== Histogram ===")
for k, v in sorted(stall_summary.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\n=== TRUE_STALL details (real per-rank kernel jitter) ===")
for it, mb, ranks, dur, ctx in true_stalls_detailed:
    print(f"  iter={it} mb={mb}  ranks={ranks}  max_gap={dur:.0f} ms  ctx={ctx}")

print(f"\n=== CASCADE details (multi-rank sync points) ===")
for it, mb, n, dur, ctx in cascades_detailed:
    print(f"  iter={it} mb={mb}  {n} ranks  max_gap={dur:.0f} ms  ctx={ctx}")
