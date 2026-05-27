"""Scan all GPU-idle gaps > 40 ms on the compute stream (stream 7) across
all 16 LLM ranks and all profiled iters/microbatches. For each gap, find
the innermost user NVTX range active at the gap midpoint, which gives
(iter, mb, layer-level context)."""

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
    rows = [r for r in rows if r[0] <= t <= r[1]]
    return rows


def short_layer(stack):
    """Find the first NVTX frame that looks like a transformer-layer-level operation."""
    interesting = (
        "_forward_mlp", "_forward_attention", "self_attention",
        "core_attention", "MambaSplit", "transformer_layer.forward",
        "_GatherFromSequenceParallel", "_ReduceFromTensorParallel",
        "_ScatterToSequenceParallel", "_AllToAll", "permute", "unpermute",
        "linear_with_grad_accumulation", "_LayerNormLinear",
        "language_model_embedding", "RotaryEmbedding",
        "moe_layer", "TopK", "Router", "Linear",
    )
    schedule = (
        "schedule.forward", "schedule.backward", "moe.dtoh_sync",
        "moe.a2a_dispatch", "moe.a2a_combine", "bridge.recv_forward",
        "bridge.send_backward_recv_forward",
    )
    layer_frame = None
    schedule_frame = None
    for s_, e_, text in stack[:50]:
        if any(x in text for x in interesting):
            if layer_frame is None:
                layer_frame = (e_ - s_, text)
        if any(x in text for x in schedule):
            if schedule_frame is None:
                schedule_frame = text
    return layer_frame, schedule_frame


print(f"{'rank':>4} {'iter':>4} {'mb':>2} {'gap_ms':>7} {'gap_start_ns':>14}  schedule  |  innermost-layer-NVTX")
print("-" * 160)
all_gaps = []
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
            # Get kernels on stream 7 in this window
            ks = cur.execute(
                "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = 7 AND start >= ? AND end <= ? ORDER BY start",
                (ws, we),
            ).fetchall()
            # find gaps
            prev_end = ws
            for k_s, k_e in ks:
                gap_ms = (k_s - prev_end) / 1e6
                if gap_ms >= GAP_THRESHOLD_MS:
                    mid = (prev_end + k_s) // 2
                    stack = nvtx_at(cur, mid)
                    layer, sched = short_layer(stack)
                    layer_text = layer[1][:120] if layer else "(no layer NVTX)"
                    sched_text = sched.split("/role")[0][:50] if sched else "(no sched)"
                    print(f"{rank:>4} {it:>4} {mb:>2} {gap_ms:>7.1f} {prev_end:>14}  {sched_text}  |  {layer_text}")
                    all_gaps.append((rank, it, mb, gap_ms))
                prev_end = max(prev_end, k_e)
    con.close()

# Summary by (iter, mb)
print(f"\n=== Summary: count of GPU-idle gaps > {GAP_THRESHOLD_MS} ms per (iter, mb) ===")
by_im = defaultdict(int)
by_im_max = defaultdict(float)
by_im_ranks = defaultdict(set)
for rank, it, mb, gap in all_gaps:
    by_im[(it, mb)] += 1
    by_im_max[(it, mb)] = max(by_im_max[(it, mb)], gap)
    by_im_ranks[(it, mb)].add(rank)
print(f"{'iter':>4} {'mb':>2} {'gaps':>5} {'max_gap_ms':>11}  ranks_with_gaps")
for (it, mb), cnt in sorted(by_im.items()):
    print(f"{it:>4} {mb:>2} {cnt:>5} {by_im_max[(it, mb)]:>11.0f}  {sorted(by_im_ranks[(it, mb)])}")

print(f"\nTotal gaps > {GAP_THRESHOLD_MS} ms across iters 30-40: {len(all_gaps)}")
