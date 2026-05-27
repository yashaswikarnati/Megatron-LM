"""Find which rank is rank 16's TP peer and what it was doing during the
489 ms stall in iter=34 mb=1.

Approach:
1. Get start time of rank 16's slow AllGather kernel.
2. For each candidate peer rank, check if it has a matching AllGather kernel
   whose END time aligns with rank 16's. NCCL collective end times line up
   across all participating ranks. The slow peer has a LATE start, but the
   END time of the collective is the same.
3. Find what NVTX range was active on the peer at the time rank 16's launch
   happened (when rank 16 had already arrived and was waiting).
"""

import sqlite3
from pathlib import Path

NSYS_DIR = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658/nsys")
RANK16 = NSYS_DIR / "rank00016.sqlite"

# Wall-clock window of rank 16's slow AllGather kernel
R16_K_S = 16734877290
R16_K_E = 17223943823
R16_LAUNCH = 16734868238  # host launch time of that kernel


# Step 1: collect on every other rank the AllGather kernel that ends near
# R16_K_E (within +/- 5 ms).
def kernel_in_window(sqlite_path, target_end_ns, name_like="%AllGather%"):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, k.correlationId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE sid.value LIKE ? AND ABS(k.end - ?) < 5000000
        ORDER BY ABS(k.end - ?) ASC LIMIT 3
        """,
        (name_like, target_end_ns, target_end_ns),
    ).fetchall()
    con.close()
    return rows


print(f"rank 16 slow AllGather: start={R16_K_S}  end={R16_K_E}  dur={(R16_K_E - R16_K_S) / 1e6:.1f} ms")
print(f"rank 16 host launch:    {R16_LAUNCH}\n")

print(f"{'rank':>4} {'k_start_ns':>16} {'k_end_ns':>16} {'dur_ms':>8} {'stream':>6}  end_offset_from_r16")
print("-" * 100)
participating = []
for rank in range(8, 24):  # all LLM ranks
    if rank == 16:
        continue
    p = NSYS_DIR / f"rank{rank:05d}.sqlite"
    if not p.exists():
        continue
    rows = kernel_in_window(p, R16_K_E)
    if rows:
        k_s, k_e, stream, corr, name = rows[0]
        delta = (k_e - R16_K_E) / 1e6
        if abs(delta) < 5:
            participating.append((rank, k_s, k_e, stream, corr))
            print(f"{rank:>4} {k_s:>16} {k_e:>16} {(k_e - k_s) / 1e6:>8.1f} {stream:>6}  {delta:+.2f} ms")

# Step 2: find what NVTX is active on the slow peer at the time rank 16 launched.
# The "slow peer" is the one whose kernel STARTED latest (last to arrive at collective).
if participating:
    participating.sort(key=lambda r: -r[1])  # latest start first
    slow_peer_rank, k_s, k_e, stream, corr = participating[0]
    print(f"\nSLOW PEER: rank {slow_peer_rank}")
    print(f"  its AG kernel started {(k_s - R16_K_S) / 1e6:.1f} ms after rank 16's")
    print(f"  meaning rank {slow_peer_rank} was still doing other work for that long")

    peer_path = NSYS_DIR / f"rank{slow_peer_rank:05d}.sqlite"
    pcon = sqlite3.connect(peer_path)
    pcur = pcon.cursor()
    # Find the slow peer's host launch time for its AllGather
    rt = pcur.execute(
        "SELECT start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = ?",
        (corr,),
    ).fetchone()
    peer_launch = rt[0] if rt else k_s
    print(f"  slow peer's host launch:    {peer_launch}")
    print(f"  rank 16's host launch:      {R16_LAUNCH}")
    print(f"  peer launched {(peer_launch - R16_LAUNCH) / 1e6:.1f} ms after rank 16\n")

    # What was the slow peer doing in the window [R16_LAUNCH, peer_launch]?
    # Sample several points in the gap and find NVTX stack at each
    def nvtx_at(cur, t):
        rows = cur.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE start <= ? AND end >= ? AND text IS NOT NULL ORDER BY (end - start) ASC",
            (t, t),
        ).fetchall()
        rows = [r for r in rows if r[0] <= t <= r[1]]
        return rows

    n_samples = 8
    print(f"Slow peer (rank {slow_peer_rank}) NVTX stack at sampled timestamps during the gap:")
    for i in range(n_samples):
        t = R16_LAUNCH + (peer_launch - R16_LAUNCH) * i // (n_samples - 1)
        stack = nvtx_at(pcur, t)
        if not stack:
            print(f"  t={t} (offset {(t - R16_LAUNCH) / 1e6:6.1f} ms): no NVTX")
            continue
        # Pick innermost user frame
        ufr = None
        for st, en, text in stack[:30]:
            if any(s in text for s in ("nccl:_", "record_param_comms", "c10d::", "NCCL", "CCCL")):
                continue
            ufr = (en - st, text)
            break
        if ufr:
            print(f"  t={t} (offset {(t - R16_LAUNCH) / 1e6:6.1f} ms): [{ufr[0] / 1e6:7.2f} ms]  {ufr[1][:110]}")

    # What kernels was the slow peer running in that gap on its compute stream?
    krows = pcur.execute(
        """
        SELECT k.start, k.end, k.streamId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE k.start >= ? AND k.end <= ?
        ORDER BY k.start
        """,
        (R16_LAUNCH, peer_launch),
    ).fetchall()
    print(f"\nSlow peer kernels in the gap [{R16_LAUNCH}, {peer_launch}]: {len(krows)} total")
    # Aggregate kernel time by name family
    from collections import defaultdict
    fam = defaultdict(int)
    by_stream = defaultdict(int)
    for k_s2, k_e2, stream2, name in krows:
        dur = k_e2 - k_s2
        if "AllGather" in name: f = "AllGather"
        elif "SendRecv" in name: f = "SendRecv"
        elif "ReduceScatter" in name: f = "ReduceScatter"
        elif "_attn" in name or "flash" in name.lower(): f = "attention"
        elif "gemm" in name.lower() or "nvjet" in name: f = "gemm"
        elif "mamba" in name.lower(): f = "mamba"
        elif "elementwise" in name.lower(): f = "elementwise"
        elif "_row_id_map" in name: f = "moe_routing"
        elif "_permute_kernel" in name: f = "moe_permute"
        elif "_unpermute_kernel" in name: f = "moe_unpermute"
        else: f = name[:40]
        fam[f] += dur
        by_stream[stream2] += dur
    print(f"kernel time by stream (ms):")
    for s, ns in sorted(by_stream.items(), key=lambda x: -x[1])[:10]:
        print(f"  stream={s:>4}: {ns / 1e6:7.1f} ms")
    print(f"kernel time by family (top 15, ms):")
    for f, ns in sorted(fam.items(), key=lambda x: -x[1])[:15]:
        print(f"  {f:>20}: {ns / 1e6:7.1f} ms")
