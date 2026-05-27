"""Compare what runs in a fast forward (iter=33 mb=0, ~449 ms) vs slow
forward (iter=38 mb=1, ~1491 ms) on rank 16. Aggregate kernel/sync time
by kind to see where the extra 1040 ms is spent.
"""

import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

SQLITE = sys.argv[1] if len(sys.argv) > 1 else "rank00016.sqlite"

con = sqlite3.connect(SQLITE)
cur = con.cursor()


def fwd_window(target):
    row = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/{target}%",),
    ).fetchone()
    return row


def kernel_breakdown(s, e):
    """Return (name -> (total_ns_overlap, count, max_dur))."""
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE k.start < ? AND k.end > ?
        """,
        (e, s),
    ).fetchall()
    by_kind = defaultdict(lambda: [0, 0, 0])  # name -> [total_overlap, count, max_dur_ms]
    by_stream = defaultdict(int)
    for k_s, k_e, stream, name in rows:
        overlap = max(0, min(k_e, e) - max(k_s, s))
        if overlap <= 0:
            continue
        # collapse to kernel family for readability
        if "AllGather" in name:
            family = "nccl_AllGather"
        elif "ReduceScatter" in name:
            family = "nccl_ReduceScatter"
        elif "SendRecv" in name:
            family = "nccl_SendRecv"
        elif "AllReduce" in name:
            family = "nccl_AllReduce"
        elif "_row_id_map" in name:
            family = "moe_row_id_map"
        elif "_permute_kernel" in name:
            family = "moe_permute"
        elif "_unpermute_kernel" in name:
            family = "moe_unpermute"
        elif "cublas_gemm" in name or "gemm" in name.lower():
            family = "gemm"
        elif "softmax" in name.lower():
            family = "softmax"
        elif "layer_norm" in name.lower() or "rmsnorm" in name.lower():
            family = "norm"
        elif "elementwise" in name.lower():
            family = "elementwise"
        elif "reduce_kernel" in name:
            family = "reduce"
        elif "mamba" in name.lower():
            family = "mamba"
        else:
            family = name.split("(")[0][:60]
        by_kind[family][0] += overlap
        by_kind[family][1] += 1
        by_kind[family][2] = max(by_kind[family][2], (k_e - k_s) / 1e6)
        by_stream[stream] += overlap
    return by_kind, by_stream


def sync_breakdown(s, e):
    rows = cur.execute(
        """
        SELECT start, end, syncType FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        WHERE start < ? AND end > ?
        """,
        (e, s),
    ).fetchall()
    total = 0
    big_count = 0
    biggest = 0
    for k_s, k_e, _ in rows:
        overlap = max(0, min(k_e, e) - max(k_s, s))
        if overlap > 0:
            total += overlap
            if overlap >= 1e7:  # >= 10 ms
                big_count += 1
                biggest = max(biggest, overlap)
    return total, big_count, biggest / 1e6


for label, target in [("FAST iter=33 mb=0", "iter=33/mb=0"), ("SLOW iter=38 mb=1", "iter=38/mb=1")]:
    print(f"\n========= {label} =========")
    win = fwd_window(target)
    if not win:
        print("not found")
        continue
    s, e = win
    dur_ms = (e - s) / 1e6
    print(f"  fwd window: {dur_ms:.1f} ms")

    by_kind, by_stream = kernel_breakdown(s, e)
    print(f"  kernel coverage on each stream (ms):")
    for stream, ns in sorted(by_stream.items(), key=lambda x: -x[1]):
        print(f"    stream={stream:>4}: {ns/1e6:7.1f} ms")
    print(f"  kernel coverage by family (top 15, ms):")
    for fam, (ns, cnt, mx) in sorted(by_kind.items(), key=lambda x: -x[1][0])[:15]:
        print(f"    {fam:>20}: total={ns/1e6:7.1f}ms  count={cnt:5d}  max_single={mx:6.1f}ms")

    sync_total, big_count, biggest = sync_breakdown(s, e)
    print(f"  cuda host syncs: total={sync_total/1e6:.1f} ms, big(>=10ms) count={big_count}, biggest={biggest:.1f} ms")
