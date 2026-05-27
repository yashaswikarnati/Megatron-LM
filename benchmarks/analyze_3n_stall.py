"""Diagnose what GPU is doing during the host stall in slow MoE microbatches.

Run inside the m_lm_energon container with one GPU; reads the rank16 sqlite
exported from nsys-rep.
"""

import sqlite3
import sys

SQLITE = sys.argv[1] if len(sys.argv) > 1 else "rank00016.sqlite"
TARGETS = [
    ("iter=38/mb=1", "schedule.forward/iter=38/mb=1"),
    ("iter=39/mb=0", "schedule.forward/iter=39/mb=0"),
    ("iter=34/mb=1", "schedule.forward/iter=34/mb=1"),
]

con = sqlite3.connect(SQLITE)
cur = con.cursor()


def nvtx_range(text_like):
    rows = cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE ?",
        (f"%{text_like}%",),
    ).fetchall()
    return [r for r in rows if r[1]]


def kernels_in_window(start_ns, end_ns):
    return cur.execute(
        """
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.start < ? AND k.end > ?
        ORDER BY k.start
        """,
        (end_ns, start_ns),
    ).fetchall()


def memcpy_in_window(start_ns, end_ns):
    return cur.execute(
        """
        SELECT m.start, m.end, m.streamId, m.copyKind, m.bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.start < ? AND m.end > ?
        ORDER BY m.start
        """,
        (end_ns, start_ns),
    ).fetchall()


def sync_in_window(start_ns, end_ns):
    """CUDA host-side synchronizations (cudaStreamSynchronize, cudaEventSynchronize)."""
    return cur.execute(
        """
        SELECT s.start, s.end, s.syncType, st.value
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
        LEFT JOIN StringIds st ON s.eventId = st.id
        WHERE s.start < ? AND s.end > ?
        ORDER BY s.start
        """,
        (end_ns, start_ns),
    ).fetchall()


for label, target in TARGETS:
    print(f"\n========= {label} =========")
    ranges = nvtx_range(target)
    if not ranges:
        print("  NOT FOUND")
        continue
    s_start, s_end, _ = ranges[0]
    print(f"  schedule.forward window: dur={(s_end - s_start) / 1e6:.1f} ms")

    # all dtoh_sync ranges inside the schedule.forward window
    label_short = label  # e.g. "iter=38/mb=1"
    dtoh = nvtx_range(f"moe.dtoh_sync/{label_short}")
    dtoh = [(s, e) for s, e, _ in dtoh if s_start <= s <= s_end]
    print(f"  {len(dtoh)} moe.dtoh_sync events inside window")
    durs = sorted([(e - s) / 1e6 for s, e in dtoh], reverse=True)
    if durs:
        print(f"  top 5 dtoh dur (ms): {[f'{d:.1f}' for d in durs[:5]]}")
        big_s, big_e = max(dtoh, key=lambda p: p[1] - p[0])
        big_dur_ms = (big_e - big_s) / 1e6
        print(f"\n  >>> deep-dive on biggest dtoh: dur={big_dur_ms:.1f} ms")
        print(f"      window: [{big_s}, {big_e}]")

        kernels = kernels_in_window(big_s, big_e)
        print(f"      kernels overlapping window: {len(kernels)}")
        # aggregate kernel coverage in the window by name+stream
        coverage = {}  # (name, stream) -> total overlap ns
        for k_s, k_e, k_stream, k_name in kernels:
            overlap = max(0, min(k_e, big_e) - max(k_s, big_s))
            key = (k_name, k_stream)
            coverage[key] = coverage.get(key, 0) + overlap
        # top kernels by coverage
        top = sorted(coverage.items(), key=lambda kv: -kv[1])[:15]
        print(f"      top kernel coverage (ms wall):")
        for (name, stream), ns in top:
            print(f"        stream={stream:>4}  {ns / 1e6:7.1f}ms  {name[:80]}")

        memcpys = memcpy_in_window(big_s, big_e)
        print(f"      memcpys overlapping: {len(memcpys)}")
        copy_kind = {0: "HtoH", 1: "HtoD", 2: "DtoH", 3: "DtoD", 8: "PtoP"}
        for m_s, m_e, m_stream, m_kind, m_bytes in memcpys[:10]:
            kind = copy_kind.get(m_kind, f"k{m_kind}")
            print(f"        stream={m_stream:>4}  {(m_e - m_s) / 1e6:7.3f}ms  {kind} {m_bytes}B")

        syncs = sync_in_window(big_s, big_e)
        print(f"      cuda syncs overlapping: {len(syncs)}")
        for s_s, s_e, s_type, _ in syncs[:8]:
            print(f"        type={s_type}  dur={(s_e - s_s) / 1e6:.1f}ms")
