"""Per-microbatch attribution between FAST and SLOW iter.

For each of the 4 microbatches in iter=33 (FAST) and iter=38 (SLOW), break LLM
rank 16 schedule.forward into kernel families. Also pull encoder-side host
data.next/fwd from JSONL per-mb to anchor.
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

NSYS_DIR = Path(sys.argv[1])  # path to nsys/ dir with rank00016.sqlite
TIMELINE_DIR = Path(sys.argv[2])  # path to timeline/ dir with all rank JSONLs


def kernel_break(cur, s_ns, e_ns):
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.streamId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE k.start < ? AND k.end > ?
        """,
        (e_ns, s_ns),
    ).fetchall()
    by_fam = defaultdict(int)
    for k_s, k_e, _stream, name in rows:
        overlap = max(0, min(k_e, e_ns) - max(k_s, s_ns))
        if overlap <= 0:
            continue
        if "AllGather" in name:
            fam = "AllGather"
        elif "SendRecv" in name:
            fam = "SendRecv (bridge)"
        elif "ReduceScatter" in name:
            fam = "ReduceScatter"
        elif "AllReduce" in name:
            fam = "AllReduce"
        else:
            fam = "compute"
        by_fam[fam] += overlap
    return by_fam


def encoder_per_mb(iter_no):
    """Return per-mb max host time across encoder ranks 0-7."""
    by_mb_data = [0] * 4  # max data.next across ranks
    by_mb_fwd = [0] * 4  # max fwd across ranks
    worst_rank = [(0, -1)] * 4  # (max_fwd, rank)
    for rank in range(8):
        per_mb_fwd = []
        per_mb_data = []
        with open(TIMELINE_DIR / f"rank{rank:05d}.jsonl") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if j.get("iteration") != iter_no:
                    continue
                ev = j.get("event")
                d = j.get("duration_us", 0) / 1000.0
                if ev == "schedule.forward":
                    per_mb_fwd.append(d)
                elif ev == "data.next":
                    per_mb_data.append(d)
        for i, v in enumerate(per_mb_fwd[:4]):
            if v > by_mb_fwd[i]:
                by_mb_fwd[i] = v
                worst_rank[i] = (v, rank)
        for i, v in enumerate(per_mb_data[:4]):
            if v > by_mb_data[i]:
                by_mb_data[i] = v
    return by_mb_data, by_mb_fwd, worst_rank


con = sqlite3.connect(NSYS_DIR / "rank00016.sqlite")
cur = con.cursor()


def fwd_window(it_no, mb_no):
    return cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={it_no}/mb={mb_no}%",),
    ).fetchone()


print(f"{'tag':<8}  {'iter':>4}  {'mb':>2}  {'fwd_ms':>7}  {'compute':>8}  "
      f"{'bridge':>8}  {'AllGather':>10}  {'enc_data':>9}  {'enc_fwd':>8}  enc_worst_rank")
print("-" * 110)
totals_fast = defaultdict(float)
totals_slow = defaultdict(float)
for tag, iter_no in [("FAST", 33), ("SLOW", 38)]:
    enc_data, enc_fwd, enc_worst = encoder_per_mb(iter_no)
    for mb in range(4):
        win = fwd_window(iter_no, mb)
        if not win:
            print(f"{tag}  {iter_no:>4} {mb:>2}  -- not found --")
            continue
        s, e = win
        fwd_ms = (e - s) / 1e6
        breakdown = kernel_break(cur, s, e)
        compute = breakdown.get("compute", 0) / 1e6
        bridge = breakdown.get("SendRecv (bridge)", 0) / 1e6
        ag = breakdown.get("AllGather", 0) / 1e6
        ed = enc_data[mb]
        ef = enc_fwd[mb]
        ew = enc_worst[mb][1]
        print(f"{tag:<8}  {iter_no:>4}  {mb:>2}  {fwd_ms:>7.1f}  {compute:>8.1f}  "
              f"{bridge:>8.1f}  {ag:>10.1f}  {ed:>9.1f}  {ef:>8.1f}  rank{ew}")
        tot = totals_fast if tag == "FAST" else totals_slow
        tot["fwd"] += fwd_ms
        tot["compute"] += compute
        tot["bridge"] += bridge
        tot["allgather"] += ag
print()
print("Iter sums (across 4 mb):")
print(f"  FAST  fwd={totals_fast['fwd']:.0f}  compute={totals_fast['compute']:.0f}  "
      f"bridge={totals_fast['bridge']:.0f}  AG={totals_fast['allgather']:.0f}")
print(f"  SLOW  fwd={totals_slow['fwd']:.0f}  compute={totals_slow['compute']:.0f}  "
      f"bridge={totals_slow['bridge']:.0f}  AG={totals_slow['allgather']:.0f}")
print(f"  delta fwd={totals_slow['fwd']-totals_fast['fwd']:.0f}  "
      f"compute={totals_slow['compute']-totals_fast['compute']:+.0f}  "
      f"bridge={totals_slow['bridge']-totals_fast['bridge']:+.0f}  "
      f"AG={totals_slow['allgather']-totals_fast['allgather']:+.0f}")
