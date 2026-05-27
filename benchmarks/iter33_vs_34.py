"""Per-microbatch breakdown for iter=33 (3524ms) vs iter=34 (4490ms).
Picks LLM rank 16. Shows kernel breakdown + the largest individual kernel
of each NCCL family per mb so you can find it directly in the nsys-ui."""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

NSYS_DIR = Path(sys.argv[1])
TIMELINE_DIR = Path(sys.argv[2])

con = sqlite3.connect(NSYS_DIR / "rank00016.sqlite")
cur = con.cursor()


def fwd_window(it, mb):
    return cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={it}/mb={mb}%",),
    ).fetchone()


def kernels_in(s, e):
    return cur.execute(
        """
        SELECT k.start, k.end, k.streamId, sid.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds sid ON k.demangledName = sid.id
        WHERE k.start < ? AND k.end > ?
        """,
        (e, s),
    ).fetchall()


def fam_of(name):
    if "AllGather" in name:
        return "AllGather"
    if "SendRecv" in name:
        return "SendRecv"
    if "ReduceScatter" in name:
        return "ReduceScatter"
    if "AllReduce" in name:
        return "AllReduce"
    return "compute"


def breakdown(s, e):
    rows = kernels_in(s, e)
    fam_total = defaultdict(int)
    fam_max = defaultdict(lambda: (0, 0, 0))  # fam -> (dur_ns, start_ns, end_ns)
    for k_s, k_e, _, name in rows:
        overlap = max(0, min(k_e, e) - max(k_s, s))
        if overlap <= 0:
            continue
        f = fam_of(name)
        fam_total[f] += overlap
        kdur = k_e - k_s
        if kdur > fam_max[f][0] and k_s >= s and k_e <= e:
            fam_max[f] = (kdur, k_s, k_e)
    return fam_total, fam_max


def encoder_mb_signals(iter_no):
    """Return per-mb (data.next, fwd) for each encoder rank."""
    out = {rank: {} for rank in range(8)}
    for rank in range(8):
        per_fwd = []
        per_data = []
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
                    per_fwd.append(d)
                elif ev == "data.next":
                    per_data.append(d)
        out[rank]["fwd"] = per_fwd
        out[rank]["data"] = per_data
    return out


print(f"{'tag':<6} {'iter':>4} {'mb':>2} {'fwd':>6} {'compute':>8} {'bridge':>7} {'AG':>6} "
      f"{'AG_max_ms':>10} {'AG_max_s_ns':>14} {'AG_max_e_ns':>14}  enc_slow_rank,data,fwd")
print("-" * 130)
totals = defaultdict(lambda: defaultdict(float))

for tag, iter_no in [("FAST", 33), ("SLOW", 34)]:
    enc_sig = encoder_mb_signals(iter_no)
    # encoder per-mb worst
    for mb in range(4):
        win = fwd_window(iter_no, mb)
        if not win:
            print(f"{tag}  {iter_no} {mb} not found")
            continue
        s, e = win
        fwd_ms = (e - s) / 1e6
        fam_total, fam_max = breakdown(s, e)
        compute = fam_total.get("compute", 0) / 1e6
        bridge = fam_total.get("SendRecv", 0) / 1e6
        ag = fam_total.get("AllGather", 0) / 1e6
        ag_max = fam_max.get("AllGather", (0, 0, 0))
        ag_max_ms = ag_max[0] / 1e6
        # encoder worst rank this mb
        worst_d, worst_f, worst_r = 0, 0, -1
        for r in range(8):
            d_ms = enc_sig[r]["data"][mb] if mb < len(enc_sig[r]["data"]) else 0
            f_ms = enc_sig[r]["fwd"][mb] if mb < len(enc_sig[r]["fwd"]) else 0
            if d_ms + f_ms > worst_d + worst_f:
                worst_d, worst_f, worst_r = d_ms, f_ms, r
        print(f"{tag:<6} {iter_no:>4} {mb:>2} {fwd_ms:>6.0f} {compute:>8.1f} {bridge:>7.1f} {ag:>6.1f} "
              f"{ag_max_ms:>10.1f} {ag_max[1]:>14} {ag_max[2]:>14}  "
              f"rank{worst_r} data={worst_d:.0f} fwd={worst_f:.0f}")
        totals[tag]["fwd"] += fwd_ms
        totals[tag]["compute"] += compute
        totals[tag]["bridge"] += bridge
        totals[tag]["AG"] += ag
print()
print(f"Iter sums (across 4 mb):")
for tag in ("FAST", "SLOW"):
    t = totals[tag]
    print(f"  {tag}:  fwd={t['fwd']:.0f}  compute={t['compute']:.0f}  bridge={t['bridge']:.0f}  AG={t['AG']:.0f}")
delta = {k: totals["SLOW"][k] - totals["FAST"][k] for k in totals["FAST"]}
print(f"  delta: fwd={delta['fwd']:+.0f}  compute={delta['compute']:+.0f}  bridge={delta['bridge']:+.0f}  AG={delta['AG']:+.0f}")
