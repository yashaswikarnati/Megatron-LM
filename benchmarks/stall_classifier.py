"""Two-stage stall classifier for the 3n nsys run (iter=34 timeline).

Stage 1 (nsys + JSONL, iters 30-40):
  For each (iter, mb) where fwd > ~400 ms on any LLM rank, identify the slow
  rank and classify the root cause by combining:
    - LLM dataloader stall  (JSONL data.next > 100 ms on any LLM rank)
    - Encoder host stall    (encoder JSONL data.next/fwd > 100 ms)
    - Mamba kernel stall    (NVTX MambaSplitConv1dScanCombinedFn > 50 ms)
    - Attention stall       (NVTX attention.forward > 100 ms w/ low GPU coverage)
    - AllGather cascade     (long AG with no other identified cause)

Stage 2 (JSONL only, all 100 iters):
  Project the same labels onto every iter using only signals available in
  JSONL across all ranks. We can detect:
    - DATALOADER  (data.next > 100 ms on any rank)
    - ENCODER     (encoder JSONL fwd > 100 ms)
    - INFERRED_KERNEL_STALL (fwd_ms variance across LLM ranks > 100 ms with
      no DATALOADER and no ENCODER spike — must be a mamba/attention kernel
      jitter or other on-GPU stall)
    - CLEAN       (fwd_ms within ~50 ms across ranks; no spikes)
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path("/lustre/fsw/portfolios/nemotron/users/ykarnati/agents-scratch/runs/mimo-nsys-gbs32-PG1-GR1-FLB1/296658")
NSYS_DIR = ROOT / "nsys"
TIMELINE_DIR = ROOT / "timeline"
LLM_RANKS = list(range(8, 24))
ENC_RANKS = list(range(0, 8))
PROFILE_ITERS = list(range(30, 41))


# ----- JSONL helpers ----------------------------------------------------------
def load_jsonl_per_iter_per_mb(rank, fields=("schedule.forward", "data.next")):
    """Return {iter: {field: [mb0, mb1, mb2, mb3]}}."""
    out = defaultdict(lambda: defaultdict(list))
    path = TIMELINE_DIR / f"rank{rank:05d}.jsonl"
    with open(path) as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = j.get("iteration")
            ev = j.get("event")
            if it is None or ev not in fields:
                continue
            out[it][ev].append(j["duration_us"] / 1000.0)
    return out


# pre-load JSONL data for all ranks (only what we need)
print("Loading JSONL for all 24 ranks…")
jsonl_data = {r: load_jsonl_per_iter_per_mb(r) for r in range(24)}


def fwd_ms(rank, it, mb):
    fs = jsonl_data[rank].get(it, {}).get("schedule.forward", [])
    return fs[mb] if mb < len(fs) else 0.0


def dnext_ms(rank, it, mb):
    ds = jsonl_data[rank].get(it, {}).get("data.next", [])
    return ds[mb] if mb < len(ds) else 0.0


# ----- nsys helpers ----------------------------------------------------------
_con_cache = {}


def _con(rank):
    if rank not in _con_cache:
        _con_cache[rank] = sqlite3.connect(NSYS_DIR / f"rank{rank:05d}.sqlite")
    return _con_cache[rank]


def fwd_window(rank, it, mb):
    cur = _con(rank).cursor()
    return cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND end IS NOT NULL",
        (f"%schedule.forward/iter={it}/mb={mb}%",),
    ).fetchone()


def max_mamba_ms(rank, it, mb):
    win = fwd_window(rank, it, mb)
    if not win:
        return 0.0
    s, e = win
    cur = _con(rank).cursor()
    row = cur.execute(
        "SELECT MAX(end - start) FROM NVTX_EVENTS WHERE text LIKE ? AND start >= ? AND end <= ?",
        ("%MambaSplitConv1dScanCombined%", s, e),
    ).fetchone()
    return (row[0] or 0) / 1e6


def max_attention_stall_ms(rank, it, mb):
    """Find longest attention.forward NVTX where GPU kernel coverage on stream 7 is < 20% of its duration."""
    win = fwd_window(rank, it, mb)
    if not win:
        return 0.0, 0.0
    s, e = win
    cur = _con(rank).cursor()
    # Look at the attention NVTX ranges
    attns = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND start >= ? AND end <= ? ORDER BY (end - start) DESC LIMIT 3",
        ("%transformer_layer._forward_attention%", s, e),
    ).fetchall()
    if not attns:
        # try alternate name
        attns = cur.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ? AND start >= ? AND end <= ? ORDER BY (end - start) DESC LIMIT 3",
            ("%self_attention%", s, e),
        ).fetchall()
    if not attns:
        return 0.0, 0.0
    a_s, a_e = attns[0]
    a_dur_ms = (a_e - a_s) / 1e6
    if a_dur_ms < 100:
        return a_dur_ms, 0.0  # not a stall
    # compute GPU kernel coverage on stream 7 inside the attention range
    krow = cur.execute(
        """
        SELECT COALESCE(SUM(MIN(k.end, ?) - MAX(k.start, ?)), 0)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE k.start < ? AND k.end > ? AND k.streamId = 7
        """,
        (a_e, a_s, a_e, a_s),
    ).fetchone()
    coverage_ms = (krow[0] or 0) / 1e6
    idle_ratio = 1 - coverage_ms / a_dur_ms if a_dur_ms > 0 else 0
    return a_dur_ms, idle_ratio


def max_ag_ms(rank, it, mb):
    win = fwd_window(rank, it, mb)
    if not win:
        return 0.0
    s, e = win
    cur = _con(rank).cursor()
    row = cur.execute(
        """
        SELECT MAX(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds sid ON k.demangledName = sid.id
        WHERE sid.value LIKE '%AllGather%' AND k.start >= ? AND k.end <= ?
        """,
        (s, e),
    ).fetchone()
    return (row[0] or 0) / 1e6


# ----- per-(iter, mb) classification ----------------------------------------
def classify_mb(it, mb, with_nsys: bool):
    """Return dict with diagnostics for this (iter, mb)."""
    # Find max LLM fwd
    fwds = [(r, fwd_ms(r, it, mb)) for r in LLM_RANKS]
    fwds.sort(key=lambda x: -x[1])
    max_fwd_rank, max_fwd = fwds[0]
    min_fwd = fwds[-1][1]
    spread = max_fwd - min_fwd

    # LLM dataloader spikes on this mb
    llm_dnext = [(r, dnext_ms(r, it, mb)) for r in LLM_RANKS]
    llm_dnext_spike = max(llm_dnext, key=lambda x: x[1])

    # Encoder fwd / dnext spikes on this mb
    enc_fwd = [(r, fwd_ms(r, it, mb)) for r in ENC_RANKS]
    enc_fwd_spike = max(enc_fwd, key=lambda x: x[1])
    enc_dnext = [(r, dnext_ms(r, it, mb)) for r in ENC_RANKS]
    enc_dnext_spike = max(enc_dnext, key=lambda x: x[1])

    result = {
        "iter": it,
        "mb": mb,
        "max_fwd_ms": max_fwd,
        "fwd_spread_ms": spread,
        "slow_rank": max_fwd_rank,
        "llm_dnext_max": llm_dnext_spike,
        "enc_fwd_max": enc_fwd_spike,
        "enc_dnext_max": enc_dnext_spike,
    }

    if with_nsys:
        # On the slow rank, check mamba and attention
        mamba = max_mamba_ms(max_fwd_rank, it, mb)
        attn_dur, attn_idle = max_attention_stall_ms(max_fwd_rank, it, mb)
        ag = max_ag_ms(max_fwd_rank, it, mb)
        # Also scan all LLM ranks for outlier mamba
        all_mamba = [(r, max_mamba_ms(r, it, mb)) for r in LLM_RANKS]
        worst_mamba = max(all_mamba, key=lambda x: x[1])
        result["mamba_on_slow_rank_ms"] = mamba
        result["worst_mamba"] = worst_mamba
        result["attn_dur_on_slow_rank_ms"] = attn_dur
        result["attn_idle_ratio_on_slow_rank"] = attn_idle
        result["ag_max_on_slow_rank_ms"] = ag
    return result


def causes_for(c, with_nsys: bool):
    causes = []
    # Dataloader stall: any rank dnext > 100 ms
    if c["llm_dnext_max"][1] > 100:
        causes.append(f"DATALOADER(rank{c['llm_dnext_max'][0]}={c['llm_dnext_max'][1]:.0f}ms)")
    if c["enc_dnext_max"][1] > 100:
        causes.append(f"ENC_DATALOADER(rank{c['enc_dnext_max'][0]}={c['enc_dnext_max'][1]:.0f}ms)")
    if c["enc_fwd_max"][1] > 100:
        causes.append(f"ENCODER_FWD(rank{c['enc_fwd_max'][0]}={c['enc_fwd_max'][1]:.0f}ms)")
    if with_nsys:
        if c["worst_mamba"][1] > 50:
            causes.append(f"MAMBA(rank{c['worst_mamba'][0]}={c['worst_mamba'][1]:.0f}ms)")
        if c["attn_dur_on_slow_rank_ms"] > 100 and c["attn_idle_ratio_on_slow_rank"] > 0.5:
            causes.append(f"ATTN_STALL({c['attn_dur_on_slow_rank_ms']:.0f}ms, gpu_idle={c['attn_idle_ratio_on_slow_rank']*100:.0f}%)")
        if c["ag_max_on_slow_rank_ms"] > 100 and not causes:
            causes.append(f"AG_CASCADE({c['ag_max_on_slow_rank_ms']:.0f}ms)")
    else:
        # No nsys — if fwd_spread > 100 and no JSONL cause found, infer kernel stall
        if c["fwd_spread_ms"] > 100 and not causes:
            causes.append(f"INFERRED_KERNEL_STALL(spread={c['fwd_spread_ms']:.0f}ms)")
    if not causes:
        causes.append("CLEAN")
    return causes


# ---------------- Stage 1: nsys + JSONL on iters 30-40 ----------------
print("\n" + "=" * 100)
print("STAGE 1 — nsys + JSONL classification (iters 30-40)")
print("=" * 100)
print(f"{'iter':>4} {'mb':>2} {'max_fwd':>8} {'spread':>7} {'slow_rank':>10}  causes")
print("-" * 110)
slow_count_s1 = defaultdict(int)
for it in PROFILE_ITERS:
    for mb in range(4):
        c = classify_mb(it, mb, with_nsys=True)
        if c["max_fwd_ms"] < 400 and c["fwd_spread_ms"] < 100:
            continue  # skip clean ones
        causes = causes_for(c, with_nsys=True)
        cause_str = " | ".join(causes)
        print(f"{it:>4} {mb:>2} {c['max_fwd_ms']:>8.0f} {c['fwd_spread_ms']:>7.0f} rank{c['slow_rank']:<6}  {cause_str}")
        for tag in causes:
            slow_count_s1[tag.split("(")[0]] += 1

print(f"\nStage 1 cause histogram (44 mb in iters 30-40):")
for k, v in sorted(slow_count_s1.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")


# ---------------- Stage 2: JSONL-only on all 100 iters ----------------
print("\n" + "=" * 100)
print("STAGE 2 — JSONL-only classification (all 100 iters, all 4 mbs)")
print("=" * 100)
print("Showing only (iter, mb) with max_fwd > 400 ms OR spread > 100 ms")
print(f"{'iter':>4} {'mb':>2} {'max_fwd':>8} {'spread':>7} {'slow_rank':>10}  causes")
print("-" * 110)
slow_count_s2 = defaultdict(int)
total_iters_seen = 0
for it in range(1, 101):
    for mb in range(4):
        c = classify_mb(it, mb, with_nsys=False)
        total_iters_seen += 1
        if c["max_fwd_ms"] < 400 and c["fwd_spread_ms"] < 100:
            continue
        causes = causes_for(c, with_nsys=False)
        cause_str = " | ".join(causes)
        for tag in causes:
            slow_count_s2[tag.split("(")[0]] += 1
        print(f"{it:>4} {mb:>2} {c['max_fwd_ms']:>8.0f} {c['fwd_spread_ms']:>7.0f} rank{c['slow_rank']:<6}  {cause_str}")

print(f"\nStage 2 cause histogram across {total_iters_seen} (iter, mb) total:")
for k, v in sorted(slow_count_s2.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
