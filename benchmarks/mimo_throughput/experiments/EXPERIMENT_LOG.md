# Experiment Log — Colocated MIMO Throughput Optimization

Model: 1B ViT (40L, H=1408) + 7B LLM (32L, H=4096) on 8xH100 80GB, PP=1.

---

### Experiment 1: Homogeneous Baseline (2026-03-28)
**Config:** 1B+7B | enc TP4/DP2, llm TP4/DP2 (homogeneous) | mbs=2, nmb=2, GBS=4
**Change:** Initial baseline
**Result:** 272.0 TFLOPs/GPU, 53,235 tok/s, 58.6 GB | fwd_bwd: 1178ms, opt: 52ms
**MFU:** ~27.5% (vs H100 BF16 peak 989 TFLOPs)
**Learning:** Baseline established. 58.6GB of 80GB used — ~21GB headroom. fwd_bwd dominates (96% of iteration). Optimizer step is fast (4%).

---

### Experiment 2: Heterogeneous Fan-In (2026-03-28)
**Config:** 1B+7B | enc TP2/DP4, llm TP4/DP2 (fan-in) | mbs=2, nmb=2, GBS=4
**Change:** Encoder TP4→TP2 (halved), encoder DP2→DP4 (doubled). Encoder gets less all-reduce overhead, more data parallelism.
**Result:** 271.2 TFLOPs/GPU, 53,077 tok/s, 58.4 GB | fwd_bwd: 1181ms, opt: 53ms
**vs Baseline:** -0.3% TFLOPs (essentially same)
**Verdict:** NEUTRAL — no improvement at this config
**Learning:** For this model size + batch size, heterogeneous TP2/DP4 for encoder doesn't help. The encoder is only ~12% of total compute (1B vs 7B), so optimizing its parallelism has small impact on total throughput. The 7B LLM dominates iteration time.

---

### Profiling Analysis (Experiment 2, iterations 5-7)

**CUDA time breakdown (3.94s total for 2 profiled iterations):**

| Category | CUDA Time | % of Total | Notes |
|----------|-----------|------------|-------|
| **NCCL all_reduce** | 1,109ms | **28.2%** | TP all-reduce for attention/FFN outputs |
| **GEMM (fwd+bwd)** | ~1,400ms | **35.6%** | Linear layers forward + backward |
| **Attention (fwd+bwd)** | ~620ms | **15.7%** | FusedAttn forward + backward |
| **LayerNorm backward** | 62ms | 1.6% | |
| **Dropout** | 55ms | 1.4% | |
| **reduce_scatter** | 53ms | 1.3% | DDP gradient sync |
| **Elementwise** | ~412ms | 10.5% | add, add_, other elementwise ops |
| **Other** | ~226ms | 5.7% | |

**Key insight: 28% of CUDA time is NCCL all_reduce (TP communication).** This is the #1 bottleneck. The actual compute (GEMM + attention) is only ~51% of CUDA time. The rest is communication and overhead.

---

### Next Steps

1. **Try TP8/DP1 for LLM** — more TP means smaller all-reduce messages (less data per all-reduce) but more all-reduce calls. Tradeoff to test.
2. **Try enc TP1/DP8, llm TP4/DP2** — eliminate encoder all-reduce entirely
3. **Increase batch size** — mbs=4 would double arithmetic intensity, reducing communication fraction
4. **Profile with larger batch** — see if GEMM fraction increases (compute-bound territory)
