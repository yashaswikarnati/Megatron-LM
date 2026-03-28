# Experiment Log — Colocated MIMO Throughput Optimization

Model: 1B ViT (40L, H=1408) + 7B LLM (32L, H=4096) on 8xH100 80GB, PP=1.

---

### Experiment 1: Homogeneous Baseline (2026-03-28)
**Config:** 1B+7B | enc TP4/DP2, llm TP4/DP2 (homogeneous) | mbs=2, nmb=2, GBS=4
**Change:** Initial baseline
**Result:** 272.0 TFLOPs/GPU, 53,235 tok/s, 58.6 GB | fwd_bwd: 1178ms, opt: 52ms
**MFU:** ~27.5%

---

### Experiment 2: Heterogeneous Fan-In (2026-03-28)
**Config:** 1B+7B | enc TP2/DP4, llm TP4/DP2 (fan-in) | mbs=2, nmb=2, GBS=4
**Change:** Encoder TP4→TP2, DP2→DP4
**Result:** 271.2 TFLOPs/GPU, 53,077 tok/s, 58.4 GB | fwd_bwd: 1181ms, opt: 53ms
**vs Baseline:** -0.3% TFLOPs
**Verdict:** NEUTRAL

---

### Profiling Analysis (2026-03-28)
**Config:** hetero enc TP2/DP4, llm TP4/DP2, mbs=2

**Phase Timeline (inclusive CUDA time, 2 profiled iterations):**
| Phase | CUDA ms | % iter | Avg/call |
|-------|---------|--------|----------|
| encoder_forward | 359ms | 17.2% | 44.9ms |
| bridge_communicate | 3.7ms | 0.2% | 0.5ms |
| text_embedding | 11.5ms | 0.6% | 1.4ms |
| align_embeddings | 14.9ms | 0.7% | 1.9ms |
| llm_forward | 1505ms | 72.3% | 188.1ms |
| optimizer_step | 188ms | 9.0% | 47.0ms |

**Op Breakdown:** Compute 50.2%, Communication 27.5% (AllReduce 1428ms), Elementwise 9.3%

**Memory:** Peak 54.4 GB (68%), Encoder activations 21.7 GB held through LLM fwd+bwd

**Key Insights:**
- Communication (28%) is the #1 bottleneck, primarily TP all-reduce in LLM
- Encoder is 19% of model compute, LLM is 81% → encoder TP optimization has limited impact
- Bridge communicate is essentially free (0.2%)
- 25.6 GB apparent headroom but backward pushes to ~75 GB actual

---

### Experiment 3: Batch Size Increase mbs=4 (2026-03-28)
**Config:** hetero enc TP2/DP4, llm TP4/DP2 | mbs=4, nmb=2
**Change:** Double micro_batch_size to improve compute/communication ratio
**Result:** OOM — 75 GB allocated, needed ~256 MB more
**Verdict:** REVERT
**Learning:** Forward peak is 54 GB at mbs=2, but backward pushes to ~75 GB. Doubling batch doubles activation memory during backward. Need activation recomputation/offload to enable larger batch.

---

### Experiment 4: Text Embedding Reorder (2026-03-28)
**Config:** hetero enc TP2/DP4, llm TP4/DP2 | mbs=2, nmb=2
**Change:** Moved get_text_embeddings() BEFORE encoder forward in _forward_all_modules(). Text embeddings are independent of encoder — reordering enables GPU overlap (embedding lookup can run while encoder kernels are queued).
**Profiling insight:** text_embedding (15ms) ran sequentially after encoder (359ms) but has no data dependency.
**Result:** 273.9 TFLOPs/GPU, 53,603 tok/s, 58.4 GB | fwd_bwd: 1169ms, opt: 53ms
**vs Previous (Exp 2):** +1.0% TFLOPs
**vs Homo Baseline (Exp 1):** +0.7% TFLOPs
**Verdict:** KEEP — small but consistent improvement, zero downside
**Learning:** Code reordering for overlap gives marginal gains. The text embedding is small (~15ms) so the improvement is bounded. Larger gains require addressing the 28% communication bottleneck.

**Updated homo baseline with same optimization:** 275.0 TFLOPs (up from 272.0)
**Hetero vs Homo (fair comparison):** 273.9 vs 275.0 → hetero is -0.4% (still slightly worse)

---

### Current Best

| Config | TFLOPs/GPU | tok/s | Memory |
|--------|-----------|-------|--------|
| Homo TP4/DP2 | **275.0** | 53,832 | 58.6 GB |
| Hetero TP2/DP4→TP4/DP2 | 273.9 | 53,603 | 58.4 GB |

---

### Next Steps

1. **Activation recomputation** — Enable for encoder to free 21.7 GB, enabling mbs=3. Needs research into how Megatron wires recompute_modules for MIMO.
2. **Try larger vision seq** — seq_length=4096 increases encoder % of compute, making heterogeneous TP split more impactful
3. **Try enc TP1/DP8, llm TP8/DP1** — extreme heterogeneous with zero encoder all-reduce
4. **Profile homo baseline** — understand if homo has different communication pattern
