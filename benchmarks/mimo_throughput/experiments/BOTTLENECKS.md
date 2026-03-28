# Bottleneck Analysis — Colocated MIMO Training

## Current Bottleneck: Communication-Bound (TP all-reduce)

**Evidence:** Profiling shows 28.2% of CUDA time in NCCL all_reduce at TP4. Only 51% of time is actual compute (GEMM + attention).

**Root cause:** With TP=4, every transformer layer does 2 all-reduces (one after attention column parallel, one after FFN row parallel). With 32 LLM layers × 2 all-reduces × 2 microbatches × 2 (fwd+bwd) = 256 all-reduce calls per iteration. At ~4.3ms average per all-reduce, this accounts for ~1.1s of the ~3.9s profiled.

**Why this matters for colocated:** The colocated communicator adds minimal overhead (fan-in is a single all-gather). The dominant communication cost is WITHIN each module's TP all-reduce, not BETWEEN modules. This means:
- Reducing encoder TP (from 4→2) saves encoder all-reduce but encoder is only 12% of compute
- The LLM's TP all-reduce is the real bottleneck

## Optimization Hypotheses (ordered by expected impact)

1. **Increase batch size (mbs=4 or mbs=8)** — doubles/quadruples GEMM size, shifting balance toward compute. GEMM time scales linearly with batch, but all-reduce time stays constant.
   - Expected: significant TFLOPs improvement
   - Risk: OOM at higher batch (currently 58.6GB of 80GB)

2. **Try TP2/DP4 for LLM** — halves TP all-reduce frequency (larger messages, but fewer calls). Each all-reduce moves 2x more data but happens with 2x fewer ranks.
   - Expected: moderate improvement if network-latency-bound (small messages)
   - Risk: TP2 means each GPU holds 2x more LLM params → more memory

3. **Enable activation recomputation** — frees memory to allow larger batch
   - Expected: enables mbs=4-8 which unlocks hypothesis #1
   - Risk: recomputation adds ~33% extra forward compute

4. **DDP bucket tuning** — current bucket_size=10000. Larger buckets = fewer collective calls for gradient sync
   - Expected: marginal improvement (reduce_scatter is only 1.3% of time)
