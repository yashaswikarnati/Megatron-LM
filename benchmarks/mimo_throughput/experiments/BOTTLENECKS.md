# Bottleneck Analysis — Colocated MIMO Training

## Current State (2026-03-28)

Model: 1B ViT + 7B LLM, 8xH100, mbs=2

**Homo 275.0 TFLOPs vs Hetero 273.9 TFLOPs — hetero is NOT winning yet.**

## Why Heterogeneous Doesn't Help (Yet)

The encoder is only 19% of model compute. The LLM (81%) dominates iteration time. Reducing encoder TP from 4→2 saves ~8ms in encoder all-reduce, but this is lost in the noise of the 1170ms iteration.

For heterogeneous to win, we need either:
- **Larger batch** (increases encoder fraction due to better encoder GEMM utilization at lower TP)
- **Longer vision sequences** (increases encoder compute relative to LLM)
- **Extreme TP splits** (enc TP1/DP8 eliminates ALL encoder communication)

## Primary Bottleneck: Communication (28% of CUDA time)

AllReduce: 1428ms out of 7911ms total CUDA time. This is TP all-reduce inside both encoder and LLM forward/backward. The LLM contributes ~90% of this.

## Memory Bottleneck: Cannot Increase Batch

- Peak at mbs=2: 54.4 GB forward, ~75 GB with backward
- mbs=4: OOM (needs ~100+ GB)
- Encoder activations: 21.7 GB held through entire LLM fwd+bwd → offload candidate
- Freeing 21.7 GB via encoder activation recompute/offload → mbs=3 might fit

## What to Try Next (Priority Order)

1. **Activation recomputation for encoder** — research how Megatron enables this, wire into MIMO path. This frees 21.7 GB → enables mbs=3 → should significantly boost TFLOPs by improving compute/communication ratio.

2. **Extreme heterogeneous: enc TP1/DP8, llm TP4/DP2** — eliminates ALL encoder all-reduce. Even if encoder is small, zero communication is always better. Requires mbs divisible by 8.

3. **Longer vision seq (4096)** — increases encoder compute fraction, makes heterogeneous TP split more impactful. Also increases memory pressure so may need recompute first.
