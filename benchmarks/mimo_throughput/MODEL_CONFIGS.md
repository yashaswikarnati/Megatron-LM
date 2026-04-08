# Proxy Model Configs for Colocated Throughput Benchmarking

Representative proxy configs based on real OSS architectures (Llama, Qwen, SigLIP, InternViT).
Goal: understand how encoder/LLM size ratio and vision seq length affect heterogeneous parallelism.

## Vision Encoder Sizes

Based on real ViT/SigLIP/InternViT architectures. Our proxy uses TransformerBlock (same compute profile).

| Name | Layers | Hidden | Heads | ~Params | Reference |
|------|--------|--------|-------|---------|-----------|
| ViT-S (small) | 24 | 1024 | 16 | 0.3B | CLIP ViT-L/14, InternViT-300M |
| ViT-M (medium) | 27 | 1152 | 16 | 0.4B | SigLIP-SO400M |
| ViT-L (large) | 40 | 1408 | 16 | 1.0B | Scaled SigLIP / custom |
| ViT-XL (xlarge) | 45 | 3200 | 25 | 6.0B | InternViT-6B |

## LLM Sizes

Based on Llama-3.x and Qwen-2.5 architectures. Use standard MHA (not GQA) for simplicity in proxy.

| Name | Layers | Hidden | Heads | Vocab | ~Params | Reference |
|------|--------|--------|-------|-------|---------|-----------|
| LLM-1B | 16 | 2048 | 32 | 32000 | 1.0B | Llama-3.2-1B |
| LLM-3B | 28 | 3072 | 24 | 32000 | 3.0B | Llama-3.2-3B |
| LLM-7B | 32 | 4096 | 32 | 32000 | 6.6B | Llama-3.1-8B / Qwen-2.5-7B |
| LLM-14B | 48 | 5120 | 40 | 32000 | 14B | Qwen-2.5-14B |
| LLM-70B | 80 | 8192 | 64 | 32000 | 66B | Llama-3.1-70B (multinode) |

## Vision Sequence Lengths

| Name | Tokens/image | Represents |
|------|-------------|------------|
| Low-res | 256 | InternVL pixel-unshuffle, small images |
| Standard | 576 | SigLIP-384px, LLaVA-336px |
| High-res | 1024 | InternViT-448px, high-res images |
| Ultra-res | 2048 | Dynamic high-res tiling |
| Extreme | 4096 | Multi-image / video frames |

## Recommended Experiment Configs

### Tier 1: Single Node 8xH100 (primary focus)

These configs fit in 8 GPUs with TP+DP scaling.

| Config ID | Vision | LLM | Vision seq | Total seq | Enc/LLM ratio | Notes |
|-----------|--------|-----|-----------|-----------|---------------|-------|
| S1 | ViT-M (0.4B) | LLM-1B | 576 | 4096 | 0.4x | Small VLM, fast iteration |
| S2 | ViT-L (1.0B) | LLM-3B | 576 | 8192 | 0.33x | Medium VLM |
| S3 | ViT-L (1.0B) | LLM-7B | 1024 | 8192 | 0.15x | **Current benchmark config** |
| S4 | ViT-L (1.0B) | LLM-7B | 2048 | 8192 | 0.15x (more vision compute) | Higher vision compute fraction |
| S5 | ViT-XL (6.0B) | LLM-7B | 576 | 8192 | 0.86x | Large encoder ≈ LLM size |
| S6 | ViT-XL (6.0B) | LLM-7B | 1024 | 8192 | 0.86x (more vision compute) | Encoder-dominated |
| S7 | ViT-M (0.4B) | LLM-7B | 576 | 8192 | 0.06x | Small encoder, large LLM |
| S8 | ViT-L (1.0B) | LLM-14B | 1024 | 8192 | 0.07x | 1B+14B, LLM-dominated |

### Tier 2: Multinode (future, for reference)

| Config ID | Vision | LLM | Vision seq | Total seq | Min GPUs | Notes |
|-----------|--------|-----|-----------|-----------|----------|-------|
| M1 | ViT-XL (6.0B) | LLM-14B | 1024 | 8192 | 16 | 2-node |
| M2 | ViT-XL (6.0B) | LLM-70B | 1024 | 8192 | 64 | 8-node |

## Key Dimensions to Ablate

For each config, sweep:
1. **Parallelism**: homo (same TP/DP) vs hetero (different TP/DP per module)
2. **Vision seq length**: 576 vs 1024 vs 2048 (changes encoder compute fraction)
3. **Batch size**: mbs=1,2,4 (limited by memory)
4. **Encoder/LLM size ratio**: small encoder + large LLM vs balanced vs encoder-dominated

## Intuition: When Does Heterogeneous Win?

Heterogeneous parallelism should help most when:
- **Encoder hidden size << LLM hidden size** → encoder wastes TP bandwidth at high TP
- **Vision seq is long** → encoder is significant fraction of compute
- **Memory is tight** → different TP allows better memory distribution
- **DP-heavy encoder** → better data throughput for encoder, which is often smaller

Configs S5/S6 (6B encoder + 7B LLM) are the most promising for showing heterogeneous benefits — the encoder is nearly as large as the LLM, so encoder TP optimization has real impact.
