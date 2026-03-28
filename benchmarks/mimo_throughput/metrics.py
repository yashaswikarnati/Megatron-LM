# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Performance monitoring for MIMO throughput benchmarking.

Tracks per-iteration TFLOPs/GPU, tokens/sec, samples/sec, and peak memory.
FLOPs are estimated using the standard transformer approximation from Megatron.
"""

import json
import statistics
import time

import torch

from benchmarks.mimo_throughput.config import BenchmarkConfig


class PerformanceMonitor:
    """Tracks per-iteration performance metrics for MIMO benchmarking."""

    def __init__(self, config: BenchmarkConfig, world_size: int):
        """
        Args:
            config: BenchmarkConfig with encoder_arch, llm_arch, data specs.
            world_size: Total number of GPUs.
        """
        self.config = config
        self.world_size = world_size
        self.history = []
        self._iter_start = None

        # Pre-compute total FLOPs per iteration
        self.total_flops = self._compute_total_flops()

    def _estimate_params(self, num_layers: int, hidden_size: int, vocab_size: int) -> int:
        """Estimate parameter count for a transformer module.

        Per-layer: attention (4h^2) + FFN (8h^2) + layernorms (4h) = 12h^2 + 4h
        Embeddings: h * vocab_size (only for LLM)

        Args:
            num_layers: Number of transformer layers.
            hidden_size: Hidden dimension.
            vocab_size: Vocabulary size (0 for vision encoder).

        Returns:
            Estimated total parameter count.
        """
        h = hidden_size
        params_per_layer = 12 * h * h + 4 * h
        transformer_params = params_per_layer * num_layers
        embedding_params = h * vocab_size if vocab_size > 0 else 0
        return transformer_params + embedding_params

    def _compute_total_flops(self) -> int:
        """Compute total FLOPs per iteration across all modules.

        For each module: flops = 6 * params * seq_length * global_batch_size
        (factor of 6 = 2 for multiply-add in forward, x3 for forward + backward)

        Total = sum of all modules x num_microbatches.

        Returns:
            Total FLOPs for one full training iteration.
        """
        cfg = self.config
        gbs = cfg.global_batch_size

        # Encoder FLOPs
        enc_params = self._estimate_params(
            cfg.encoder_arch.num_layers,
            cfg.encoder_arch.hidden_size,
            cfg.encoder_arch.vocab_size,
        )
        enc_flops_per_mb = 6 * enc_params * cfg.encoder_arch.seq_length * gbs

        # LLM FLOPs
        llm_params = self._estimate_params(
            cfg.llm_arch.num_layers,
            cfg.llm_arch.hidden_size,
            cfg.llm_arch.vocab_size,
        )
        llm_flops_per_mb = 6 * llm_params * cfg.llm_arch.seq_length * gbs

        total_per_mb = enc_flops_per_mb + llm_flops_per_mb
        return total_per_mb * cfg.data.num_microbatches

    def start_iteration(self):
        """Mark the start of a training iteration."""
        torch.cuda.synchronize()
        self._iter_start = time.time()

    def end_iteration(
        self, fwd_bwd_ms: float = 0.0, opt_step_ms: float = 0.0
    ) -> dict:
        """Mark the end of a training iteration and record metrics.

        Args:
            fwd_bwd_ms: Wall-clock time for forward+backward pass in ms.
            opt_step_ms: Wall-clock time for optimizer step + zero_grad in ms.

        Returns:
            Dict with iteration number, elapsed time, TFLOPs/GPU,
            tokens/sec, samples/sec, peak memory in GB, and phase timings.
        """
        torch.cuda.synchronize()
        elapsed = time.time() - self._iter_start

        cfg = self.config
        total_samples = cfg.global_batch_size * cfg.data.num_microbatches

        tflops_per_gpu = self.total_flops / (elapsed * 1e12 * self.world_size)
        tokens_per_sec = (total_samples * cfg.llm_arch.seq_length) / elapsed
        samples_per_sec = total_samples / elapsed
        mem_gb = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            'iteration': len(self.history) + 1,
            'elapsed_sec': elapsed,
            'tflops_per_gpu': tflops_per_gpu,
            'tokens_per_sec': tokens_per_sec,
            'samples_per_sec': samples_per_sec,
            'max_memory_gb': mem_gb,
            'fwd_bwd_ms': fwd_bwd_ms,
            'opt_step_ms': opt_step_ms,
        }
        self.history.append(metrics)
        return metrics

    def get_summary(self, exclude_warmup: bool = True) -> dict:
        """Get median metrics, excluding warmup iterations.

        Args:
            exclude_warmup: If True, skip the first ``warmup_iterations``
                entries from the config.

        Returns:
            Dict with median values for all metric keys plus a count of
            iterations included in the summary.
        """
        start_idx = self.config.experiment.warmup_iterations if exclude_warmup else 0
        filtered = self.history[start_idx:]

        if not filtered:
            return {
                'median_tflops_per_gpu': 0.0,
                'median_tokens_per_sec': 0.0,
                'median_samples_per_sec': 0.0,
                'median_elapsed_sec': 0.0,
                'max_memory_gb': 0.0,
                'median_fwd_bwd_ms': 0.0,
                'median_opt_step_ms': 0.0,
                'num_iterations': 0,
            }

        return {
            'median_tflops_per_gpu': statistics.median(m['tflops_per_gpu'] for m in filtered),
            'median_tokens_per_sec': statistics.median(m['tokens_per_sec'] for m in filtered),
            'median_samples_per_sec': statistics.median(m['samples_per_sec'] for m in filtered),
            'median_elapsed_sec': statistics.median(m['elapsed_sec'] for m in filtered),
            'max_memory_gb': max(m['max_memory_gb'] for m in filtered),
            'median_fwd_bwd_ms': statistics.median(m['fwd_bwd_ms'] for m in filtered),
            'median_opt_step_ms': statistics.median(m['opt_step_ms'] for m in filtered),
            'num_iterations': len(filtered),
        }

    def save(self, filepath: str):
        """Save full per-iteration metrics and summary to JSON.

        Args:
            filepath: Output path for the JSON file.
        """
        data = {
            'config': {
                'experiment': self.config.experiment.name,
                'world_size': self.world_size,
                'encoder_arch': {
                    'num_layers': self.config.encoder_arch.num_layers,
                    'hidden_size': self.config.encoder_arch.hidden_size,
                    'seq_length': self.config.encoder_arch.seq_length,
                },
                'llm_arch': {
                    'num_layers': self.config.llm_arch.num_layers,
                    'hidden_size': self.config.llm_arch.hidden_size,
                    'seq_length': self.config.llm_arch.seq_length,
                    'vocab_size': self.config.llm_arch.vocab_size,
                },
                'encoder_parallel': {
                    'tp': self.config.encoder_parallel.tp,
                    'dp': self.config.encoder_parallel.dp,
                    'pp': self.config.encoder_parallel.pp,
                },
                'llm_parallel': {
                    'tp': self.config.llm_parallel.tp,
                    'dp': self.config.llm_parallel.dp,
                    'pp': self.config.llm_parallel.pp,
                },
                'data': {
                    'micro_batch_size': self.config.data.micro_batch_size,
                    'num_microbatches': self.config.data.num_microbatches,
                    'global_batch_size': self.config.global_batch_size,
                },
                'total_flops_per_iteration': self.total_flops,
            },
            'summary': self.get_summary(exclude_warmup=True),
            'history': self.history,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
