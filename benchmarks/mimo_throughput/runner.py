# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CLI entry point for MIMO throughput benchmarking.

Usage:
    # Single experiment
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        -m benchmarks.mimo_throughput.runner --config configs/fan_in_pp1.yaml

    # Sweep all experiments in a directory
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        -m benchmarks.mimo_throughput.runner --configs-dir configs/
"""

import argparse
import csv
import json
import logging
import os

import torch
import torch.distributed as dist

from benchmarks.mimo_throughput.config import BenchmarkConfig
from benchmarks.mimo_throughput.config_loader import load_all_configs, load_config
from benchmarks.mimo_throughput.training import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _save_result(summary: dict, config: BenchmarkConfig, results_dir: str):
    """Save a single experiment's metrics to JSON.

    Args:
        summary: Summary dict returned by run_benchmark (includes '_monitor').
        config: The BenchmarkConfig for this experiment.
        results_dir: Output directory.
    """
    _ensure_dir(results_dir)
    monitor = summary.pop('_monitor', None)
    filepath = os.path.join(results_dir, f"{config.experiment.name}.json")
    if monitor is not None:
        monitor.save(filepath)
    else:
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {filepath}")


def aggregate_results_csv(results_dir: str, output_path: str = None):
    """Combine per-experiment JSON results into a single CSV.

    Reads all ``*.json`` files in *results_dir*, extracts the summary row,
    and writes a combined CSV suitable for comparison.

    Args:
        results_dir: Directory containing per-experiment JSON files.
        output_path: Path for the output CSV. Defaults to
            ``<results_dir>/summary.csv``.
    """
    if output_path is None:
        output_path = os.path.join(results_dir, "summary.csv")

    json_files = sorted(
        f for f in os.listdir(results_dir) if f.endswith(".json")
    )

    if not json_files:
        logger.warning(f"No JSON result files found in {results_dir}")
        return

    rows = []
    for filename in json_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath) as f:
            data = json.load(f)

        cfg = data.get('config', {})
        summary = data.get('summary', {})
        enc_p = cfg.get('encoder_parallel', {})
        llm_p = cfg.get('llm_parallel', {})
        enc_a = cfg.get('encoder_arch', {})
        llm_a = cfg.get('llm_arch', {})
        data_cfg = cfg.get('data', {})

        rows.append({
            'experiment': cfg.get('experiment', filename.replace('.json', '')),
            'world_size': cfg.get('world_size', ''),
            # Encoder arch
            'enc_layers': enc_a.get('num_layers', ''),
            'enc_hidden': enc_a.get('hidden_size', ''),
            'enc_seq_len': enc_a.get('seq_length', ''),
            # LLM arch
            'llm_layers': llm_a.get('num_layers', ''),
            'llm_hidden': llm_a.get('hidden_size', ''),
            'llm_seq_len': llm_a.get('seq_length', ''),
            'llm_vocab': llm_a.get('vocab_size', ''),
            # Parallelism
            'enc_tp': enc_p.get('tp', ''),
            'enc_dp': enc_p.get('dp', ''),
            'enc_pp': enc_p.get('pp', ''),
            'llm_tp': llm_p.get('tp', ''),
            'llm_dp': llm_p.get('dp', ''),
            'llm_pp': llm_p.get('pp', ''),
            # Data
            'micro_batch_size': data_cfg.get('micro_batch_size', ''),
            'num_microbatches': data_cfg.get('num_microbatches', ''),
            'global_batch_size': data_cfg.get('global_batch_size', ''),
            # Metrics
            'median_tflops_per_gpu': _fmt(summary.get('median_tflops_per_gpu')),
            'median_tokens_per_sec': _fmt(summary.get('median_tokens_per_sec')),
            'median_samples_per_sec': _fmt(summary.get('median_samples_per_sec')),
            'median_elapsed_sec': _fmt(summary.get('median_elapsed_sec')),
            'max_memory_gb': _fmt(summary.get('max_memory_gb')),
            'median_fwd_bwd_ms': _fmt(summary.get('median_fwd_bwd_ms')),
            'median_opt_step_ms': _fmt(summary.get('median_opt_step_ms')),
            'num_iterations': summary.get('num_iterations', ''),
        })

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Aggregated CSV written to {output_path}")


def _fmt(value):
    """Format a numeric value for CSV output, rounding floats."""
    if value is None:
        return ''
    if isinstance(value, float):
        return round(value, 3)
    return value


def main():
    parser = argparse.ArgumentParser(description='MIMO Throughput Benchmark')
    parser.add_argument(
        '--config', type=str, help='Path to a single YAML config file.'
    )
    parser.add_argument(
        '--configs-dir', type=str, help='Directory of YAML configs to sweep.'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results).',
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False,
        help='Enable torch.profiler for the iterations specified by --profile-steps.',
    )
    parser.add_argument(
        '--profile-steps',
        type=str,
        default=None,
        help='Iteration range to profile, e.g. "5-7" (0-based). Requires --profile.',
    )
    args = parser.parse_args()

    if not args.config and not args.configs_dir:
        parser.error("Provide either --config or --configs-dir")

    # Parse --profile-steps into a tuple
    profile_steps = None
    if args.profile:
        if args.profile_steps is None:
            parser.error("--profile requires --profile-steps START-END (e.g. --profile-steps 5-7)")
        parts = args.profile_steps.split('-')
        if len(parts) != 2:
            parser.error("--profile-steps must be in START-END format (e.g. 5-7)")
        profile_steps = (int(parts[0]), int(parts[1]))

    # Initialize distributed (torch.distributed should already be initialized by torchrun)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    rank = dist.get_rank()

    if args.config:
        # Single experiment
        config = load_config(args.config)
        config.validate()
        if rank == 0:
            logger.info(f"Running single experiment: {config.experiment.name}")
        summary = run_benchmark(
            config, profile_steps=profile_steps, results_dir=args.results_dir
        )
        if rank == 0:
            _save_result(summary, config, args.results_dir)

    elif args.configs_dir:
        # Sweep all configs in directory
        configs = load_all_configs(args.configs_dir)
        if rank == 0:
            logger.info(f"Loaded {len(configs)} experiment configs from {args.configs_dir}")

        for cfg in configs:
            cfg.validate()
            if rank == 0:
                logger.info(f"--- Running experiment: {cfg.experiment.name} ---")
            summary = run_benchmark(
                cfg, profile_steps=profile_steps, results_dir=args.results_dir
            )
            if rank == 0:
                _save_result(summary, cfg, args.results_dir)

        # Aggregate all results into a single CSV
        if rank == 0:
            aggregate_results_csv(args.results_dir)

    if rank == 0:
        logger.info("All benchmarks complete.")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
