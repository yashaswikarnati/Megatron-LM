# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""YAML config loading with baseline inheritance (deep merge) for MIMO benchmarks."""

import os
from typing import List

import yaml

from .config import (
    BenchmarkConfig,
    DataSpec,
    ExperimentSpec,
    ModuleArch,
    ParallelSpec,
)

_BASELINE_FILENAME = "baseline.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    - Dict values are merged recursively.
    - All other values in override replace the base value.
    - Keys only in base are preserved.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(raw: dict) -> BenchmarkConfig:
    """Construct a BenchmarkConfig from a raw YAML dict."""
    experiment = ExperimentSpec(**raw["experiment"])

    model = raw["model"]
    encoder_arch = ModuleArch(**model["encoder"])
    llm_arch = ModuleArch(**model["llm"])

    parallelism = raw["parallelism"]
    encoder_parallel = ParallelSpec(**parallelism["encoder"])
    llm_parallel = ParallelSpec(**parallelism["llm"])

    data = DataSpec(**raw["data"])

    return BenchmarkConfig(
        experiment=experiment,
        encoder_arch=encoder_arch,
        llm_arch=llm_arch,
        encoder_parallel=encoder_parallel,
        llm_parallel=llm_parallel,
        data=data,
    )


def load_config(path: str) -> BenchmarkConfig:
    """Load a single YAML config file, merging with baseline if present.

    If ``baseline.yaml`` exists in the same directory as *path*, the experiment
    config is deep-merged on top of the baseline so that experiments only need
    to specify their overrides.
    """
    config_dir = os.path.dirname(os.path.abspath(path))
    baseline_path = os.path.join(config_dir, _BASELINE_FILENAME)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if os.path.isfile(baseline_path) and os.path.basename(path) != _BASELINE_FILENAME:
        with open(baseline_path) as f:
            baseline_raw = yaml.safe_load(f)
        raw = deep_merge(baseline_raw, raw)

    return _build_config(raw)


def load_all_configs(dir_path: str) -> List[BenchmarkConfig]:
    """Load all YAML configs in *dir_path*, each merged with baseline.

    The ``baseline.yaml`` file itself is excluded from the returned list.
    Files are sorted by name for deterministic ordering.
    """
    configs: List[BenchmarkConfig] = []
    yaml_files = sorted(
        f
        for f in os.listdir(dir_path)
        if f.endswith((".yaml", ".yml")) and f != _BASELINE_FILENAME
    )
    for filename in yaml_files:
        filepath = os.path.join(dir_path, filename)
        configs.append(load_config(filepath))
    return configs
