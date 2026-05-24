#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare hetero Energon sample-signature traces between continuous and resumed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous_trace_dir", type=Path)
    parser.add_argument("resumed_trace_dir", type=Path)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    return parser.parse_args()


def load_trace_dir(path: Path) -> dict[str, dict[int, dict]]:
    if not path.is_dir():
        raise FileNotFoundError(f"trace directory does not exist: {path}")

    traces: dict[str, dict[int, dict]] = {}
    for file_path in sorted(path.glob("*.jsonl")):
        records: dict[int, dict] = {}
        with file_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                record = json.loads(line)
                step = int(record["step"])
                if step in records:
                    raise ValueError(f"duplicate step {step} in {file_path}:{line_number}")
                records[step] = record
        traces[file_path.name] = records
    return traces


def in_range(step: int, start_step: int | None, end_step: int | None) -> bool:
    if start_step is not None and step < start_step:
        return False
    if end_step is not None and step > end_step:
        return False
    return True


def main() -> None:
    args = parse_args()
    continuous = load_trace_dir(args.continuous_trace_dir)
    resumed = load_trace_dir(args.resumed_trace_dir)

    compared = 0
    for name, resumed_records in sorted(resumed.items()):
        if name not in continuous:
            raise AssertionError(f"continuous trace is missing file present in resumed run: {name}")
        continuous_records = continuous[name]
        for step, resumed_record in sorted(resumed_records.items()):
            if not in_range(step, args.start_step, args.end_step):
                continue
            if step not in continuous_records:
                raise AssertionError(f"continuous trace {name} is missing resumed step {step}")
            continuous_record = continuous_records[step]
            if continuous_record != resumed_record:
                raise AssertionError(
                    f"trace mismatch for {name} step {step}:\n"
                    f"continuous={continuous_record}\n"
                    f"resumed={resumed_record}"
                )
            compared += 1

    if compared == 0:
        raise AssertionError("no trace records compared; check trace dirs and step filters")
    print(f"matched {compared} resumed trace records against continuous run")


if __name__ == "__main__":
    main()
