# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Low-overhead rank-local timeline tracing for pipeline debug runs."""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO

import torch


@dataclass
class PipelineTimelineRecorder:
    """Collect rank-local timeline events and write them as JSONL."""

    output_dir: Path
    rank: int
    world_size: int
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cuda_events: bool = False
    nvtx: bool = False
    iteration_start: Optional[int] = None
    iteration_end: Optional[int] = None
    iteration: Optional[int] = None
    _records: list[dict[str, Any]] = field(default_factory=list)
    _context_stack: list[dict[str, Any]] = field(default_factory=list)
    _file: Optional[TextIO] = None

    @contextlib.contextmanager
    def record(self, event: str, cuda: bool = False, **metadata) -> Iterator[None]:
        """Record one event duration without synchronizing by default."""
        event_metadata = {}
        for context_metadata in self._context_stack:
            event_metadata.update(context_metadata)
        event_metadata.update(metadata)
        nvtx_enabled = self.nvtx and torch.cuda.is_available()
        cuda_start = None
        cuda_end = None
        use_cuda_events = self.cuda_events and cuda and torch.cuda.is_available()

        if nvtx_enabled:
            torch.cuda.nvtx.range_push(self._format_nvtx(event, event_metadata))
        if use_cuda_events:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        start_time_ns = time.time_ns()
        start_perf_ns = time.perf_counter_ns()
        ok = True
        self._context_stack.append(event_metadata)
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            end_perf_ns = time.perf_counter_ns()
            self._context_stack.pop()
            if use_cuda_events:
                cuda_end.record()
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()

            record = {
                "event": event,
                "iteration": self.iteration,
                "rank": self.rank,
                "world_size": self.world_size,
                "role": self.role,
                "start_time_ns": start_time_ns,
                "start_perf_ns": start_perf_ns,
                "duration_us": (end_perf_ns - start_perf_ns) / 1000.0,
                "ok": ok,
            }
            record.update(self.metadata)
            record.update(event_metadata)
            if use_cuda_events:
                record["_cuda_start"] = cuda_start
                record["_cuda_end"] = cuda_end
            self._records.append(record)

    def flush(self) -> None:
        """Write pending events for this rank."""
        if not self._records:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._file is None:
            path = self.output_dir / f"rank{self.rank:05d}.jsonl"
            self._file = path.open("a", encoding="utf-8")

        for record in sorted(self._records, key=lambda item: item["start_time_ns"]):
            cuda_start = record.pop("_cuda_start", None)
            cuda_end = record.pop("_cuda_end", None)
            if cuda_start is not None and cuda_end is not None:
                cuda_end.synchronize()
                record["cuda_ms"] = cuda_start.elapsed_time(cuda_end)
            self._file.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        self._file.flush()
        self._records.clear()

    def close(self) -> None:
        """Flush and close the rank-local output file."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def is_active(self) -> bool:
        """Return whether the current iteration should be recorded."""
        if self.iteration_start is None and self.iteration_end is None:
            return True
        if self.iteration is None:
            return False
        if self.iteration_start is not None and self.iteration < self.iteration_start:
            return False
        if self.iteration_end is not None and self.iteration > self.iteration_end:
            return False
        return True

    def _format_nvtx(self, event: str, metadata: dict[str, Any]) -> str:
        microbatch = metadata.get("microbatch")
        if microbatch is None:
            return f"{event}/iter={self.iteration}/role={self.role}"
        return f"{event}/iter={self.iteration}/mb={microbatch}/role={self.role}"


_RECORDER: Optional[PipelineTimelineRecorder] = None


def configure_pipeline_timeline(
    *,
    enabled: bool,
    output_dir: str,
    rank: int,
    world_size: int,
    role: str,
    metadata: Optional[dict[str, Any]] = None,
    cuda_events: bool = False,
    nvtx: bool = False,
    iteration_start: Optional[int] = None,
    iteration_end: Optional[int] = None,
) -> None:
    """Configure the process-local pipeline timeline recorder."""
    global _RECORDER
    close_pipeline_timeline()
    if not enabled:
        _RECORDER = None
        return
    _RECORDER = PipelineTimelineRecorder(
        output_dir=Path(output_dir),
        rank=rank,
        world_size=world_size,
        role=role,
        metadata=metadata or {},
        cuda_events=cuda_events,
        nvtx=nvtx,
        iteration_start=iteration_start,
        iteration_end=iteration_end,
    )


def set_pipeline_timeline_iteration(iteration: int) -> None:
    """Set the current training iteration attached to subsequent events."""
    if _RECORDER is not None:
        _RECORDER.iteration = iteration


def flush_pipeline_timeline() -> None:
    """Flush pending rank-local timeline events."""
    if _RECORDER is not None:
        _RECORDER.flush()


def close_pipeline_timeline() -> None:
    """Close the process-local timeline recorder."""
    global _RECORDER
    if _RECORDER is not None:
        _RECORDER.close()
        _RECORDER = None


def timeline_event(event: str, cuda: bool = False, **metadata):
    """Return a no-op or recording context manager for one timeline event."""
    if _RECORDER is None or not _RECORDER.is_active():
        return contextlib.nullcontext()
    return _RECORDER.record(event, cuda=cuda, **metadata)


def is_pipeline_timeline_active() -> bool:
    """Return whether the current rank/iteration is writing pipeline timeline events."""
    return _RECORDER is not None and _RECORDER.is_active()


def timeline_instant(event: str, **metadata) -> None:
    """Write a zero-work timeline event with metadata for the current rank/iteration."""
    if _RECORDER is None or not _RECORDER.is_active():
        return
    with _RECORDER.record(event, **metadata):
        pass


def _jsonable(value):
    """Convert common non-JSON values used in trace metadata."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Size):
        return list(value)
    return value
