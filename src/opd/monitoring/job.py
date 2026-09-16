from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from opd.schemas import JobMetrics
from opd.tableio import read_json, write_json


class JobTimer(AbstractContextManager["JobTimer"]):
    def __init__(self, job_type: str, output_path: str | Path, **metadata: Any) -> None:
        self.job_type = job_type
        self.output_path = Path(output_path)
        self.metadata = metadata
        self.started_at = datetime.now(UTC)
        self._start = 0.0
        self.counters = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "teacher_tokens": 0,
            "records": 0,
        }

    def __enter__(self) -> JobTimer:
        self._start = perf_counter()
        return self

    def add(self, **counters: int) -> None:
        for key, value in counters.items():
            if key not in self.counters:
                raise KeyError(f"Unknown job counter: {key}")
            self.counters[key] += value

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        finished_at = datetime.now(UTC)
        wall_time = perf_counter() - self._start
        gpu_type = "cpu"
        gpu_count = 0
        runtime_metadata = dict(self.metadata)
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = int(torch.cuda.device_count())
                gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
                gpu_type = ", ".join(sorted(set(gpu_names)))
                runtime_metadata["gpu_names"] = gpu_names
                runtime_metadata["peak_memory_bytes"] = [
                    int(torch.cuda.max_memory_allocated(index)) for index in range(gpu_count)
                ]
        except ImportError:
            pass
        safe_wall_time = max(wall_time, 1e-9)
        runtime_metadata["records_per_second"] = self.counters["records"] / safe_wall_time
        runtime_metadata["prompt_tokens_per_second"] = (
            self.counters["prompt_tokens"] / safe_wall_time
        )
        runtime_metadata["response_tokens_per_second"] = (
            self.counters["response_tokens"] / safe_wall_time
        )
        runtime_metadata["teacher_tokens_per_second"] = (
            self.counters["teacher_tokens"] / safe_wall_time
        )
        started_at: Any = self.started_at
        cumulative_wall_time = wall_time
        cumulative_gpu_hours = wall_time * gpu_count / 3600.0
        cumulative_counters = dict(self.counters)
        artifact_run_id = self.metadata.get("artifact_run_id")
        if artifact_run_id and self.output_path.exists():
            previous = read_json(self.output_path)
            previous_metadata = previous.get("metadata", {})
            if previous_metadata.get("artifact_run_id") == artifact_run_id:
                started_at = previous["started_at"]
                cumulative_wall_time += float(previous.get("wall_time_seconds", 0.0))
                cumulative_gpu_hours += float(previous.get("estimated_gpu_hours", 0.0))
                for key in cumulative_counters:
                    cumulative_counters[key] += int(previous.get(key, 0))
                runtime_metadata = {**previous_metadata, **runtime_metadata}
                runtime_metadata["invocations"] = int(previous_metadata.get("invocations", 1)) + 1
                if gpu_count == 0 and int(previous.get("gpu_count", 0)) > 0:
                    gpu_count = int(previous["gpu_count"])
                    gpu_type = str(previous.get("gpu_type", gpu_type))
        runtime_metadata.setdefault("invocations", 1)
        cumulative_safe_wall_time = max(cumulative_wall_time, 1e-9)
        runtime_metadata["records_per_second"] = (
            cumulative_counters["records"] / cumulative_safe_wall_time
        )
        runtime_metadata["prompt_tokens_per_second"] = (
            cumulative_counters["prompt_tokens"] / cumulative_safe_wall_time
        )
        runtime_metadata["response_tokens_per_second"] = (
            cumulative_counters["response_tokens"] / cumulative_safe_wall_time
        )
        runtime_metadata["teacher_tokens_per_second"] = (
            cumulative_counters["teacher_tokens"] / cumulative_safe_wall_time
        )
        metrics = JobMetrics(
            job_type=self.job_type,
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=cumulative_wall_time,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            estimated_gpu_hours=cumulative_gpu_hours,
            metadata={**runtime_metadata, "failed": exc_type is not None},
            **cumulative_counters,
        )
        write_json(self.output_path, metrics.model_dump(mode="json"))
        return None
