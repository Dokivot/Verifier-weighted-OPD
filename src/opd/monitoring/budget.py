from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.exceptions import OPDError
from opd.tableio import read_json


def check_budget(config: dict[str, Any]) -> dict[str, Any]:
    roots = {
        Path(config["paths"][key])
        for key in ("artifact_dir", "data_dir", "checkpoint_dir")
        if key in config["paths"]
    }
    metric_paths = sorted(
        {
            metric_path
            for root in roots
            if root.exists()
            for metric_path in root.rglob("job_metrics.json")
        }
    )
    gpu_hours = 0.0
    jobs: list[dict[str, Any]] = []
    for path in metric_paths:
        metrics = read_json(path)
        hours = float(metrics.get("estimated_gpu_hours", 0.0))
        gpu_hours += hours
        jobs.append(
            {
                "path": str(path),
                "job_type": metrics.get("job_type", "unknown"),
                "gpu_hours": hours,
            }
        )
    target = float(config["project"]["gpu_hour_target"])
    hard_cap = float(config["project"]["gpu_hour_hard_cap"])
    result = {
        "gpu_hours": gpu_hours,
        "target": target,
        "hard_cap": hard_cap,
        "remaining_to_target": max(0.0, target - gpu_hours),
        "remaining_to_hard_cap": max(0.0, hard_cap - gpu_hours),
        "status": "hard_cap_exceeded"
        if gpu_hours >= hard_cap
        else "target_exceeded"
        if gpu_hours >= target
        else "within_budget",
        "jobs": jobs,
    }
    if gpu_hours >= hard_cap:
        raise OPDError(
            f"GPU-hour hard cap reached: {gpu_hours:.2f} >= {hard_cap:.2f}; stop new training jobs"
        )
    return result
