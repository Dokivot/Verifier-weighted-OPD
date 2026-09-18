from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_manifest_id
from opd.config import normalized_online_training_config
from opd.hashing import stable_hash
from opd.monitoring.job import JobTimer
from opd.tableio import read_json, read_records
from opd.training.hf import train_hf
from opd.training.mock import train_mock
from opd.training.modes import parameter_update_mode
from opd.training.online_k2 import train_online_k2


def train(config: dict[str, Any]) -> Path:
    backend = config["training"].get("backend", "mock")
    input_path = Path(config["training"]["input_path"])
    if backend == "online_k2" or config["training"]["method"] == "sft":
        data_dir = Path(config["paths"]["data_dir"])
        upstream_manifest = (
            data_dir / "contamination/manifest.json"
            if input_path.parent.name == "contamination"
            else data_dir / "manifests/data_prepare.json"
        )
    else:
        upstream_manifest = input_path.with_suffix(".manifest.json")
    upstream_id = verified_manifest_id(upstream_manifest)
    output_dir = Path(config["training"]["output_dir"])
    metrics_path = output_dir / "job_metrics.json"
    available_record_count = len(read_records(input_path))
    max_records_value = config["training"].get("max_records")
    max_records = int(max_records_value) if max_records_value is not None else None
    if max_records is not None and max_records <= 0:
        raise ValueError("training.max_records must be positive when configured")
    selected_record_count = min(available_record_count, max_records or available_record_count)
    run_config = config
    if backend == "online_k2":
        run_config = {
            **config,
            "training": normalized_online_training_config(config["training"]),
        }
    training_run_id = stable_hash(
        {"config": run_config, "upstream_artifact_id": upstream_id}, length=20
    )
    with JobTimer(
        "training",
        metrics_path,
        method=config["training"]["method"],
        experiment_seed=int(config["project"]["seed"]),
        artifact_run_id=training_run_id,
    ) as timer:
        if backend == "mock":
            checkpoint = train_mock(config)
        elif backend == "transformers":
            checkpoint = train_hf(config)
        elif backend == "online_k2":
            checkpoint = train_online_k2(config)
        else:
            raise ValueError(f"Unsupported training backend: {backend}")
        completed_records = selected_record_count
        response_tokens = 0
        teacher_tokens = 0
        if backend == "online_k2":
            summary = read_json(output_dir / "training_summary.json")
            completed_records = int(summary["last_invocation_records"])
            response_tokens = int(summary["last_invocation_response_tokens"])
            teacher_tokens = response_tokens
        timer.add(
            records=completed_records,
            response_tokens=response_tokens,
            teacher_tokens=teacher_tokens,
        )

    files = [
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    ]
    manifest = build_manifest(
        artifact_type="checkpoint",
        stage="train",
        config=config,
        files=files,
        record_count=1,
        success_count=1,
        upstream_artifact_ids=[upstream_id],
        metadata={
            "method": config["training"]["method"],
            "backend": backend,
            "parameter_update_mode": parameter_update_mode(config["training"]),
            "checkpoint": str(checkpoint),
            "available_record_count": available_record_count,
            "selected_record_count": selected_record_count,
            "max_records": max_records,
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return checkpoint
