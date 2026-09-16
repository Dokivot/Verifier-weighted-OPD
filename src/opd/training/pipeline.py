from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_manifest_id
from opd.hashing import stable_hash
from opd.monitoring.job import JobTimer
from opd.tableio import read_records
from opd.training.hf import train_hf
from opd.training.mock import train_mock


def train(config: dict[str, Any]) -> Path:
    backend = config["training"].get("backend", "mock")
    input_path = Path(config["training"]["input_path"])
    if config["training"]["method"] == "sft":
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
    record_count = len(read_records(input_path))
    training_run_id = stable_hash(
        {"config": config, "upstream_artifact_id": upstream_id}, length=20
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
        else:
            raise ValueError(f"Unsupported training backend: {backend}")
        timer.add(records=record_count)

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
            "checkpoint": str(checkpoint),
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return checkpoint
