from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    load_manifest,
    save_manifest,
    verified_artifact_manifest_id,
    verified_manifest_id,
)
from opd.tableio import atomic_write_text, read_json, write_json


def _checkpoint_files(checkpoint: Path) -> list[Path]:
    if checkpoint.is_file():
        return [checkpoint]
    patterns = (
        "adapter_model*.safetensors",
        "adapter_model*.bin",
        "model*.safetensors",
        "pytorch_model*.bin",
        "checkpoint.json",
    )
    return [path for pattern in patterns for path in checkpoint.glob(pattern)]


def evaluate_promotion(
    config: dict[str, Any],
    *,
    baseline_summary_path: str | Path,
    candidate_summary_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    promotion = config.get("promotion", {})
    baseline = read_json(baseline_summary_path)
    candidate = read_json(candidate_summary_path)
    checkpoint = Path(checkpoint_path)
    training_output = Path(config["training"]["output_dir"])
    training_manifest_path = training_output / "manifest.json"
    training_manifest_id = verified_manifest_id(training_manifest_path)
    baseline_manifest_id = verified_artifact_manifest_id(baseline_summary_path)
    candidate_manifest_id = verified_artifact_manifest_id(candidate_summary_path)
    training_manifest = load_manifest(training_manifest_path)
    training_summary_path = training_output / "training_summary.json"
    training_summary = read_json(training_summary_path) if training_summary_path.exists() else {}

    checks: dict[str, bool] = {}
    checks["checkpoint_exists"] = checkpoint.exists() and bool(_checkpoint_files(checkpoint))
    checks["artifact_complete"] = training_manifest.success_count == 1
    checks["lineage_present"] = bool(training_manifest.upstream_artifact_ids)
    checks["git_commit_known"] = training_manifest.git_commit != "unknown" or bool(
        promotion.get("allow_unknown_git_commit", False)
    )

    losses = [float(item["loss"]) for item in training_summary.get("history", [])]
    checks["finite_training_loss"] = all(math.isfinite(loss) for loss in losses)
    gradient_norms = [
        float(item["gradient_norm"])
        for item in training_summary.get("history", [])
        if "gradient_norm" in item
    ]
    checks["finite_gradient_norm"] = all(math.isfinite(norm) for norm in gradient_norms)
    baseline_accuracy = float(baseline["accuracy"])
    candidate_accuracy = float(candidate["accuracy"])
    maximum_regression = float(promotion.get("max_accuracy_regression", 0.0))
    checks["accuracy_gate"] = candidate_accuracy >= baseline_accuracy - maximum_regression

    baseline_length = float(baseline.get("mean_response_tokens", 0.0))
    candidate_length = float(candidate.get("mean_response_tokens", 0.0))
    if baseline_length <= 0:
        length_ratio = 1.0
    else:
        length_ratio = candidate_length / baseline_length
    minimum_ratio = float(promotion.get("min_length_ratio", 0.7))
    maximum_ratio = float(promotion.get("max_length_ratio", 1.3))
    checks["length_gate"] = minimum_ratio <= length_ratio <= maximum_ratio

    baseline_unknown = int(baseline.get("status_counts", {}).get("unknown", 0)) / max(
        1, int(baseline.get("samples", 0))
    )
    candidate_unknown = int(candidate.get("status_counts", {}).get("unknown", 0)) / max(
        1, int(candidate.get("samples", 0))
    )
    maximum_unknown_increase = float(promotion.get("max_unknown_rate_increase", 0.02))
    checks["unknown_rate_gate"] = candidate_unknown <= baseline_unknown + maximum_unknown_increase

    decision = {
        "promotable": all(checks.values()),
        "checks": checks,
        "training_manifest_id": training_manifest_id,
        "checkpoint": str(checkpoint),
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_difference": candidate_accuracy - baseline_accuracy,
        "length_ratio": length_ratio,
        "baseline_unknown_rate": baseline_unknown,
        "candidate_unknown_rate": candidate_unknown,
        "experiment_seed": int(config["project"]["seed"]),
    }
    destination = Path(output_path)
    write_json(destination, decision)
    marker = destination.parent / "PROMOTABLE"
    if decision["promotable"]:
        atomic_write_text(marker, f"{candidate.get('run_id', 'unknown')}\n")
    elif marker.exists():
        marker.unlink()
    files = [destination]
    if marker.exists():
        files.append(marker)
    manifest = build_manifest(
        artifact_type="checkpoint_promotion",
        stage="checkpoint.promote",
        config=config,
        files=files,
        record_count=1,
        success_count=int(bool(decision["promotable"])),
        failure_count=int(not bool(decision["promotable"])),
        upstream_artifact_ids=list(
            dict.fromkeys([training_manifest_id, baseline_manifest_id, candidate_manifest_id])
        ),
        metadata={
            "promotable": decision["promotable"],
            "checkpoint": str(checkpoint),
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    save_manifest(destination.with_suffix(".manifest.json"), manifest)
    return decision
