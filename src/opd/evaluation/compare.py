from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
from opd.evaluation.statistics import mcnemar_exact, paired_bootstrap
from opd.tableio import read_json, read_records, write_json


def compare_runs(
    config: dict[str, Any],
    baseline_summary_path: str | Path,
    candidate_summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    upstream_ids = [
        verified_artifact_manifest_id(baseline_summary_path),
        verified_artifact_manifest_id(candidate_summary_path),
    ]
    baseline_summary = read_json(baseline_summary_path)
    candidate_summary = read_json(candidate_summary_path)
    baseline_rows = {
        row["sample_id"]: row for row in read_records(baseline_summary["predictions_path"])
    }
    candidate_rows = {
        row["sample_id"]: row for row in read_records(candidate_summary["predictions_path"])
    }
    common = sorted(baseline_rows.keys() & candidate_rows.keys())
    if not common:
        raise ValueError("Evaluation runs have no common sample ids")
    baseline_scores = [float(baseline_rows[sample_id]["score"]) for sample_id in common]
    candidate_scores = [float(candidate_rows[sample_id]["score"]) for sample_id in common]
    bootstrap = paired_bootstrap(baseline_scores, candidate_scores, seed=seed)
    mcnemar = mcnemar_exact(
        [bool(score) for score in baseline_scores],
        [bool(score) for score in candidate_scores],
    )
    comparison = {
        "baseline_run_id": baseline_summary["run_id"],
        "candidate_run_id": candidate_summary["run_id"],
        "common_samples": len(common),
        "baseline_accuracy": sum(baseline_scores) / len(common),
        "candidate_accuracy": sum(candidate_scores) / len(common),
        "mean_difference": bootstrap.mean_difference,
        "ci95": [bootstrap.ci_low, bootstrap.ci_high],
        "mcnemar": mcnemar,
        "single_seed_limitation": (
            "This interval measures evaluation-sample uncertainty and does not estimate "
            "training-seed variance."
        ),
    }
    destination = Path(output_path)
    write_json(destination, comparison)
    manifest = build_manifest(
        artifact_type="evaluation_comparison",
        stage="evaluate.compare",
        config=config,
        files=[destination],
        record_count=len(common),
        success_count=len(common),
        upstream_artifact_ids=list(dict.fromkeys(upstream_ids)),
        metadata={
            "baseline_run_id": baseline_summary["run_id"],
            "candidate_run_id": candidate_summary["run_id"],
            "experiment_seed": seed,
        },
    )
    save_manifest(destination.with_suffix(".manifest.json"), manifest)
    return comparison
