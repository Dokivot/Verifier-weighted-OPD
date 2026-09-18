from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
)
from opd.reporting.official import load_official_scores
from opd.tableio import atomic_write_text, read_json, write_json


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _training_curve_svg(name: str, history: list[dict[str, Any]]) -> str:
    points = [
        (int(item["step"]), float(item.get("composite_score", item["validation_accuracy"])))
        for item in history
        if "validation_accuracy" in item
    ]
    width = 760
    height = 420
    margin = 60
    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<text x="30" y="50">No validation points for {name}</text></svg>'
        )
    maximum_step = max(step for step, _score in points)
    path_points = []
    for step, score in points:
        x = margin + step / max(1, maximum_step) * (width - 2 * margin)
        y = height - margin - score * (height - 2 * margin)
        path_points.append(f"{x:.1f},{y:.1f}")
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2}" y="28" text-anchor="middle">{name}</text>',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
            f'y2="{height - margin}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
            'stroke="black"/>',
            f'<polyline points="{" ".join(path_points)}" fill="none" stroke="#1877f2" '
            'stroke-width="3"/>',
            f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">Optimizer step</text>',
            '<text x="18" y="210" transform="rotate(-90 18 210)" '
            'text-anchor="middle">Composite validation score</text>',
            "</svg>",
        ]
    )


def build_report(config: dict[str, Any], experiment_id: str) -> Path:
    report_config = config["report"]
    summary_paths = [Path(path) for path in report_config.get("summary_paths", [])]
    comparison_paths = [Path(path) for path in report_config.get("comparison_paths", [])]
    training_runs = list(report_config.get("training_runs", []))
    training_summary_paths = [Path(run["summary_path"]) for run in training_runs]
    summaries = [read_json(path) for path in summary_paths]
    comparisons = [read_json(path) for path in comparison_paths]
    training_summaries = [read_json(path) for path in training_summary_paths]
    official_manifest_paths = [
        Path(path) for path in report_config.get("official_benchmark_manifests", [])
    ]
    official_benchmarks = load_official_scores(config, official_manifest_paths)
    calibration_path_value = report_config.get("internal_verifier_calibration_path")
    internal_verifier: dict[str, Any]
    if calibration_path_value:
        calibration_path = Path(str(calibration_path_value))
        calibration_manifest_id = verified_artifact_manifest_id(calibration_path)
        internal_verifier = {
            "calibration_path": str(calibration_path),
            "manifest_id": calibration_manifest_id,
            "calibration": read_json(calibration_path),
        }
    else:
        internal_verifier = {
            "status": "not_configured",
            "note": (
                "Internal verifier outputs are auxiliary training telemetry. Run verifier "
                "calibration with manual labels before claiming verifier accuracy."
            ),
        }
    upstream_ids = [
        verified_artifact_manifest_id(path)
        for path in [*summary_paths, *comparison_paths, *training_summary_paths]
    ]
    upstream_ids.extend(item["manifest_id"] for item in official_benchmarks)
    if "manifest_id" in internal_verifier:
        upstream_ids.append(internal_verifier["manifest_id"])
    ood_regression_path_value = report_config.get("ood_regression_path")
    ood_regression: dict[str, Any]
    if ood_regression_path_value:
        ood_regression_path = Path(str(ood_regression_path_value))
        ood_regression_manifest_id = verified_artifact_manifest_id(ood_regression_path)
        ood_regression = {
            "path": str(ood_regression_path),
            "manifest_id": ood_regression_manifest_id,
            "report": read_json(ood_regression_path),
        }
        if ood_regression["report"].get("status") != "passed":
            raise ValueError(
                "Refusing to build a publishable report from a failed OOD regression gate"
            )
        upstream_ids.append(ood_regression_manifest_id)
    else:
        ood_regression = {"status": "not_configured"}
    output_dir = Path(report_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_id}.md"

    lines = [
        f"# Experiment Report: {experiment_id}",
        "",
        "> All training runs use the pre-registered single seed. Bootstrap intervals below do not "
        "measure training-seed variance.",
        "",
        "## Evaluation Runs",
        "",
        "| Model | Suite | Samples | Accuracy | Mean response tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {model} | {suite} | {samples} | {accuracy:.4f} | {tokens:.1f} |".format(
                model=summary["model_name"],
                suite=summary["suite"],
                samples=summary["samples"],
                accuracy=summary["accuracy"],
                tokens=summary["mean_response_tokens"],
            )
        )
    lines.extend(["", "## Paired Comparisons", ""])
    if not comparisons:
        lines.append("No paired comparisons were configured.")
    for comparison in comparisons:
        lines.extend(
            [
                f"- `{comparison['baseline_run_id']}` → `{comparison['candidate_run_id']}`: "
                f"Δ={comparison['mean_difference']:.4f}, "
                f"95% CI=[{comparison['ci95'][0]:.4f}, {comparison['ci95'][1]:.4f}], "
                f"McNemar p={comparison['mcnemar']['p_value']:.4g}",
            ]
        )
    lines.extend(["", "## Official Benchmark", ""])
    if official_benchmarks:
        lines.extend(
            [
                "| Model | Benchmark | Task | Metric | Score |",
                "|---|---|---|---|---:|",
            ]
        )
        for item in official_benchmarks:
            lines.append(
                "| {model} | {benchmark} | `{task}` | `{metric}` | {score:.4f} |".format(**item)
            )
        lines.append("")
        lines.append("Official artifacts:")
        lines.extend(
            f"- `{item['manifest_path']}` (manifest `{item['manifest_id']}`)"
            for item in official_benchmarks
        )
    else:
        lines.append("No official benchmark artifacts were configured.")
    lines.extend(["", "## OOD Regression", ""])
    if "report" in ood_regression:
        ood_report = ood_regression["report"]
        lines.extend(
            [
                f"- Metric: `{ood_report['metric']}`",
                f"- Gate status: `{ood_report['status']}`",
                f"- Maximum allowed drop: `{ood_report['gate']['max_allowed_drop']:.4f}`",
            ]
        )
        for name, delta in sorted(ood_report.get("deltas", {}).items()):
            lines.append(
                f"- `{name}` overall delta: `{delta['overall_delta']:.4f}` "
                f"(gate passed: `{delta['regression_gate_passed']}`)."
            )
    else:
        lines.append("No OOD regression report was configured.")
    lines.extend(["", "## Internal Verifier Calibration", ""])
    if "calibration_path" in internal_verifier:
        calibration = internal_verifier["calibration"]
        calibration_section = calibration.get("internal_verifier", {})
        lines.extend(
            [
                f"- Calibration artifact: `{internal_verifier['calibration_path']}`",
                f"- Calibration status: `{calibration.get('status', 'unknown')}`",
                f"- Reviewed samples: `{calibration_section.get('reviewed_samples', 0)}`",
                "- These metrics are auxiliary and are not the official benchmark score.",
            ]
        )
    else:
        lines.append(str(internal_verifier["note"]))
    lines.extend(["", "## Training Curves", ""])
    curve_paths: list[Path] = []
    if not training_runs:
        lines.append("No training curves were configured.")
    for run, summary in zip(training_runs, training_summaries, strict=True):
        name = str(run["name"])
        curve_path = output_dir / f"{experiment_id}_{_safe_name(name)}_curve.svg"
        atomic_write_text(curve_path, _training_curve_svg(name, summary.get("history", [])))
        curve_paths.append(curve_path)
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Best step: {summary.get('best_step', 'unknown')}",
                f"- Best validation score: {summary.get('best_validation_score', 'unknown')}",
                f"- Curve: `{curve_path}`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Required Interpretation",
            "",
            "- Compare quality at the same Teacher-token and training-token budget.",
            "- Report negative or statistically unclear results directly.",
            "- Inspect subject, difficulty, length, verifier status, and failure-type slices.",
            "- Do not treat the single-seed bootstrap interval as evidence of training stability.",
            "",
        ]
    )
    atomic_write_text(output_path, "\n".join(lines))
    json_path = output_dir / f"{experiment_id}.json"
    write_json(
        json_path,
        {
            "experiment_id": experiment_id,
            "summaries": summaries,
            "comparisons": comparisons,
            "official_benchmark": official_benchmarks,
            "ood_regression": ood_regression,
            "internal_verifier": internal_verifier,
            "training_runs": [
                {"name": run["name"], "summary": summary}
                for run, summary in zip(training_runs, training_summaries, strict=True)
            ],
        },
    )
    manifest = build_manifest(
        artifact_type="experiment_report",
        stage="report.build",
        config=config,
        files=[output_path, json_path, *curve_paths],
        record_count=(
            len(summaries) + len(comparisons) + len(training_runs) + len(official_benchmarks)
        ),
        success_count=(
            len(summaries) + len(comparisons) + len(training_runs) + len(official_benchmarks)
        ),
        upstream_artifact_ids=list(dict.fromkeys(upstream_ids)),
        metadata={"experiment_id": experiment_id, "experiment_seed": config["project"]["seed"]},
    )
    save_manifest(output_dir / f"{experiment_id}.manifest.json", manifest)
    return output_path
