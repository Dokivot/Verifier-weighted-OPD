from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
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
    upstream_ids = [
        verified_artifact_manifest_id(path)
        for path in [*summary_paths, *comparison_paths, *training_summary_paths]
    ]
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
        record_count=len(summaries) + len(comparisons) + len(training_runs),
        success_count=len(summaries) + len(comparisons) + len(training_runs),
        upstream_artifact_ids=list(dict.fromkeys(upstream_ids)),
        metadata={"experiment_id": experiment_id, "experiment_seed": config["project"]["seed"]},
    )
    save_manifest(output_dir / f"{experiment_id}.manifest.json", manifest)
    return output_path
