from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
from opd.tableio import atomic_write_text, read_json, read_records, write_json


def build_failure_analysis(
    config: dict[str, Any], predictions_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    upstream_id = verified_artifact_manifest_id(predictions_path)
    rows = read_records(predictions_path)
    failures = [row for row in rows if float(row.get("score", 0.0)) < 1.0]
    by_status = Counter(str(row.get("status", "unknown")) for row in failures)
    by_subject: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    for row in failures:
        by_subject[str(row.get("subject", "unknown"))] += 1
        by_difficulty[str(row.get("difficulty", "unknown"))] += 1
    examples = sorted(
        failures,
        key=lambda row: int(row.get("response_tokens", 0)),
        reverse=True,
    )[:20]
    failure_examples: list[dict[str, Any]] = [
        {
            "sample_id": row.get("sample_id"),
            "subject": row.get("subject"),
            "difficulty": row.get("difficulty"),
            "prompt": row.get("prompt"),
            "reference_answer": row.get("reference_answer"),
            "extracted_answer": row.get("extracted_answer"),
            "status": row.get("status"),
            "response": row.get("response"),
            "response_tokens": row.get("response_tokens"),
        }
        for row in examples
    ]
    report: dict[str, Any] = {
        "samples": len(rows),
        "failures": len(failures),
        "failure_rate": len(failures) / max(1, len(rows)),
        "status_counts": dict(sorted(by_status.items())),
        "by_subject": dict(sorted(by_subject.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "examples": failure_examples,
        "interpretation_note": (
            "These are evaluator-observed failures. Inspect raw responses before assigning a "
            "reasoning-error category, especially when status is unknown."
        ),
    }
    destination = Path(output_path)
    write_json(destination, report)
    markdown_path = destination.with_suffix(".md")
    lines = [
        "# Failure Analysis",
        "",
        f"- Samples: {len(rows)}",
        f"- Failures: {len(failures)}",
        f"- Failure rate: {report['failure_rate']:.2%}",
        "",
        "## Status",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(by_status.items()))
    lines.extend(["", "## Longest Failure Examples", ""])
    for example in failure_examples:
        lines.extend(
            [
                f"### {example['sample_id']}",
                "",
                f"- Subject: {example['subject']}",
                f"- Difficulty: {example['difficulty']}",
                f"- Status: {example['status']}",
                f"- Reference: `{example['reference_answer']}`",
                f"- Extracted: `{example['extracted_answer']}`",
                "",
                str(example["response"]),
                "",
            ]
        )
    atomic_write_text(markdown_path, "\n".join(lines))
    manifest = build_manifest(
        artifact_type="failure_analysis",
        stage="report.failures",
        config=config,
        files=[destination, markdown_path],
        record_count=len(rows),
        success_count=len(rows),
        upstream_artifact_ids=[upstream_id],
        metadata={"failure_count": len(failures), "experiment_seed": config["project"]["seed"]},
    )
    save_manifest(destination.with_suffix(".manifest.json"), manifest)
    return report


def _is_dominated(point: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    for other in points:
        if other is point:
            continue
        no_worse_cost = float(other["gpu_hours"]) <= float(point["gpu_hours"])
        no_worse_quality = float(other["accuracy"]) >= float(point["accuracy"])
        strictly_better = float(other["gpu_hours"]) < float(point["gpu_hours"]) or float(
            other["accuracy"]
        ) > float(point["accuracy"])
        if no_worse_cost and no_worse_quality and strictly_better:
            return True
    return False


def _pareto_svg(points: list[dict[str, Any]]) -> str:
    width = 800
    height = 500
    margin = 70
    maximum_hours = max((float(point["gpu_hours"]) for point in points), default=1.0)
    maximum_hours = max(maximum_hours, 1.0)

    def x_position(hours: float) -> float:
        return margin + hours / maximum_hours * (width - 2 * margin)

    def y_position(accuracy: float) -> float:
        return height - margin - accuracy * (height - 2 * margin)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle">GPU hours</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" '
        'text-anchor="middle">Accuracy</text>',
    ]
    for point in points:
        x = x_position(float(point["gpu_hours"]))
        y = y_position(float(point["accuracy"]))
        color = "#1877f2" if point["pareto_optimal"] else "#999999"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        elements.append(
            f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" font-size="13">{point["name"]}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def build_cost_pareto(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    runs = config["report"].get("runs", [])
    if not runs:
        raise ValueError("report.runs must configure at least one run")
    points: list[dict[str, Any]] = []
    upstream_ids: list[str] = []
    for run in runs:
        summary_path = Path(run["summary_path"])
        summary = read_json(summary_path)
        metric_paths = [Path(path) for path in run.get("job_metric_paths", [])]
        metrics = [read_json(path) for path in metric_paths]
        for artifact_path in [summary_path, *metric_paths]:
            manifest_id = verified_artifact_manifest_id(artifact_path)
            if manifest_id not in upstream_ids:
                upstream_ids.append(manifest_id)
        points.append(
            {
                "name": str(run["name"]),
                "run_id": summary.get("run_id"),
                "accuracy": float(summary["accuracy"]),
                "gpu_hours": sum(float(item.get("estimated_gpu_hours", 0.0)) for item in metrics),
                "teacher_tokens": sum(int(item.get("teacher_tokens", 0)) for item in metrics),
                "response_tokens": sum(int(item.get("response_tokens", 0)) for item in metrics),
                "metric_paths": [str(path) for path in metric_paths],
            }
        )
    for point in points:
        point["pareto_optimal"] = not _is_dominated(point, points)
    points.sort(key=lambda point: (float(point["gpu_hours"]), -float(point["accuracy"])))
    report = {
        "points": points,
        "single_seed_limitation": (
            "Quality values come from seed 42 runs; the Pareto frontier does not represent "
            "training-seed uncertainty."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "quality_cost.json"
    svg_path = destination / "quality_cost.svg"
    markdown_path = destination / "quality_cost.md"
    write_json(json_path, report)
    atomic_write_text(svg_path, _pareto_svg(points))
    lines = [
        "# Quality–Cost Report",
        "",
        "| Method | Accuracy | GPU hours | Teacher tokens | Pareto |",
        "|---|---:|---:|---:|---|",
    ]
    for point in points:
        lines.append(
            f"| {point['name']} | {point['accuracy']:.4f} | {point['gpu_hours']:.2f} | "
            f"{point['teacher_tokens']} | {'yes' if point['pareto_optimal'] else 'no'} |"
        )
    lines.extend(["", f"> {report['single_seed_limitation']}", ""])
    atomic_write_text(markdown_path, "\n".join(lines))
    manifest = build_manifest(
        artifact_type="quality_cost_report",
        stage="report.cost",
        config=config,
        files=[json_path, svg_path, markdown_path],
        record_count=len(points),
        success_count=len(points),
        upstream_artifact_ids=upstream_ids,
        metadata={"run_count": len(points), "experiment_seed": config["project"]["seed"]},
    )
    save_manifest(destination / "manifest.json", manifest)
    return report
