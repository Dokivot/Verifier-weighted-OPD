from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from opd.artifacts import load_manifest, verified_manifest_id
from opd.tableio import read_json


def _metric_values(value: Any, metric_name: str) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text == metric_name or key_text.startswith(f"{metric_name},"):
                if not isinstance(child, int | float) or isinstance(child, bool):
                    raise ValueError(f"Official metric {metric_name!r} is not numeric")
                score = float(child)
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"Official metric {metric_name!r} is outside [0, 1]: {score}")
                values.append(score)
            else:
                values.extend(_metric_values(child, metric_name))
    elif isinstance(value, list):
        for child in value:
            values.extend(_metric_values(child, metric_name))
    return values


def _task_metric(result_files: list[Path], *, task_identifier: str, metric_name: str) -> float:
    values: list[float] = []
    for path in result_files:
        payload = read_json(path)
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, dict):
            for task_key, task_value in results.items():
                if str(task_key) == task_identifier:
                    values.extend(_metric_values(task_value, metric_name))
        if not values:
            values.extend(_metric_values(payload, metric_name))
    if not values:
        raise ValueError(f"Official metric {metric_name!r} for {task_identifier!r} was not found")
    if any(score != values[0] for score in values[1:]):
        raise ValueError(
            f"Conflicting official {metric_name!r} values for {task_identifier!r}: {values}"
        )
    return values[0]


def _benchmark_name_for_task(
    task_identifier: str, task_names: list[str], registry: dict[str, Any]
) -> str:
    for name in task_names:
        entry = registry.get(name)
        if isinstance(entry, dict) and entry.get("task") == task_identifier:
            return name
    raise ValueError(f"No benchmark registry entry matches LightEval task {task_identifier!r}")


def _model_label(path: Path, config: dict[str, Any], checkpoint: str) -> str:
    benchmark = config.get("benchmark", {})
    resolved = path.parent.resolve()
    if Path(str(benchmark.get("base_output_dir", ""))).resolve() == resolved:
        return "Base"
    if Path(str(benchmark.get("candidate_output_dir", ""))).resolve() == resolved:
        return "Candidate"
    if Path(str(benchmark.get("base_math_output_dir", ""))).resolve() == resolved:
        return "Base"
    if Path(str(benchmark.get("candidate_math_output_dir", ""))).resolve() == resolved:
        return "Candidate"
    lowered = f"{path.parent.name} {checkpoint}".lower()
    return "Candidate" if any(token in lowered for token in ("candidate", "sure_k2")) else "Base"


def load_official_scores(
    config: dict[str, Any], manifest_paths: list[Path]
) -> list[dict[str, Any]]:
    benchmark_config = config.get("benchmark", {})
    registry_path = Path(benchmark_config.get("registry_path", "eval/registry.yaml"))
    registry_payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    registry = registry_payload.get("benchmarks", {})
    scores: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest_id = verified_manifest_id(manifest_path)
        manifest = load_manifest(manifest_path)
        metadata = manifest.metadata
        task_names = metadata.get("task_names")
        task_identifiers = metadata.get("task_identifiers")
        if not isinstance(task_names, list) or not task_names:
            raise ValueError(f"Official benchmark manifest has no task_names: {manifest_path}")
        if not isinstance(task_identifiers, str) or not task_identifiers:
            raise ValueError(
                f"Official benchmark manifest has no task_identifiers: {manifest_path}"
            )
        identifiers = [item for item in task_identifiers.split(",") if item]
        if len(identifiers) != len(task_names):
            raise ValueError(
                f"Official task metadata is inconsistent in {manifest_path}: "
                f"{task_names!r} vs {task_identifiers!r}"
            )
        result_files = [
            Path(file_path)
            for file_path in manifest.files
            if Path(file_path).suffix == ".json"
            and Path(file_path).name not in {"manifest.json", "command.json", "job_metrics.json"}
        ]
        if not result_files:
            raise ValueError(f"No official LightEval result JSON in {manifest_path}")
        checkpoint = str(metadata.get("checkpoint", ""))
        for task_name, task_identifier in zip(task_names, identifiers, strict=True):
            benchmark_name = _benchmark_name_for_task(task_identifier, [str(task_name)], registry)
            entry = registry[benchmark_name]
            metric_name = str(entry["primary_metric"])
            scores.append(
                {
                    "model": _model_label(manifest_path, config, checkpoint),
                    "benchmark": benchmark_name,
                    "task": task_identifier,
                    "metric": metric_name,
                    "score": _task_metric(
                        result_files,
                        task_identifier=task_identifier,
                        metric_name=metric_name,
                    ),
                    "manifest_path": str(manifest_path),
                    "manifest_id": manifest_id,
                    "generation_protocol": metadata.get("generation_protocol"),
                }
            )
    return scores
