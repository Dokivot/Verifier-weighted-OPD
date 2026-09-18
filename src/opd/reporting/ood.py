from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, load_manifest, save_manifest, verified_manifest_id
from opd.hashing import stable_hash
from opd.tableio import read_json, write_json


def _metric_key(key: str, metric_name: str) -> bool:
    return key == metric_name or key.startswith(f"{metric_name},")


def _walk_metrics(
    value: Any,
    *,
    metric_name: str,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            if _metric_key(key_text, metric_name):
                if not isinstance(child, int | float) or isinstance(child, bool):
                    raise ValueError(
                        f"LightEval metric {metric_name} at {'/'.join(child_path)} is not numeric"
                    )
                score = float(child)
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"LightEval metric {metric_name} at {'/'.join(child_path)} "
                        f"is outside [0, 1]: {score}"
                    )
                yield child_path, score
            else:
                yield from _walk_metrics(child, metric_name=metric_name, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_metrics(
                child,
                metric_name=metric_name,
                path=(*path, str(index)),
            )


def _task_from_path(path: tuple[str, ...]) -> str:
    for component in reversed(path[:-1]):
        lowered = component.lower()
        if "ifeval" in lowered or "ifeval" in lowered.replace("_", ""):
            return component
    return "ifeval"


def _load_ifeval_metrics(
    run_dir: Path,
    *,
    metric_name: str,
) -> dict[str, float]:
    json_paths = sorted(
        path
        for path in run_dir.rglob("*.json")
        if path.name not in {"manifest.json", "command.json", "job_metrics.json"}
    )
    if not json_paths:
        raise ValueError(f"No LightEval result JSON found in {run_dir}")

    found: dict[str, list[float]] = {}
    for path in json_paths:
        payload = read_json(path)
        for metric_path, score in _walk_metrics(payload, metric_name=metric_name):
            task = _task_from_path(metric_path)
            found.setdefault(task, []).append(score)
    if not found:
        raise ValueError(
            f"LightEval metric {metric_name!r} was not found in result JSON under {run_dir}"
        )

    metrics: dict[str, float] = {}
    for task, scores in sorted(found.items()):
        if any(score != scores[0] for score in scores[1:]):
            raise ValueError(
                f"Conflicting {metric_name} values were found for task {task} in {run_dir}"
            )
        metrics[task] = scores[0]
    return metrics


def _load_run(
    run_dir: str | Path,
    *,
    metric_name: str,
) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    manifest_id = verified_manifest_id(manifest_path)
    manifest = load_manifest(manifest_path)
    metadata = manifest.metadata
    task_names = metadata.get("task_names")
    task_identifiers = metadata.get("task_identifiers")
    generation_protocol = metadata.get("generation_protocol")
    if not isinstance(task_names, list) or not task_names:
        raise ValueError(f"LightEval manifest has no task_names: {manifest_path}")
    if not isinstance(task_identifiers, str) or not task_identifiers:
        raise ValueError(f"LightEval manifest has no task_identifiers: {manifest_path}")
    if not isinstance(generation_protocol, dict):
        raise ValueError(f"LightEval manifest has no generation_protocol: {manifest_path}")
    if not any("ifeval" in str(name).lower() for name in task_names):
        raise ValueError(f"Run is not an IFEval run: {manifest_path}")
    return {
        "path": str(directory),
        "manifest_id": manifest_id,
        "run_id": metadata.get("run_id", manifest.artifact_id),
        "task_names": [str(name) for name in task_names],
        "task_identifiers": task_identifiers,
        "generation_protocol": generation_protocol,
        "metrics": _load_ifeval_metrics(directory, metric_name=metric_name),
    }


def _protocol_signature(run: dict[str, Any]) -> str:
    return stable_hash(
        {
            "task_names": run["task_names"],
            "task_identifiers": run["task_identifiers"],
            "generation_protocol": run["generation_protocol"],
        }
    )


def _write_failed_report(
    config: dict[str, Any],
    output_path: Path,
    report: dict[str, Any],
    upstream_ids: list[str],
) -> None:
    write_json(output_path, report)
    manifest = build_manifest(
        artifact_type="ood_regression",
        stage="report.ood_regression",
        config=config,
        files=[output_path],
        record_count=0,
        success_count=0,
        failure_count=len(report["gate"]["failures"]),
        upstream_artifact_ids=upstream_ids,
        metadata={"status": "failed", "metric": report["metric"]},
    )
    save_manifest(output_path.with_suffix(".manifest.json"), manifest)


def build_ood_regression(
    config: dict[str, Any],
    *,
    baseline_dir: str | Path,
    candidate_dirs: dict[str, str | Path],
    output_path: str | Path,
) -> Path:
    if not candidate_dirs:
        raise ValueError("At least one candidate IFEval run is required")
    regression_config = config.get("benchmark", {}).get("ood_regression", {})
    metric_name = str(regression_config.get("metric", "prompt_level_strict_acc"))
    max_allowed_drop = float(regression_config.get("max_allowed_drop", 0.0))
    if not 0.0 <= max_allowed_drop <= 1.0:
        raise ValueError("benchmark.ood_regression.max_allowed_drop must be in [0, 1]")

    baseline = _load_run(baseline_dir, metric_name=metric_name)
    candidates = {
        name: _load_run(path, metric_name=metric_name)
        for name, path in sorted(candidate_dirs.items())
    }
    runs = {"base": baseline, **candidates}
    protocol_signatures = {name: _protocol_signature(run) for name, run in runs.items()}
    if len(set(protocol_signatures.values())) != 1:
        raise ValueError(
            "Base and candidate IFEval runs do not share the same task or generation protocol"
        )

    baseline_tasks = set(baseline["metrics"])
    if not baseline_tasks:
        raise ValueError("Base IFEval result has no task-level metrics")
    failures: list[str] = []
    deltas: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        candidate_tasks = set(candidate["metrics"])
        if candidate_tasks != baseline_tasks:
            raise ValueError(
                f"Base/candidate IFEval task mismatch for {name}: "
                f"{sorted(baseline_tasks)} != {sorted(candidate_tasks)}"
            )
        task_deltas = {
            task: candidate["metrics"][task] - baseline["metrics"][task]
            for task in sorted(baseline_tasks)
        }
        overall_base = sum(baseline["metrics"].values()) / len(baseline_tasks)
        overall_candidate = sum(candidate["metrics"].values()) / len(baseline_tasks)
        overall_delta = overall_candidate - overall_base
        for task, delta in task_deltas.items():
            if delta < -max_allowed_drop:
                failures.append(
                    f"{name}/{task} regression {delta:.6f} exceeds allowed drop "
                    f"{-max_allowed_drop:.6f}"
                )
        if overall_delta < -max_allowed_drop:
            failures.append(
                f"{name}/overall regression {overall_delta:.6f} exceeds allowed drop "
                f"{-max_allowed_drop:.6f}"
            )
        deltas[name] = {
            "base": baseline["metrics"],
            "candidate": candidate["metrics"],
            "delta": task_deltas,
            "overall_base": overall_base,
            "overall_candidate": overall_candidate,
            "overall_delta": overall_delta,
            "regression_gate_passed": not any(
                failure.startswith(f"{name}/") for failure in failures
            ),
        }

    destination = Path(output_path)
    report = {
        "report_type": "ood_regression",
        "status": "failed" if failures else "passed",
        "metric": metric_name,
        "baseline": baseline,
        "candidates": candidates,
        "deltas": deltas,
        "protocol_signature": next(iter(protocol_signatures.values())),
        "protocols": protocol_signatures,
        "gate": {
            "max_allowed_drop": max_allowed_drop,
            "failures": failures,
            "passed": not failures,
        },
    }
    upstream_ids = [baseline["manifest_id"]] + [run["manifest_id"] for run in candidates.values()]
    if failures:
        _write_failed_report(config, destination, report, upstream_ids)
        raise ValueError(
            "OOD regression gate failed; see "
            f"{destination} for the non-publishable diagnostic report"
        )

    write_json(destination, report)
    manifest = build_manifest(
        artifact_type="ood_regression",
        stage="report.ood_regression",
        config=config,
        files=[destination],
        record_count=len(deltas),
        success_count=len(deltas),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "status": "passed",
            "metric": metric_name,
            "max_allowed_drop": max_allowed_drop,
            "candidate_names": sorted(candidates),
        },
    )
    save_manifest(destination.with_suffix(".manifest.json"), manifest)
    return destination
