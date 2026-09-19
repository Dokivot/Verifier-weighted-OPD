from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from opd.artifacts import load_manifest, verified_manifest_id
from opd.config import config_hash, load_config
from opd.hashing import file_sha256
from opd.runtime_profile import BLACKWELL_CUDA_VERSION, BLACKWELL_TORCH_VERSION
from opd.tableio import read_records
from opd.training.online_k2 import _checkpoint_storage_plan


class ReadinessError(RuntimeError):
    """Raised when a formal-training readiness gate cannot be satisfied."""


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _expected_eval_manifest_hash(
    config: dict[str, Any],
    *,
    name: str,
    source: dict[str, Any],
) -> str:
    fetch_config = {
        **config,
        "evaluation_fetch": {
            "name": name,
            "dataset_name": source["dataset_name"],
            "subset": source.get("dataset_subset"),
            "split": source.get("split", "test"),
            "revision": source["revision"],
        },
    }
    return config_hash(fetch_config)


def _check_manifest(
    path: Path,
    *,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    artifact_id = verified_manifest_id(path)
    manifest = load_manifest(path)
    if expected_config_hash is not None and manifest.config_hash != expected_config_hash:
        raise ReadinessError(
            f"Manifest config hash mismatch for {path}: "
            f"{manifest.config_hash} != {expected_config_hash}"
        )
    return {
        "path": str(path),
        "artifact_id": artifact_id,
        "config_hash": manifest.config_hash,
        "record_count": manifest.record_count,
        "metadata": manifest.metadata,
    }


def _check_count(path: Path, expected: int) -> int:
    actual = len(read_records(path))
    if actual != expected:
        raise ReadinessError(f"Record count mismatch for {path}: {actual} != {expected}")
    return actual


def _check_data(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["artifact_dir"])
    data_dir = Path(config["paths"]["data_dir"])
    expected_hash = config_hash(config)
    prepare_manifest_path = data_dir / "manifests/data_prepare.json"
    prepare = _check_manifest(prepare_manifest_path, expected_config_hash=expected_hash)
    counts = {key: int(value) for key, value in config["data"]["counts"].items()}
    split_counts = prepare["metadata"].get("split_counts", {})
    for key, expected in {
        "dataset_name": config["data"].get("dataset_name"),
        "dataset_revision": config["data"].get("dataset_revision"),
        "dataset_split": config["data"].get("dataset_split", "train"),
        "filters": config["data"].get("filters", {}),
    }.items():
        if prepare["metadata"].get(key) != expected:
            raise ReadinessError(
                f"Prepared data manifest has stale {key}: "
                f"{prepare['metadata'].get(key)!r} != {expected!r}"
            )
    if prepare["record_count"] != sum(counts.values()):
        raise ReadinessError("Prepared data manifest total record count is stale")
    files: dict[str, int] = {}
    for split, expected in counts.items():
        path = data_dir / f"curated/{split}.{config['data'].get('format', 'parquet')}"
        files[str(path)] = _check_count(path, expected)
        if int(split_counts.get(split, -1)) != expected:
            raise ReadinessError(
                f"Prepared split count mismatch for {split}: "
                f"{split_counts.get(split)} != {expected}"
            )
        manifest = load_manifest(prepare_manifest_path)
        if str(path) not in manifest.files:
            raise ReadinessError(f"Prepared split is absent from manifest: {path}")

    eval_manifests: dict[str, Any] = {}
    for name, source_value in config.get("evaluation_data", {}).items():
        source = dict(source_value)
        extension = config["data"].get("format", "parquet")
        input_path = Path(config["paths"]["data_dir"]) / f"eval/{name}.{extension}"
        manifest_path = input_path.with_suffix(".manifest.json")
        expected_eval_hash = _expected_eval_manifest_hash(config, name=name, source=source)
        evaluation_manifest = _check_manifest(
            manifest_path,
            expected_config_hash=expected_eval_hash,
        )
        metadata = evaluation_manifest["metadata"]
        for key, expected in {
            "name": name,
            "dataset_name": source["dataset_name"],
            "split": source.get("split", "test"),
            "revision": source["revision"],
        }.items():
            if metadata.get(key) != expected:
                raise ReadinessError(
                    f"Evaluation manifest {manifest_path} has {key}={metadata.get(key)!r}; "
                    f"expected {expected!r}"
                )
        eval_manifests[name] = {
            **evaluation_manifest,
            "record_count": _check_count(input_path, evaluation_manifest["record_count"]),
            "revision": source["revision"],
            "checksum": file_sha256(input_path),
        }

    contamination_path = root / "data/contamination/manifest.json"
    contamination = _check_manifest(contamination_path, expected_config_hash=expected_hash)
    contamination_manifest = load_manifest(contamination_path)
    report_path = root / "data/contamination/report.json"
    report = _load_object(report_path)
    train_path = Path(config["contamination"]["train_path"])
    eval_paths = [str(Path(path)) for path in config["contamination"].get("eval_paths", [])]
    if report.get("train_path") != str(train_path) or report.get("eval_paths") != eval_paths:
        raise ReadinessError("Contamination report inputs do not match the current config")
    if report.get("train_checksum") != file_sha256(train_path):
        raise ReadinessError("Contamination report train checksum is stale")
    expected_eval_checksums = {path: file_sha256(Path(path)) for path in eval_paths}
    if report.get("eval_checksums") != expected_eval_checksums:
        raise ReadinessError("Contamination report evaluation checksums are stale")
    clean_path = root / "data/contamination/train_clean.parquet"
    clean_count = _check_count(clean_path, int(report["clean_count"]))
    if contamination_manifest.record_count != int(report["train_count"]):
        raise ReadinessError("Contamination manifest input record count is stale")
    if contamination_manifest.success_count != clean_count:
        raise ReadinessError("Contamination manifest clean record count is stale")
    expected_upstream_ids = {prepare["artifact_id"]}
    expected_upstream_ids.update(
        evaluation["artifact_id"] for evaluation in eval_manifests.values()
    )
    if not expected_upstream_ids.issubset(set(contamination_manifest.upstream_artifact_ids)):
        raise ReadinessError("Contamination manifest upstream artifact chain is incomplete")
    if (
        str(clean_path) not in contamination_manifest.files
        or str(report_path) not in contamination_manifest.files
    ):
        raise ReadinessError("Contamination manifest does not cover all audit outputs")

    return {
        "prepare_manifest": prepare,
        "prepared_splits": files,
        "evaluation_manifests": eval_manifests,
        "contamination_manifest": contamination,
        "clean_train": {"path": str(clean_path), "record_count": clean_count},
    }


def _expected_generation_protocol(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["benchmark"]
    generation = config["benchmark"]["generation"]
    return {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "temperature": float(generation.get("temperature", 1.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "top_k": int(generation.get("top_k", -1)),
        "max_model_length": int(benchmark["max_model_length"]),
        "max_prompt_tokens": (
            int(benchmark["max_prompt_tokens"])
            if benchmark.get("max_prompt_tokens") is not None
            else None
        ),
        "use_chat_template": bool(benchmark.get("use_chat_template", True)),
        "seed": int(config["project"]["seed"]),
        "enable_thinking": bool(benchmark.get("enable_thinking", False)),
        "thinking_marker": benchmark.get("thinking_marker"),
        "stop_strategy": (
            "chat_template_eos"
            if bool(benchmark.get("use_chat_template", True))
            else "task_stop_sequence"
        ),
    }


def _expected_internal_generation_protocol(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    generation = evaluation["generation"]
    return {
        "temperature": float(generation.get("temperature", 0.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "top_k": int(generation.get("top_k", -1)),
        "max_new_tokens": int(generation.get("max_new_tokens", 512)),
        "max_model_length": int(evaluation.get("max_model_length", 4096)),
        "enable_thinking": bool(generation.get("enable_thinking", False)),
        "thinking_marker": generation.get("thinking_marker"),
        "chat_template": "tokenizer_default_if_available",
    }


def _check_base_evaluations(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["evaluation"]["output_dir"])
    expected_hash = config_hash(config)
    expected_internal_protocol = _expected_internal_generation_protocol(config)
    result: dict[str, Any] = {}
    for suite in ("math500", "amc23"):
        summary_path = output_dir / suite / "summary.json"
        manifest_path = output_dir / suite / "manifest.json"
        summary = _load_object(summary_path)
        if summary.get("suite") != suite:
            raise ReadinessError(f"Base evaluation suite mismatch: {summary_path}")
        if summary.get("model_name") != config["evaluation"]["model_name"]:
            raise ReadinessError(f"Base evaluation model mismatch: {summary_path}")
        if summary.get("model_revision") != config["evaluation"]["model_revision"]:
            raise ReadinessError(f"Base evaluation revision mismatch: {summary_path}")
        if summary.get("generation_protocol") != expected_internal_protocol:
            raise ReadinessError(f"Base evaluation generation protocol mismatch: {summary_path}")
        manifest = _check_manifest(manifest_path, expected_config_hash=expected_hash)
        eval_name = suite
        data_format = config["data"].get("format", "parquet")
        eval_input = Path(config["paths"]["data_dir"]) / "eval" / f"{eval_name}.{data_format}"
        eval_manifest = _check_manifest(eval_input.with_suffix(".manifest.json"))
        if eval_manifest["artifact_id"] not in load_manifest(manifest_path).upstream_artifact_ids:
            raise ReadinessError(f"Base evaluation is not linked to current {eval_name} input")
        if manifest["metadata"].get("tokenizer_revision") != config["evaluation"].get(
            "tokenizer_revision", config["evaluation"]["model_revision"]
        ):
            raise ReadinessError(f"Base evaluation tokenizer revision mismatch: {manifest_path}")
        if manifest["record_count"] != int(summary.get("samples", -1)):
            raise ReadinessError(f"Base evaluation manifest count mismatch: {manifest_path}")
        result[suite] = {"summary": summary, "manifest": manifest}

    benchmark_root = Path(config["paths"]["artifact_dir"]) / "benchmark"
    expected_protocol = _expected_generation_protocol(config)
    benchmark_results: dict[str, Any] = {}
    for name, task_names in (("base", ["ifeval"]), ("base_math500", ["math500"])):
        benchmark_path = benchmark_root / name / "manifest.json"
        benchmark = _check_manifest(benchmark_path, expected_config_hash=expected_hash)
        metadata = benchmark["metadata"]
        if metadata.get("checkpoint") != config["models"]["student"]["name"]:
            raise ReadinessError(f"Base benchmark checkpoint mismatch: {benchmark_path}")
        if metadata.get("task_names") != task_names:
            raise ReadinessError(
                f"Base benchmark must contain exactly {task_names}: {benchmark_path}"
            )
        if metadata.get("generation_protocol") != expected_protocol:
            raise ReadinessError(f"Base benchmark generation protocol mismatch: {benchmark_path}")
        if metadata.get("model_revision") not in {None, config["models"]["student"]["revision"]}:
            raise ReadinessError(f"Base benchmark model revision mismatch: {benchmark_path}")
        benchmark_results[name] = benchmark
    return {
        "evaluations": result,
        "official_benchmark": benchmark_results,
    }


def _companion_config_path(formal_config_path: Path, name: str) -> Path:
    """Resolve the smoke or pilot config paired with a formal config."""
    stem = formal_config_path.stem
    if stem.endswith("_24h"):
        stem = stem[: -len("_24h")]
    return formal_config_path.parent / f"{stem}_{name}.yaml"


def _load_variant_config(formal_config_path: Path, name: str) -> dict[str, Any]:
    variant_path = _companion_config_path(formal_config_path, name)
    return load_config(variant_path)


def _hardware_is_target(hardware: dict[str, Any]) -> bool:
    name = str(hardware.get("device_name", "")).upper()
    capability = tuple(int(value) for value in hardware.get("compute_capability", []))
    architectures = {str(value) for value in hardware.get("compiled_architectures", [])}
    return (
        int(hardware.get("gpu_count", 0)) == 1
        and "RTX PRO 6000" in name
        and "BLACKWELL" in name
        and float(hardware.get("total_memory_gib", 0.0)) >= 90.0
        and capability >= (12, 0)
        and bool(architectures & {"sm_120", "compute_120"})
        and hardware.get("torch_version") == BLACKWELL_TORCH_VERSION
        and hardware.get("torch_cuda") == BLACKWELL_CUDA_VERSION
    )


def _check_training_gate(
    formal_config: dict[str, Any],
    formal_config_path: Path,
    *,
    name: str,
    expected_steps: int,
) -> dict[str, Any]:
    variant_config = _load_variant_config(formal_config_path, name)
    variant_hash = config_hash(variant_config)
    output_dir = Path(variant_config["training"]["output_dir"])
    manifest_path = output_dir / "manifest.json"
    manifest = _check_manifest(manifest_path, expected_config_hash=variant_hash)
    loaded_manifest = load_manifest(manifest_path)
    summary_path = output_dir / "training_summary.json"
    if str(summary_path) not in loaded_manifest.files:
        raise ReadinessError(f"{name} training summary is absent from its manifest")
    summary = _load_object(summary_path)
    if (
        summary.get("status") != "completed"
        or int(summary.get("global_step", -1)) != expected_steps
    ):
        raise ReadinessError(
            f"{name} training gate did not complete exactly {expected_steps} steps"
        )
    if summary.get("config_hash") != variant_hash:
        raise ReadinessError(f"{name} summary config hash is stale")
    protocol = summary.get("generation_protocol")
    expected_protocol = {
        "enable_thinking": bool(formal_config["training"].get("enable_thinking", False)),
        "thinking_marker": formal_config["training"].get("thinking_marker"),
    }
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected_protocol.items()
    ):
        raise ReadinessError(f"{name} training thinking protocol is stale")
    if not summary.get("run_id") or not summary.get("hardware_fingerprint"):
        raise ReadinessError(f"{name} summary is missing run or hardware identity")
    hardware = dict(summary.get("hardware", {}))
    if not _hardware_is_target(hardware):
        raise ReadinessError(f"{name} did not run on the target RTX PRO 6000 Blackwell GPU")
    model_identity = dict(summary.get("model_identity", {}))
    expected_identity = {
        "student_name": formal_config["models"]["student"]["name"],
        "student_revision": formal_config["models"]["student"]["revision"],
        "student_tokenizer_revision": formal_config["models"]["student"]["tokenizer_revision"],
        "teacher_name": formal_config["models"]["teacher"]["name"],
        "teacher_revision": formal_config["models"]["teacher"]["revision"],
        "teacher_tokenizer_revision": formal_config["models"]["teacher"]["tokenizer_revision"],
    }
    if model_identity != expected_identity:
        raise ReadinessError(f"{name} model revisions do not match the formal config")
    expected_training = formal_config["training"]
    if hardware.get("dtype") != expected_training.get("dtype", "bfloat16"):
        raise ReadinessError(f"{name} compute dtype does not match the formal config")
    if hardware.get("master_parameter_dtype") != expected_training.get(
        "master_parameter_dtype", "float32"
    ):
        raise ReadinessError(f"{name} master parameter dtype does not match the formal config")
    if hardware.get("device") != f"cuda:{int(expected_training.get('device_index', 0))}":
        raise ReadinessError(f"{name} device does not match the formal config")
    input_path = Path(variant_config["training"]["input_path"])
    if summary.get("input_path") != str(input_path):
        raise ReadinessError(f"{name} input path does not match its variant config")
    if summary.get("input_checksum") != file_sha256(input_path):
        raise ReadinessError(f"{name} input checksum is stale")
    return {
        "manifest": manifest,
        "summary": summary,
        "hardware": hardware,
        "hardware_fingerprint": summary["hardware_fingerprint"],
        "variant_config_hash": variant_hash,
        "output_dir": str(output_dir),
    }


def _check_disk_gate(config: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(pilot["output_dir"])
    pilot_plan_path = output_dir / "checkpoint_storage_plan.json"
    pilot_plan = _load_object(pilot_plan_path)
    if not bool(pilot_plan.get("passed")):
        raise ReadinessError("Pilot checkpoint storage plan did not pass")
    parameter_count = int(pilot_plan["parameter_count"])
    checkpointing = dict(config.get("checkpointing", {}))
    training = config["training"]
    free_bytes = shutil.disk_usage(Path(config["paths"]["artifact_dir"])).free
    plan = _checkpoint_storage_plan(
        parameter_count=parameter_count,
        max_steps=int(training["max_steps"]),
        milestone_steps={int(value) for value in training.get("milestone_steps", [])},
        checkpointing=checkpointing,
        free_bytes=free_bytes,
    )
    if not plan["passed"]:
        raise ReadinessError(
            "Current free disk space is insufficient for the formal peak checkpoint footprint: "
            f"{plan['available_free_disk_gib']:.1f} < {plan['required_free_disk_gib']:.1f} GiB"
        )
    return {"formal_plan": plan, "pilot_plan": pilot_plan}


def validate_sure_k2_readiness(config_path: str | Path) -> dict[str, Any]:
    formal_path = Path(config_path)
    config = load_config(formal_path)
    report: dict[str, Any] = {
        "config": str(formal_path),
        "config_hash": config_hash(config),
        "status": "failed",
        "checks": {},
    }
    failures: list[str] = []

    def run_check(name: str, function: Any) -> None:
        try:
            report["checks"][name] = {"status": "passed", "details": function()}
        except Exception as exc:
            report["checks"][name] = {"status": "failed", "error": str(exc)}
            failures.append(f"{name}: {exc}")

    run_check("data_manifests", lambda: _check_data(config))
    run_check("base_evaluations", lambda: _check_base_evaluations(config))
    smoke_config = load_config(_companion_config_path(formal_path, "smoke"))
    pilot_config = load_config(_companion_config_path(formal_path, "pilot"))
    run_check(
        "smoke",
        lambda: _check_training_gate(
            config,
            formal_path,
            name="smoke",
            expected_steps=int(smoke_config["training"]["max_steps"]),
        ),
    )
    run_check(
        "pilot",
        lambda: _check_training_gate(
            config,
            formal_path,
            name="pilot",
            expected_steps=int(pilot_config["training"]["max_steps"]),
        ),
    )

    smoke_check = report["checks"].get("smoke", {})
    pilot_check = report["checks"].get("pilot", {})
    if smoke_check.get("status") == "passed" and pilot_check.get("status") == "passed":
        smoke = smoke_check["details"]
        pilot = pilot_check["details"]
        if smoke["hardware_fingerprint"] != pilot["hardware_fingerprint"]:
            report["checks"]["hardware_consistency"] = {
                "status": "failed",
                "error": "Smoke and pilot hardware fingerprints differ",
            }
            failures.append("hardware_consistency: Smoke and pilot hardware fingerprints differ")
        else:
            report["checks"]["hardware_consistency"] = {
                "status": "passed",
                "details": {"hardware_fingerprint": smoke["hardware_fingerprint"]},
            }
        pilot_budget_path = Path(pilot["output_dir"]) / "pilot_budget_report.json"
        try:
            budget = _load_object(pilot_budget_path)
            if budget.get("status") != "passed":
                raise ReadinessError("Pilot budget status is not passed")
            if budget.get("config_hash") != config_hash(pilot_config):
                raise ReadinessError("Pilot budget report config hash is stale")
            if budget.get("formal_config_hash") != config_hash(config):
                raise ReadinessError("Pilot budget report formal config hash is stale")
            if budget.get("run_id") != pilot["summary"].get("run_id"):
                raise ReadinessError("Pilot budget report run identity is stale")
            if budget.get("hardware_fingerprint") != pilot["hardware_fingerprint"]:
                raise ReadinessError("Pilot budget report hardware identity is stale")
            pilot_manifest = load_manifest(Path(pilot["output_dir"]) / "manifest.json")
            if str(pilot_budget_path) not in pilot_manifest.files:
                raise ReadinessError("Pilot budget report is absent from the pilot manifest")
            report["checks"]["pilot_budget"] = {"status": "passed", "details": budget}
        except Exception as exc:
            report["checks"]["pilot_budget"] = {"status": "failed", "error": str(exc)}
            failures.append(f"pilot_budget: {exc}")
        if pilot_check.get("status") == "passed":
            run_check("disk", lambda: _check_disk_gate(config, pilot))

    report["status"] = "passed" if not failures else "failed"
    report_path = Path(config["paths"]["report_dir"]) / "readiness/readiness_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failures:
        raise ReadinessError("; ".join(failures))
    return report
