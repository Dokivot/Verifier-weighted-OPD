from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
from opd.hashing import stable_hash
from opd.monitoring.job import JobTimer
from opd.tableio import write_json


def run_lighteval(
    config: dict[str, Any],
    *,
    checkpoint: str,
    output_dir: str | Path,
    task_names: list[str] | None = None,
) -> Path:
    executable = shutil.which("lighteval")
    if executable is None:
        raise FileNotFoundError("lighteval is missing; install the `eval` optional dependencies")
    benchmark = config.get("benchmark", {})
    registry_path = Path(benchmark.get("registry_path", "eval/registry.yaml"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registered = registry["benchmarks"]
    selected = task_names or list(benchmark.get("tasks", registered.keys()))
    unknown = [name for name in selected if name not in registered]
    if unknown:
        raise ValueError(f"Unknown benchmark task names: {', '.join(unknown)}")
    tasks = ",".join(str(registered[name]["task"]) for name in selected)
    custom_task_modules = {
        str(registered[name]["custom_tasks"])
        for name in selected
        if registered[name].get("custom_tasks")
    }
    if len(custom_task_modules) > 1:
        raise ValueError("Selected benchmarks require incompatible custom task modules")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint)
    upstream_ids = (
        [verified_artifact_manifest_id(checkpoint_path)] if checkpoint_path.exists() else []
    )
    model_arguments = {
        "model_name": checkpoint,
        "dtype": benchmark.get("dtype", "bfloat16"),
        "max_model_length": int(benchmark.get("max_model_length", 4096)),
        "tensor_parallel_size": int(benchmark.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(benchmark.get("gpu_memory_utilization", 0.9)),
        "seed": int(config["project"]["seed"]),
    }
    student = config["models"]["student"]
    if not checkpoint_path.exists() and checkpoint == student["name"]:
        model_arguments["revision"] = student["revision"]
    serialized_model_arguments = ",".join(
        f"{key}={value}" for key, value in model_arguments.items()
    )
    command = [
        executable,
        str(benchmark.get("launcher", "vllm")),
        serialized_model_arguments,
        tasks,
    ]
    if bool(benchmark.get("use_chat_template", True)):
        command.append("--use-chat-template")
    if bool(benchmark.get("save_details", True)):
        command.append("--save-details")
    if custom_task_modules:
        command.extend(["--custom-tasks", next(iter(custom_task_modules))])
    if benchmark.get("max_samples") is not None:
        command.extend(["--max-samples", str(int(benchmark["max_samples"]))])
    command.extend(["--output-dir", str(destination)])
    run_id = stable_hash(
        {
            "checkpoint": checkpoint,
            "tasks": selected,
            "model_arguments": model_arguments,
            "seed": config["project"]["seed"],
        },
        length=20,
    )
    command_path = destination / "command.json"
    write_json(
        command_path,
        {
            "command": command,
            "task_names": selected,
            "tasks": tasks,
            "custom_tasks": sorted(custom_task_modules),
            "run_id": run_id,
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    metrics_path = destination / "job_metrics.json"
    with JobTimer(
        "lighteval",
        metrics_path,
        artifact_run_id=run_id,
        task_names=selected,
        experiment_seed=int(config["project"]["seed"]),
    ):
        subprocess.run(command, check=True)

    output_files = [
        path for path in destination.rglob("*") if path.is_file() and path.name != "manifest.json"
    ]
    result_files = [path for path in output_files if path not in {command_path, metrics_path}]
    if not result_files:
        raise ValueError(f"LightEval produced no result files in {destination}")
    manifest = build_manifest(
        artifact_type="benchmark_run",
        stage="benchmark.lighteval",
        config=config,
        files=output_files,
        record_count=0,
        success_count=1,
        upstream_artifact_ids=upstream_ids,
        metadata={
            "run_id": run_id,
            "checkpoint": checkpoint,
            "task_names": selected,
            "task_identifiers": tasks,
            "custom_tasks": sorted(custom_task_modules),
            "result_file_count": len(result_files),
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    return save_manifest(destination / "manifest.json", manifest)
