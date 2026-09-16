from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
    verified_manifest_id,
)
from opd.hashing import stable_hash
from opd.monitoring.job import JobTimer
from opd.rollout.hf_backend import HFRolloutBackend
from opd.rollout.mock import MockRolloutBackend
from opd.rollout.vllm_backend import VLLMRolloutBackend
from opd.schemas import PromptRecord
from opd.tableio import read_records, write_json, write_records
from opd.verifier.math import MathVerifier


def _backend(config: dict[str, Any]) -> Any:
    evaluation = config["evaluation"]
    if evaluation["backend"] == "mock":
        return MockRolloutBackend(
            model_name=evaluation.get("model_name", "mock-eval-model"),
            model_revision=evaluation.get("model_revision", "mock-v1"),
        )
    if evaluation["backend"] == "transformers":
        return HFRolloutBackend(
            model_name=evaluation["model_name"],
            model_revision=evaluation["model_revision"],
            tokenizer_revision=evaluation.get("tokenizer_revision", evaluation["model_revision"]),
            generation_config=evaluation["generation"],
            dtype=evaluation.get("dtype", "bfloat16"),
        )
    if evaluation["backend"] == "vllm":
        return VLLMRolloutBackend(
            model_name=evaluation["model_name"],
            model_revision=evaluation["model_revision"],
            tokenizer_revision=evaluation.get("tokenizer_revision", evaluation["model_revision"]),
            generation_config=evaluation["generation"],
            tensor_parallel_size=int(evaluation.get("tensor_parallel_size", 1)),
            dtype=str(evaluation.get("dtype", "bfloat16")),
            max_model_length=int(evaluation.get("max_model_length", 4096)),
            gpu_memory_utilization=float(evaluation.get("gpu_memory_utilization", 0.9)),
        )
    raise ValueError(f"Unsupported evaluation backend: {evaluation['backend']}")


def evaluate(config: dict[str, Any], *, suite: str) -> Path:
    evaluation = config["evaluation"]
    suite_config = evaluation["suites"][suite]
    prompt_path = Path(suite_config["input_path"])
    data_dir = Path(config["paths"]["data_dir"])
    prompt_manifest_id = (
        verified_manifest_id(data_dir / "manifests" / "data_prepare.json")
        if prompt_path.parent.name == "curated"
        else verified_artifact_manifest_id(prompt_path)
    )
    upstream_ids = [prompt_manifest_id]
    model_path = Path(evaluation["model_name"])
    if model_path.exists():
        checkpoint_manifest_id = verified_artifact_manifest_id(model_path)
        if checkpoint_manifest_id not in upstream_ids:
            upstream_ids.append(checkpoint_manifest_id)
    output_dir = Path(evaluation["output_dir"]) / suite
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "job_metrics.json"
    prompts = [PromptRecord.model_validate(row) for row in read_records(prompt_path)]
    max_samples = suite_config.get("max_samples")
    if max_samples is not None:
        prompts = prompts[: int(max_samples)]
    backend = _backend(config)
    batch_size = int(evaluation.get("batch_size", 8))
    verifier = MathVerifier()
    rows: list[dict[str, Any]] = []
    seed = int(config["project"]["seed"])
    run_id = stable_hash(
        {
            "model": backend.model_name,
            "revision": backend.model_revision,
            "suite": suite,
            "seed": seed,
        },
        length=20,
    )

    with JobTimer(
        "evaluation",
        metrics_path,
        suite=suite,
        experiment_seed=seed,
        artifact_run_id=run_id,
    ) as timer:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            started = perf_counter()
            generations = backend.generate([record.problem for record in batch], seed=seed + start)
            elapsed_ms = int((perf_counter() - started) * 1000)
            for record, generation in zip(batch, generations, strict=True):
                verification = verifier.verify(
                    rollout_id=f"eval_{record.sample_id}",
                    sample_id=record.sample_id,
                    response=generation.text,
                    reference_answer=record.reference_answer,
                )
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "subject": record.subject,
                        "difficulty": record.difficulty,
                        "prompt": record.problem,
                        "reference_answer": record.reference_answer,
                        "response": generation.text,
                        "score": verification.score,
                        "status": verification.status.value,
                        "extracted_answer": verification.extracted_answer,
                        "response_tokens": generation.response_tokens,
                        "latency_ms": max(1, elapsed_ms // max(1, len(batch))),
                    }
                )
                timer.add(
                    records=1,
                    prompt_tokens=generation.prompt_tokens,
                    response_tokens=generation.response_tokens,
                )

    write_records(output_path, rows)
    by_subject: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(float(row["score"]))
        by_difficulty[str(row["difficulty"])].append(float(row["score"]))
    summary = {
        "run_id": run_id,
        "suite": suite,
        "model_name": backend.model_name,
        "model_revision": backend.model_revision,
        "seed": seed,
        "samples": len(rows),
        "accuracy": sum(float(row["score"]) for row in rows) / max(1, len(rows)),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "mean_response_tokens": sum(int(row["response_tokens"]) for row in rows)
        / max(1, len(rows)),
        "by_subject": {
            key: sum(values) / len(values) for key, values in sorted(by_subject.items())
        },
        "by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "predictions_path": str(output_path),
    }
    write_json(summary_path, summary)
    manifest = build_manifest(
        artifact_type="evaluation_run",
        stage="evaluate",
        config=config,
        files=[output_path, summary_path, metrics_path],
        record_count=len(rows),
        success_count=len(rows),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "suite": suite,
            "model_name": backend.model_name,
            "model_revision": backend.model_revision,
            "tokenizer_revision": backend.tokenizer_revision,
            "experiment_seed": seed,
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return summary_path
