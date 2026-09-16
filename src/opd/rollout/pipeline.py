from __future__ import annotations

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
from opd.schemas import PromptRecord, RecordStatus, RolloutRecord
from opd.sharding import chunks, load_complete_shard, shard_path
from opd.tableio import read_records, write_records


def _backend(config: dict[str, Any]) -> Any:
    rollout = config["rollout"]
    if rollout["backend"] == "mock":
        return MockRolloutBackend()
    if rollout["backend"] == "transformers":
        model = config["models"]["student"]
        return HFRolloutBackend(
            model_name=model["name"],
            model_revision=model["revision"],
            tokenizer_revision=model.get("tokenizer_revision", model["revision"]),
            generation_config=rollout["generation"],
            dtype=rollout.get("dtype", "bfloat16"),
        )
    if rollout["backend"] == "vllm":
        model = config["models"]["student"]
        return VLLMRolloutBackend(
            model_name=model["name"],
            model_revision=model["revision"],
            tokenizer_revision=model.get("tokenizer_revision", model["revision"]),
            generation_config=rollout["generation"],
            tensor_parallel_size=int(rollout.get("tensor_parallel_size", 1)),
            dtype=str(rollout.get("dtype", "bfloat16")),
            max_model_length=int(rollout.get("max_model_length", 4096)),
            gpu_memory_utilization=float(rollout.get("gpu_memory_utilization", 0.9)),
        )
    raise ValueError(f"Unsupported rollout backend: {rollout['backend']}")


def generate_rollouts(config: dict[str, Any], *, round_id: int) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    extension = config["data"].get("format", "jsonl")
    split = config["rollout"].get("split", "train")
    prompt_path = Path(
        config["rollout"].get("input_path") or data_dir / "curated" / f"{split}.{extension}"
    )
    upstream_id = (
        verified_manifest_id(data_dir / "manifests" / "data_prepare.json")
        if prompt_path.parent.name == "curated"
        else verified_artifact_manifest_id(prompt_path)
    )
    output_dir = data_dir / "rollouts" / f"round_{round_id}"
    output_path = output_dir / f"rollouts.{extension}"
    metrics_path = output_dir / "job_metrics.json"
    prompts = [PromptRecord.model_validate(row) for row in read_records(prompt_path)]
    backend = _backend(config)
    generation_config = config["rollout"]["generation"]
    sampling_hash = stable_hash(generation_config, length=16)
    shard_cache_hash = stable_hash(
        {
            "student_model": backend.model_name,
            "student_revision": backend.model_revision,
            "tokenizer_revision": backend.tokenizer_revision,
            "tokenizer_fingerprint": backend.tokenizer_fingerprint,
            "sampling": generation_config,
            "upstream_artifact_id": upstream_id,
        },
        length=16,
    )
    batch_size = int(config["rollout"].get("batch_size", 16))
    num_samples = int(generation_config.get("num_samples", 1))
    shard_size = int(config["rollout"].get("shard_size", 200))
    seed = int(config["project"]["seed"])
    records: list[RolloutRecord] = []
    reused_shards = 0
    generated_shards = 0
    failures = 0
    shard_directory = output_dir / "shards" / shard_cache_hash

    with JobTimer(
        "rollout",
        metrics_path,
        round_id=round_id,
        artifact_run_id=shard_cache_hash,
    ) as timer:
        for shard_index, prompt_shard in chunks(prompts, shard_size):
            part_path = shard_path(shard_directory, shard_index, extension)
            existing = load_complete_shard(
                part_path,
                model=RolloutRecord,
                expected_count=len(prompt_shard) * num_samples,
                is_complete=lambda record: record.status == RecordStatus.SUCCESS,
            )
            if existing is not None:
                records.extend(existing)
                reused_shards += 1
                continue

            shard_records: list[RolloutRecord] = []
            shard_seed = seed + shard_index
            for start in range(0, len(prompt_shard), batch_size):
                batch = prompt_shard[start : start + batch_size]
                repeated = [record for record in batch for _ in range(num_samples)]
                generated_prompts = [record.problem for record in repeated]
                started = perf_counter()
                try:
                    generations = backend.generate(generated_prompts, seed=shard_seed)
                    if len(generations) != len(repeated):
                        raise ValueError("Rollout backend returned an unexpected number of records")
                    generation_error: Exception | None = None
                except Exception as exc:
                    generations = []
                    generation_error = exc
                batch_latency = int((perf_counter() - started) * 1000)
                for offset, prompt in enumerate(repeated):
                    candidate_index = offset % num_samples
                    rollout_id = f"r{round_id}_{prompt.sample_id}_{candidate_index}"
                    if generation_error is None:
                        generation = generations[offset]
                        response = generation.text
                        prompt_tokens = generation.prompt_tokens
                        response_tokens = generation.response_tokens
                        status = RecordStatus.SUCCESS
                        error = None
                    else:
                        response = ""
                        prompt_tokens = 0
                        response_tokens = 0
                        status = RecordStatus.FAILED
                        error = str(generation_error)
                        failures += 1
                    record = RolloutRecord(
                        rollout_id=rollout_id,
                        sample_id=prompt.sample_id,
                        round_id=round_id,
                        candidate_index=candidate_index,
                        prompt=prompt.problem,
                        response=response,
                        student_model=backend.model_name,
                        student_revision=backend.model_revision,
                        tokenizer_revision=backend.tokenizer_revision,
                        tokenizer_fingerprint=backend.tokenizer_fingerprint,
                        sampling_config_hash=sampling_hash,
                        seed=shard_seed,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        latency_ms=max(1, batch_latency // max(1, len(repeated))),
                        status=status,
                        error=error,
                    )
                    shard_records.append(record)
                    timer.add(
                        records=1,
                        prompt_tokens=record.prompt_tokens,
                        response_tokens=record.response_tokens,
                    )
            write_records(part_path, [record.model_dump(mode="json") for record in shard_records])
            records.extend(shard_records)
            generated_shards += 1

    write_records(output_path, [record.model_dump(mode="json") for record in records])
    shard_files = sorted(shard_directory.glob(f"*.{extension}"))
    manifest = build_manifest(
        artifact_type="rollout",
        stage="rollout.generate",
        config=config,
        files=[output_path, metrics_path, *shard_files],
        record_count=len(records),
        success_count=len(records) - failures,
        failure_count=failures,
        upstream_artifact_ids=[upstream_id],
        metadata={
            "round_id": round_id,
            "student_revision": backend.model_revision,
            "shard_size": shard_size,
            "shard_count": len(shard_files),
            "reused_shards": reused_shards,
            "generated_shards": generated_shards,
            "experiment_seed": seed,
            "shard_cache_hash": shard_cache_hash,
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return output_path
