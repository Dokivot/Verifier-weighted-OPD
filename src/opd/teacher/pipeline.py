from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
from opd.hashing import file_sha256, stable_hash
from opd.monitoring.job import JobTimer
from opd.schemas import RecordStatus, RolloutRecord, TeacherAnnotationRecord
from opd.sharding import chunks, load_complete_shard, shard_path
from opd.tableio import read_records, write_records
from opd.teacher.hf import HFTeacherAnnotator
from opd.teacher.mock import MockTeacherAnnotator


def _annotator(config: dict[str, Any]) -> Any:
    teacher = config["teacher"]
    if teacher["backend"] == "mock":
        return MockTeacherAnnotator(top_k=int(teacher.get("top_k", 4)))
    if teacher["backend"] == "transformers":
        model = config["models"]["teacher"]
        return HFTeacherAnnotator(
            model_name=model["name"],
            model_revision=model["revision"],
            tokenizer_revision=model.get("tokenizer_revision", model["revision"]),
            top_k=int(teacher.get("top_k", 64)),
            dtype=teacher.get("dtype", "bfloat16"),
            load_in_8bit=bool(teacher.get("load_in_8bit", False)),
            max_length=int(teacher.get("max_length", 2048)),
        )
    raise ValueError(f"Unsupported teacher backend: {teacher['backend']}")


def annotate_rollouts(config: dict[str, Any], *, round_id: int) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    extension = config["data"].get("format", "jsonl")
    rollout_path = Path(
        config["teacher"].get("input_path")
        or data_dir / "annotation_selection" / f"round_{round_id}" / f"selected.{extension}"
    )
    output_dir = Path(
        config["teacher"].get("output_dir") or data_dir / "annotations" / f"round_{round_id}"
    )
    output_path = output_dir / f"teacher.{extension}"
    metrics_path = output_dir / "job_metrics.json"
    upstream_id = verified_artifact_manifest_id(rollout_path)
    rollouts = [RolloutRecord.model_validate(row) for row in read_records(rollout_path)]
    annotator = _annotator(config)
    records: list[TeacherAnnotationRecord] = []
    failures = 0
    reused_shards = 0
    generated_shards = 0
    shard_size = int(config["teacher"].get("shard_size", config["rollout"].get("shard_size", 200)))
    annotation_hash = stable_hash(
        {
            "teacher": config["models"]["teacher"],
            "annotation": config["teacher"],
            "rollout_checksum": file_sha256(rollout_path),
        },
        length=16,
    )
    shard_directory = output_dir / "shards" / annotation_hash

    with JobTimer(
        "teacher_annotation",
        metrics_path,
        round_id=round_id,
        artifact_run_id=annotation_hash,
    ) as timer:
        for shard_index, rollout_shard in chunks(rollouts, shard_size):
            part_path = shard_path(shard_directory, shard_index, extension)
            existing = load_complete_shard(
                part_path,
                model=TeacherAnnotationRecord,
                expected_count=len(rollout_shard),
                is_complete=lambda record: record.status == RecordStatus.SUCCESS,
            )
            if existing is not None:
                records.extend(existing)
                reused_shards += 1
                continue

            shard_records: list[TeacherAnnotationRecord] = []
            for rollout in rollout_shard:
                started = perf_counter()
                try:
                    if rollout.status != RecordStatus.SUCCESS:
                        raise ValueError(f"Upstream rollout status is {rollout.status.value}")
                    annotation = annotator.annotate(rollout.prompt, rollout.response)
                    record = TeacherAnnotationRecord(
                        rollout_id=rollout.rollout_id,
                        sample_id=rollout.sample_id,
                        teacher_model=annotator.model_name,
                        teacher_revision=annotator.model_revision,
                        tokenizer_revision=str(annotation["tokenizer_revision"]),
                        tokenizer_fingerprint=str(annotation["tokenizer_fingerprint"]),
                        input_ids=annotation["input_ids"],
                        response_start=annotation["response_start"],
                        response_token_ids=annotation["response_token_ids"],
                        topk_token_ids=annotation["topk_token_ids"],
                        topk_logprobs=annotation["topk_logprobs"],
                        tail_mass=annotation["tail_mass"],
                        token_entropy=annotation["token_entropy"],
                        teacher_tokens=len(annotation["input_ids"]),
                        latency_ms=int((perf_counter() - started) * 1000),
                    )
                except Exception as exc:
                    failures += 1
                    record = TeacherAnnotationRecord(
                        rollout_id=rollout.rollout_id,
                        sample_id=rollout.sample_id,
                        teacher_model=annotator.model_name,
                        teacher_revision=annotator.model_revision,
                        tokenizer_revision=getattr(annotator, "tokenizer_revision", "unknown"),
                        tokenizer_fingerprint=getattr(
                            annotator, "tokenizer_fingerprint", "unknown"
                        ),
                        input_ids=[1],
                        response_start=1,
                        response_token_ids=[],
                        topk_token_ids=[],
                        topk_logprobs=[],
                        tail_mass=[],
                        token_entropy=[],
                        teacher_tokens=0,
                        latency_ms=int((perf_counter() - started) * 1000),
                        status=RecordStatus.FAILED,
                        error=str(exc),
                    )
                shard_records.append(record)
                timer.add(records=1, teacher_tokens=record.teacher_tokens)
            write_records(part_path, [record.model_dump(mode="json") for record in shard_records])
            records.extend(shard_records)
            generated_shards += 1

    write_records(output_path, [record.model_dump(mode="json") for record in records])
    shard_files = sorted(shard_directory.glob(f"*.{extension}"))
    manifest = build_manifest(
        artifact_type="teacher_annotation",
        stage="teacher.annotate",
        config=config,
        files=[output_path, metrics_path, *shard_files],
        record_count=len(records),
        success_count=len(records) - failures,
        failure_count=failures,
        upstream_artifact_ids=[upstream_id],
        metadata={
            "round_id": round_id,
            "top_k": int(config["teacher"].get("top_k", 64)),
            "shard_size": shard_size,
            "shard_count": len(shard_files),
            "reused_shards": reused_shards,
            "generated_shards": generated_shards,
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    if failures:
        raise RuntimeError(
            f"Teacher annotation produced {failures} failed records; "
            "rerun the stage after fixing the recorded errors"
        )
    return output_path
