from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_manifest_id
from opd.schemas import (
    RecordStatus,
    RolloutRecord,
    TeacherAnnotationRecord,
    TrainingRecord,
    VerificationRecord,
)
from opd.tableio import read_records, write_records
from opd.training.weighting import confidence_weights, verifier_weight


def build_training_view(config: dict[str, Any], *, round_id: int, method: str) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    extension = config["data"].get("format", "jsonl")
    rollout_path = data_dir / "rollouts" / f"round_{round_id}" / f"rollouts.{extension}"
    verification_path = data_dir / "verifications" / f"round_{round_id}" / f"math.{extension}"
    annotation_path = data_dir / "annotations" / f"round_{round_id}" / f"teacher.{extension}"
    output_dir = data_dir / "training_views" / f"round_{round_id}"
    output_path = output_dir / f"{method}.{extension}"
    upstream_ids = [
        verified_manifest_id(data_dir / "rollouts" / f"round_{round_id}" / "manifest.json"),
        verified_manifest_id(data_dir / "verifications" / f"round_{round_id}" / "manifest.json"),
        verified_manifest_id(data_dir / "annotations" / f"round_{round_id}" / "manifest.json"),
    ]

    rollouts = {
        item.rollout_id: item
        for item in (RolloutRecord.model_validate(row) for row in read_records(rollout_path))
    }
    verifications = {
        item.rollout_id: item
        for item in (
            VerificationRecord.model_validate(row) for row in read_records(verification_path)
        )
    }
    annotations = {
        item.rollout_id: item
        for item in (
            TeacherAnnotationRecord.model_validate(row) for row in read_records(annotation_path)
        )
        if item.status == RecordStatus.SUCCESS
    }
    common_ids = sorted(rollouts.keys() & verifications.keys() & annotations.keys())
    weighting = config["weighting"]
    vocab_size = int(config["models"].get("vocab_size", 256))
    max_length = int(config["training"].get("max_length", 2048))
    records: list[TrainingRecord] = []
    length_filtered = 0
    for rollout_id in common_ids:
        rollout = rollouts[rollout_id]
        verification = verifications[rollout_id]
        annotation = annotations[rollout_id]
        if annotation.tokenizer_fingerprint != rollout.tokenizer_fingerprint:
            raise ValueError(
                f"Tokenizer vocabulary mismatch for {rollout_id}: "
                f"{rollout.tokenizer_fingerprint} != {annotation.tokenizer_fingerprint}"
            )
        available_response_tokens = max_length - annotation.response_start
        if available_response_tokens <= 0:
            length_filtered += 1
            continue
        response_token_count = min(len(annotation.response_token_ids), available_response_tokens)
        input_ids = annotation.input_ids[: annotation.response_start + response_token_count]
        response_token_ids = annotation.response_token_ids[:response_token_count]
        topk_token_ids = annotation.topk_token_ids[:response_token_count]
        topk_logprobs = annotation.topk_logprobs[:response_token_count]
        tail_mass = annotation.tail_mass[:response_token_count]
        token_entropy = annotation.token_entropy[:response_token_count]
        if method in {"verifier_opd", "weighted_opd"}:
            sample_weight = verifier_weight(verification.status, weighting["verifier"])
        else:
            sample_weight = 1.0
        if method in {"confidence_opd", "weighted_opd"}:
            token_weights = confidence_weights(
                token_entropy,
                vocab_size=vocab_size,
                minimum=float(weighting["confidence"]["minimum"]),
                maximum=float(weighting["confidence"]["maximum"]),
            )
        else:
            token_weights = [1.0] * len(response_token_ids)
        record = TrainingRecord(
            rollout_id=rollout_id,
            sample_id=rollout.sample_id,
            prompt=rollout.prompt,
            response=rollout.response,
            tokenizer_fingerprint=annotation.tokenizer_fingerprint,
            input_ids=input_ids,
            response_start=annotation.response_start,
            response_token_ids=response_token_ids,
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            tail_mass=tail_mass,
            confidence_weights=token_weights,
            verifier_status=verification.status,
            verifier_weight=sample_weight,
            method=method,
        )
        records.append(record)

    if not records:
        raise ValueError("No successful rollout/verification/annotation records to join")
    write_records(output_path, [record.model_dump(mode="json") for record in records])
    manifest = build_manifest(
        artifact_type="training_view",
        stage="data.build_view",
        config={**config, "selected_method": method},
        files=[output_path],
        record_count=len(records),
        success_count=len(records),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "method": method,
            "round_id": round_id,
            "zero_weight_records": sum(record.verifier_weight == 0 for record in records),
            "length_filtered_records": length_filtered,
            "max_length": max_length,
        },
    )
    save_manifest(output_dir / f"{method}.manifest.json", manifest)
    return output_path
