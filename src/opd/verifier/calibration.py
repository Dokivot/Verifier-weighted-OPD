from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
)
from opd.hashing import stable_hash
from opd.schemas import RecordStatus, RolloutRecord, VerificationRecord, VerificationStatus
from opd.tableio import read_records, write_json, write_records

_STATUS_NAMES = tuple(status.value for status in VerificationStatus)
_HUMAN_LABEL_FIELDS = {
    "human_status",
    "human_error_type",
    "reviewer",
    "notes",
}


def _truncation_info(
    rollout: RolloutRecord,
    *,
    max_new_tokens: int,
    verification_truncated: bool = False,
) -> tuple[bool, str]:
    if verification_truncated:
        return True, "verification_record"
    if rollout.finish_reason == "length":
        return True, "finish_reason"
    if rollout.status == RecordStatus.TRUNCATED:
        return True, "record_status"
    if rollout.finish_reason == "unknown" and rollout.response_tokens >= max_new_tokens:
        return True, "length_inferred"
    return False, "not_truncated"


def _load_inputs(
    verification_path: Path,
    rollout_path: Path,
) -> tuple[list[VerificationRecord], dict[str, RolloutRecord], list[str]]:
    verification_manifest_id = verified_artifact_manifest_id(verification_path)
    rollout_manifest_id = verified_artifact_manifest_id(rollout_path)
    verifications = [
        VerificationRecord.model_validate(row) for row in read_records(verification_path)
    ]
    rollouts = [RolloutRecord.model_validate(row) for row in read_records(rollout_path)]
    rollout_by_id: dict[str, RolloutRecord] = {}
    for rollout in rollouts:
        if rollout.rollout_id in rollout_by_id:
            raise ValueError(f"Duplicate rollout_id in rollout input: {rollout.rollout_id}")
        rollout_by_id[rollout.rollout_id] = rollout
    missing: list[str] = []
    for record in verifications:
        if record.rollout_id not in rollout_by_id:
            missing.append(record.rollout_id)
    if missing:
        raise ValueError(f"Verification has no matching rollout: {missing[0]}")
    return verifications, rollout_by_id, [verification_manifest_id, rollout_manifest_id]


def _review_rows(
    verifications: list[VerificationRecord],
    rollout_by_id: dict[str, RolloutRecord],
    *,
    samples_per_stratum: int,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive")
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (*_STATUS_NAMES, "truncated")
    }
    for verification in verifications:
        rollout = rollout_by_id[verification.rollout_id]
        truncated, truncation_source = _truncation_info(
            rollout,
            max_new_tokens=max_new_tokens,
            verification_truncated=verification.truncated,
        )
        stratum = "truncated" if truncated else verification.status.value
        candidates[stratum].append(
            {
                "rollout_id": verification.rollout_id,
                "sample_id": verification.sample_id,
                "stratum": stratum,
                "verifier_status": verification.status.value,
                "verifier_error_type": verification.error_type,
                "verifier_reason": verification.details.get("reason", "missing"),
                "extraction_confidence": verification.details.get("extraction_confidence", "none"),
                "truncated": truncated,
                "truncation_source": truncation_source,
                "finish_reason": rollout.finish_reason,
                "response_tokens": rollout.response_tokens,
                "prompt": rollout.prompt,
                "reference_answer": verification.reference_answer,
                "response": rollout.response,
                "human_status": None,
                "human_error_type": None,
                "reviewer": None,
                "notes": None,
            }
        )

    selected: list[dict[str, Any]] = []
    for stratum in (*_STATUS_NAMES, "truncated"):
        ordered = sorted(
            candidates[stratum],
            key=lambda row: stable_hash(
                {"seed": seed, "stratum": stratum, "rollout_id": row["rollout_id"]}
            ),
        )
        selected.extend(ordered[:samples_per_stratum])
    return selected


def _labelled_rows(
    queue_rows: list[dict[str, Any]],
    labels_path: Path,
) -> list[dict[str, Any]]:
    labels = read_records(labels_path)
    queue_by_id = {str(row["rollout_id"]): row for row in queue_rows}
    labels_by_id: dict[str, dict[str, Any]] = {}
    for label in labels:
        rollout_id = str(label.get("rollout_id", ""))
        if rollout_id not in queue_by_id:
            raise ValueError(f"Manual label is not present in review queue: {rollout_id}")
        if rollout_id in labels_by_id:
            raise ValueError(f"Duplicate manual label: {rollout_id}")
        status = str(label.get("human_status", ""))
        if status not in (*_STATUS_NAMES,):
            raise ValueError(f"human_status for {rollout_id} must be one of {list(_STATUS_NAMES)}")
        labels_by_id[rollout_id] = label
    missing = sorted(set(queue_by_id) - set(labels_by_id))
    if missing:
        raise ValueError(f"Manual labels are missing review examples: {missing[0]}")

    result: list[dict[str, Any]] = []
    for row in queue_rows:
        label = labels_by_id[str(row["rollout_id"])]
        merged = dict(row)
        for field in _HUMAN_LABEL_FIELDS:
            if field in label:
                merged[field] = label[field]
        merged["human_status"] = str(merged["human_status"])
        result.append(merged)
    return result


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {verifier: {human: 0 for human in _STATUS_NAMES} for verifier in _STATUS_NAMES}
    for row in rows:
        verifier = str(row["verifier_status"])
        human = str(row["human_status"])
        matrix.setdefault(verifier, {label: 0 for label in _STATUS_NAMES})
        matrix[verifier].setdefault(human, 0)
        matrix[verifier][human] += 1
    return matrix


def _class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for label in _STATUS_NAMES:
        true_positive = sum(
            row["verifier_status"] == label and row["human_status"] == label for row in rows
        )
        false_positive = sum(
            row["verifier_status"] == label and row["human_status"] != label for row in rows
        )
        false_negative = sum(
            row["verifier_status"] != label and row["human_status"] == label for row in rows
        )
        support = sum(row["human_status"] == label for row in rows)
        metrics[label] = {
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            "recall": true_positive / support if support else None,
        }
    return metrics


def _calibration_buckets(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(field, "unknown")), []).append(row)
    return {
        key: {
            "samples": len(bucket),
            "human_pass_rate": sum(row["human_status"] == "pass" for row in bucket) / len(bucket),
            "verifier_pass_rate": sum(row["verifier_status"] == "pass" for row in bucket)
            / len(bucket),
            "agreement_rate": sum(row["human_status"] == row["verifier_status"] for row in bucket)
            / len(bucket),
        }
        for key, bucket in sorted(buckets.items())
    }


def _build_report(rows: list[dict[str, Any]], *, labels_path: Path) -> dict[str, Any]:
    status_counts = Counter(str(row["verifier_status"]) for row in rows)
    human_counts = Counter(str(row["human_status"]) for row in rows)
    reasons = Counter(str(row.get("verifier_reason", "missing")) for row in rows)
    disagreements = [
        {
            "rollout_id": row["rollout_id"],
            "sample_id": row["sample_id"],
            "stratum": row["stratum"],
            "verifier_status": row["verifier_status"],
            "human_status": row["human_status"],
            "verifier_error_type": row["verifier_error_type"],
            "human_error_type": row.get("human_error_type"),
            "truncated": row["truncated"],
            "response": row["response"],
            "reference_answer": row["reference_answer"],
            "notes": row.get("notes"),
        }
        for row in rows
        if row["verifier_status"] != row["human_status"]
    ]
    return {
        "report_type": "verifier_calibration",
        "status": "complete",
        "internal_verifier": {
            "verifier_status_counts": dict(sorted(status_counts.items())),
            "human_status_counts": dict(sorted(human_counts.items())),
            "confusion_matrix": _confusion_matrix(rows),
            "class_metrics": _class_metrics(rows),
            "unknown_reason_counts": dict(sorted(reasons.items())),
            "overall_agreement_rate": sum(
                row["human_status"] == row["verifier_status"] for row in rows
            )
            / max(1, len(rows)),
            "calibration_by_extraction_confidence": _calibration_buckets(
                rows, "extraction_confidence"
            ),
            "calibration_by_stratum": _calibration_buckets(rows, "stratum"),
            "reviewed_samples": len(rows),
            "manual_labels_path": str(labels_path),
        },
        "official_benchmark": {
            "status": "reported_separately",
            "metric_source": "LightEval official task or public benchmark grader",
            "internal_verifier_not_used_as_primary_benchmark": True,
        },
        "disagreements": {
            "count": len(disagreements),
            "path": "disagreements.jsonl",
        },
        "disagreement_rows": disagreements,
        "unknown_is_preserved": True,
    }


def calibrate_verifier(
    config: dict[str, Any],
    *,
    verification_path: str | Path,
    rollout_path: str | Path,
    output_dir: str | Path,
    labels_path: str | Path | None = None,
    samples_per_stratum: int = 20,
    seed: int | None = None,
) -> Path:
    verification = Path(verification_path)
    rollouts = Path(rollout_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    actual_seed = int(config["project"]["seed"] if seed is None else seed)
    max_new_tokens = int(
        config.get("rollout", {}).get("generation", {}).get("max_new_tokens", 4096)
    )
    verifications, rollout_by_id, upstream_ids = _load_inputs(verification, rollouts)
    queue_rows = _review_rows(
        verifications,
        rollout_by_id,
        samples_per_stratum=samples_per_stratum,
        seed=actual_seed,
        max_new_tokens=max_new_tokens,
    )
    queue_path = destination / "review_queue.jsonl"
    write_records(queue_path, queue_rows)
    files = [queue_path]
    result: dict[str, Any]
    if labels_path is None:
        result = {
            "report_type": "verifier_calibration",
            "status": "awaiting_manual_labels",
            "internal_verifier": {
                "verifier_status_counts": dict(
                    sorted(Counter(str(row["verifier_status"]) for row in queue_rows).items())
                ),
                "stratum_counts": dict(
                    sorted(Counter(str(row["stratum"]) for row in queue_rows).items())
                ),
                "reviewed_samples": 0,
                "samples_per_stratum": samples_per_stratum,
                "review_queue_path": str(queue_path),
            },
            "official_benchmark": {
                "status": "reported_separately",
                "metric_source": "LightEval official task or public benchmark grader",
                "internal_verifier_not_used_as_primary_benchmark": True,
            },
            "unknown_is_preserved": True,
        }
    else:
        source_labels = Path(labels_path)
        labelled_rows = _labelled_rows(queue_rows, source_labels)
        copied_labels = destination / "human_labels.jsonl"
        write_records(copied_labels, labelled_rows)
        result = _build_report(labelled_rows, labels_path=copied_labels)
        disagreements_path = destination / "disagreements.jsonl"
        write_records(disagreements_path, result.pop("disagreement_rows"))
        files.extend([copied_labels, disagreements_path])

    report_path = destination / "calibration_report.json"
    write_json(report_path, result)
    files.append(report_path)
    manifest = build_manifest(
        artifact_type="verifier_calibration",
        stage="verify.calibrate",
        config={
            **config,
            "verifier_calibration": {
                "verification_path": str(verification),
                "rollout_path": str(rollouts),
                "samples_per_stratum": samples_per_stratum,
                "seed": actual_seed,
            },
        },
        files=files,
        record_count=len(queue_rows),
        success_count=len(queue_rows),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "status": result["status"],
            "strata": [*_STATUS_NAMES, "truncated"],
            "samples_per_stratum": samples_per_stratum,
            "manual_labels_provided": labels_path is not None,
            "max_new_tokens": max_new_tokens,
        },
    )
    save_manifest(destination / "manifest.json", manifest)
    return report_path
