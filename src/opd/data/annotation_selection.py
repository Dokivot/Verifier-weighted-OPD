from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
    verified_manifest_id,
)
from opd.hashing import stable_hash
from opd.schemas import (
    AnnotationSelectionRecord,
    PromptRecord,
    RecordStatus,
    RolloutRecord,
    VerificationRecord,
    VerificationStatus,
)
from opd.tableio import read_records, write_json, write_records

_VALID_STRATEGIES = {"dense", "random_budget", "verifier_filtered", "vfs"}
_VALID_GROUPS = {"boundary", "uncertain", "solved", "failed"}


def _group_type(verifications: list[VerificationRecord]) -> str:
    statuses = [record.status for record in verifications]
    has_pass = VerificationStatus.PASS in statuses
    if has_pass and any(status != VerificationStatus.PASS for status in statuses):
        return "boundary"
    if statuses and all(status == VerificationStatus.PASS for status in statuses):
        return "solved"
    answers = {
        record.extracted_answer.strip().lower()
        for record in verifications
        if record.extracted_answer and record.extracted_answer.strip()
    }
    if VerificationStatus.UNKNOWN in statuses or len(answers) > 1:
        return "uncertain"
    return "failed"


def _round_robin(
    records: list[AnnotationSelectionRecord],
    *,
    stratify_by: list[str],
) -> list[AnnotationSelectionRecord]:
    if not records:
        return []
    if not stratify_by:
        return records
    buckets: dict[tuple[str, ...], deque[AnnotationSelectionRecord]] = defaultdict(deque)
    for record in records:
        key = tuple(str(getattr(record, field)) for field in stratify_by)
        buckets[key].append(record)
    ordered: list[AnnotationSelectionRecord] = []
    keys = sorted(buckets)
    while keys:
        next_keys: list[tuple[str, ...]] = []
        for key in keys:
            ordered.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _candidate_order(
    records: list[AnnotationSelectionRecord],
    *,
    strategy: str,
    status_priority: list[str],
    group_priority: list[str],
    stratify_by: list[str],
) -> list[AnnotationSelectionRecord]:
    status_rank = {name: index for index, name in enumerate(status_priority)}
    group_rank = {name: index for index, name in enumerate(group_priority)}
    if strategy == "random_budget":
        return sorted(records, key=lambda record: record.stable_tiebreak)
    if strategy == "dense":
        return sorted(records, key=lambda record: record.rollout_id)

    eligible = records
    if strategy == "verifier_filtered":
        eligible = [
            record
            for record in records
            if record.verifier_status in {VerificationStatus.PASS, VerificationStatus.UNKNOWN}
        ]

    by_sample: dict[str, list[AnnotationSelectionRecord]] = defaultdict(list)
    for record in eligible:
        by_sample[record.sample_id].append(record)
    candidate_waves: dict[int, list[AnnotationSelectionRecord]] = defaultdict(list)
    for sample_records in by_sample.values():
        sample_records.sort(
            key=lambda record: (
                record.truncated,
                status_rank.get(record.verifier_status.value, len(status_rank)),
                record.estimated_teacher_tokens,
                record.stable_tiebreak,
            )
        )
        for candidate_rank, record in enumerate(sample_records):
            candidate_waves[candidate_rank].append(record)

    ordered: list[AnnotationSelectionRecord] = []
    for candidate_rank in sorted(candidate_waves):
        wave = candidate_waves[candidate_rank]
        if strategy == "vfs":
            wave.sort(
                key=lambda record: (
                    group_rank.get(record.group_type, len(group_rank)),
                    record.truncated,
                    status_rank.get(record.verifier_status.value, len(status_rank)),
                    record.estimated_teacher_tokens,
                    record.stable_tiebreak,
                )
            )
            for group in group_priority:
                group_records = [record for record in wave if record.group_type == group]
                ordered.extend(_round_robin(group_records, stratify_by=stratify_by))
            unknown_groups = [record for record in wave if record.group_type not in group_rank]
            ordered.extend(_round_robin(unknown_groups, stratify_by=stratify_by))
        else:
            wave.sort(
                key=lambda record: (
                    status_rank.get(record.verifier_status.value, len(status_rank)),
                    record.truncated,
                    record.estimated_teacher_tokens,
                    record.stable_tiebreak,
                )
            )
            ordered.extend(_round_robin(wave, stratify_by=stratify_by))
    return ordered


def _validate_choice_list(values: Iterable[str], allowed: set[str], name: str) -> list[str]:
    result = [str(value) for value in values]
    if len(result) != len(set(result)) or set(result) != allowed:
        raise ValueError(f"{name} must contain each of {sorted(allowed)} exactly once")
    return result


def select_annotation_rollouts(config: dict[str, Any], *, round_id: int) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    extension = config["data"].get("format", "jsonl")
    selection = config.get("annotation_selection", {})
    strategy = str(selection.get("strategy", "vfs"))
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"annotation_selection.strategy must be one of {sorted(_VALID_STRATEGIES)}"
        )
    enabled = bool(selection.get("enabled", True))
    if not enabled:
        strategy = "dense"

    rollout_path = Path(
        selection.get("rollout_path")
        or data_dir / "rollouts" / f"round_{round_id}" / f"rollouts.{extension}"
    )
    verification_path = Path(
        selection.get("verification_path")
        or config.get("verification", {}).get("internal_output_dir")
        or config.get("verification", {}).get("output_dir")
        or data_dir / "internal_verifier" / f"round_{round_id}" / f"math.{extension}"
    )
    prompt_path = Path(
        selection.get("prompt_path")
        or config.get("verification", {}).get("prompt_path")
        or data_dir / "curated" / f"train.{extension}"
    )
    output_dir = Path(
        selection.get("output_dir") or data_dir / "annotation_selection" / f"round_{round_id}"
    )
    selected_path = output_dir / f"selected.{extension}"
    decisions_path = output_dir / f"decisions.{extension}"
    report_path = output_dir / "selection_report.json"

    upstream_ids = [
        verified_artifact_manifest_id(rollout_path),
        verified_artifact_manifest_id(verification_path),
        (
            verified_manifest_id(data_dir / "manifests" / "data_prepare.json")
            if prompt_path.parent.name == "curated"
            else verified_artifact_manifest_id(prompt_path)
        ),
    ]
    rollouts = [RolloutRecord.model_validate(row) for row in read_records(rollout_path)]
    expected_rollouts = 1
    if enabled:
        expected_rollouts = int(
            selection.get(
                "rollouts_per_prompt",
                config["rollout"]["generation"].get("num_samples", 1),
            )
        )
        if expected_rollouts < 2:
            raise ValueError(
                "Enabled annotation selection requires at least two rollouts per prompt"
            )
        candidate_indices: dict[str, set[int]] = defaultdict(set)
        for rollout in rollouts:
            candidate_indices[rollout.sample_id].add(rollout.candidate_index)
        invalid_samples = [
            sample_id
            for sample_id, indices in candidate_indices.items()
            if len(indices) != expected_rollouts
        ]
        if invalid_samples:
            raise ValueError(
                f"Expected {expected_rollouts} rollout candidates for sample {invalid_samples[0]}"
            )
    verification_by_id = {
        record.rollout_id: record
        for record in (
            VerificationRecord.model_validate(row) for row in read_records(verification_path)
        )
    }
    prompts = {
        record.sample_id: record
        for record in (PromptRecord.model_validate(row) for row in read_records(prompt_path))
    }
    missing_verifications = [
        rollout.rollout_id for rollout in rollouts if rollout.rollout_id not in verification_by_id
    ]
    missing_prompts = [
        rollout.sample_id for rollout in rollouts if rollout.sample_id not in prompts
    ]
    if missing_verifications:
        raise ValueError(f"Missing verification for rollout {missing_verifications[0]}")
    if missing_prompts:
        raise ValueError(f"Missing prompt metadata for sample {missing_prompts[0]}")

    status_priority = _validate_choice_list(
        selection.get("status_priority", ["pass", "unknown", "fail"]),
        {status.value for status in VerificationStatus},
        "annotation_selection.status_priority",
    )
    group_priority = _validate_choice_list(
        selection.get("group_priority", ["boundary", "uncertain", "solved", "failed"]),
        _VALID_GROUPS,
        "annotation_selection.group_priority",
    )
    stratify_by = [str(value) for value in selection.get("stratify_by", [])]
    unsupported_strata = set(stratify_by) - {"subject", "difficulty"}
    if unsupported_strata:
        raise ValueError("annotation_selection.stratify_by only supports subject and difficulty")

    verifications_by_sample: dict[str, list[VerificationRecord]] = defaultdict(list)
    for rollout in rollouts:
        verifications_by_sample[rollout.sample_id].append(verification_by_id[rollout.rollout_id])
    group_by_sample = {
        sample_id: _group_type(records) for sample_id, records in verifications_by_sample.items()
    }
    seed = int(config["project"]["seed"])
    max_new_tokens = int(config["rollout"]["generation"].get("max_new_tokens", 1024))
    teacher_max_length = int(config["teacher"].get("max_length", 2048))
    decisions: list[AnnotationSelectionRecord] = []
    rollout_by_id = {rollout.rollout_id: rollout for rollout in rollouts}
    for rollout in rollouts:
        verification = verification_by_id[rollout.rollout_id]
        prompt = prompts[rollout.sample_id]
        estimated_tokens = min(
            rollout.prompt_tokens + rollout.response_tokens,
            teacher_max_length,
        )
        decisions.append(
            AnnotationSelectionRecord(
                rollout_id=rollout.rollout_id,
                sample_id=rollout.sample_id,
                candidate_index=rollout.candidate_index,
                group_type=group_by_sample[rollout.sample_id],
                verifier_status=verification.status,
                extracted_answer=verification.extracted_answer,
                subject=prompt.subject,
                difficulty=prompt.difficulty,
                estimated_teacher_tokens=estimated_tokens,
                truncated=rollout.response_tokens >= max_new_tokens,
                selected=False,
                selection_reason=(
                    "upstream_rollout_failed"
                    if rollout.status != RecordStatus.SUCCESS
                    else "not_considered"
                ),
                strategy=strategy,
                stable_tiebreak=stable_hash(
                    {"seed": seed, "strategy": strategy, "rollout_id": rollout.rollout_id}
                ),
            )
        )

    eligible = [
        record
        for record in decisions
        if rollout_by_id[record.rollout_id].status == RecordStatus.SUCCESS
    ]
    dense_tokens = sum(record.estimated_teacher_tokens for record in eligible)
    budget_tokens_value = selection.get("budget_tokens")
    budget_ratio = float(selection.get("budget_ratio", 1.0))
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError("annotation_selection.budget_ratio must be in (0, 1]")
    if budget_tokens_value is not None:
        budget_tokens = int(budget_tokens_value)
        if budget_tokens <= 0:
            raise ValueError("annotation_selection.budget_tokens must be positive")
    else:
        budget_tokens = int(dense_tokens * budget_ratio)
    if strategy == "dense":
        budget_tokens = dense_tokens

    ordered = _candidate_order(
        eligible,
        strategy=strategy,
        status_priority=status_priority,
        group_priority=group_priority,
        stratify_by=stratify_by,
    )
    selected_ids: set[str] = set()
    used_tokens = 0
    selection_rank = 0
    for record in ordered:
        before = budget_tokens - used_tokens
        if record.estimated_teacher_tokens > before:
            record.selection_reason = "teacher_token_budget_exceeded"
            record.budget_before = before
            record.budget_after = before
            continue
        selection_rank += 1
        used_tokens += record.estimated_teacher_tokens
        record.selected = True
        record.selection_rank = selection_rank
        record.selection_reason = f"selected_by_{strategy}"
        record.budget_before = before
        record.budget_after = budget_tokens - used_tokens
        selected_ids.add(record.rollout_id)

    considered_ids = {record.rollout_id for record in ordered}
    for record in decisions:
        if (
            record.rollout_id in considered_ids
            or record.selection_reason == "upstream_rollout_failed"
        ):
            continue
        record.selection_reason = (
            "verifier_status_filtered" if strategy == "verifier_filtered" else "not_selected"
        )

    selected_rollouts = [rollout for rollout in rollouts if rollout.rollout_id in selected_ids]
    if not selected_rollouts:
        raise ValueError("Annotation selection produced no records")
    rank_by_id = {
        record.rollout_id: record.selection_rank or 0 for record in decisions if record.selected
    }
    selected_rollouts.sort(key=lambda rollout: rank_by_id[rollout.rollout_id])
    write_records(
        selected_path,
        [record.model_dump(mode="json") for record in selected_rollouts],
    )
    write_records(decisions_path, [record.model_dump(mode="json") for record in decisions])
    report = {
        "round_id": round_id,
        "enabled": enabled,
        "strategy": strategy,
        "seed": seed,
        "candidate_records": len(decisions),
        "eligible_records": len(eligible),
        "selected_records": len(selected_rollouts),
        "rollouts_per_prompt": expected_rollouts,
        "dense_estimated_teacher_tokens": dense_tokens,
        "teacher_token_budget": budget_tokens,
        "selected_estimated_teacher_tokens": used_tokens,
        "budget_utilization": used_tokens / budget_tokens if budget_tokens else 0.0,
        "selected_token_ratio": used_tokens / dense_tokens if dense_tokens else 0.0,
        "group_counts": dict(Counter(record.group_type for record in decisions)),
        "selected_group_counts": dict(
            Counter(record.group_type for record in decisions if record.selected)
        ),
        "status_counts": dict(Counter(record.verifier_status.value for record in decisions)),
        "selected_status_counts": dict(
            Counter(record.verifier_status.value for record in decisions if record.selected)
        ),
        "truncated_records": sum(record.truncated for record in decisions),
        "selected_truncated_records": sum(
            record.truncated for record in decisions if record.selected
        ),
    }
    write_json(report_path, report)
    manifest = build_manifest(
        artifact_type="annotation_selection",
        stage="data.select_annotations",
        config=config,
        files=[selected_path, decisions_path, report_path],
        record_count=len(decisions),
        success_count=len(selected_rollouts),
        upstream_artifact_ids=upstream_ids,
        metadata=report,
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return selected_path
