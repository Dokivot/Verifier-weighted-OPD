from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
    verified_manifest_id,
)
from opd.schemas import PromptRecord, RolloutRecord
from opd.tableio import read_records, write_records
from opd.verifier.math import MathVerifier


def verify_rollouts(config: dict[str, Any], *, round_id: int) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    extension = config["data"].get("format", "jsonl")
    prompt_path = Path(
        config["verification"].get("prompt_path") or data_dir / "curated" / f"train.{extension}"
    )
    rollout_path = Path(
        config["verification"].get("rollout_path")
        or data_dir / "rollouts" / f"round_{round_id}" / f"rollouts.{extension}"
    )
    output_dir = Path(
        config["verification"].get("output_dir") or data_dir / "verifications" / f"round_{round_id}"
    )
    output_path = output_dir / f"math.{extension}"
    prompt_manifest_id = (
        verified_manifest_id(data_dir / "manifests" / "data_prepare.json")
        if prompt_path.parent.name == "curated"
        else verified_artifact_manifest_id(prompt_path)
    )
    upstream_ids = [
        prompt_manifest_id,
        verified_manifest_id(data_dir / "rollouts" / f"round_{round_id}" / "manifest.json"),
    ]

    prompts = {
        record.sample_id: record
        for record in (PromptRecord.model_validate(row) for row in read_records(prompt_path))
    }
    rollouts = [RolloutRecord.model_validate(row) for row in read_records(rollout_path)]
    verifier = MathVerifier()
    results = []
    for rollout in rollouts:
        prompt = prompts[rollout.sample_id]
        result = verifier.verify(
            rollout_id=rollout.rollout_id,
            sample_id=rollout.sample_id,
            response=rollout.response,
            reference_answer=prompt.reference_answer,
        )
        results.append(result)
    write_records(output_path, [result.model_dump(mode="json") for result in results])
    counts = Counter(result.status.value for result in results)
    extraction_modes = Counter(
        str(result.details.get("extraction_mode", "unknown")) for result in results
    )
    manifest = build_manifest(
        artifact_type="verification",
        stage="verify.math",
        config=config,
        files=[output_path],
        record_count=len(results),
        success_count=len(results),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "round_id": round_id,
            "verifier_name": verifier.name,
            "verifier_version": verifier.version,
            "status_counts": dict(counts),
            "extraction_mode_counts": dict(extraction_modes),
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return output_path
