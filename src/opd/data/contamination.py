from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import (
    build_manifest,
    save_manifest,
    verified_artifact_manifest_id,
    verified_manifest_id,
)
from opd.data.normalize import jaccard_similarity, normalize_text, token_shingles
from opd.hashing import stable_hash
from opd.tableio import read_records, write_json, write_records


def audit_contamination(config: dict[str, Any]) -> dict[str, Any]:
    train_path = Path(config["contamination"]["train_path"])
    eval_paths = [Path(path) for path in config["contamination"].get("eval_paths", [])]
    threshold = float(config["contamination"].get("jaccard_threshold", 0.8))
    shingle_size = int(config["contamination"].get("shingle_size", 5))
    output_dir = Path(config["paths"]["data_dir"]) / "contamination"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_eval_paths = [path for path in eval_paths if not path.exists()]
    if missing_eval_paths:
        missing = ", ".join(str(path) for path in missing_eval_paths)
        raise FileNotFoundError(
            f"Contamination benchmark data is missing: {missing}. "
            "Fetch pinned records with `opd data fetch-eval`, or use "
            "`opd data import-eval` for an offline export."
        )

    train_records = read_records(train_path)
    eval_records = [record for path in eval_paths for record in read_records(path)]
    eval_exact = {
        normalize_text(
            str(row.get("problem") or row.get("prompt") or row.get("question") or "")
        ): row
        for row in eval_records
    }
    eval_shingles = [
        (
            row,
            token_shingles(
                str(row.get("problem") or row.get("prompt") or row.get("question") or ""),
                shingle_size,
            ),
        )
        for row in eval_records
    ]

    matches: list[dict[str, Any]] = []
    for train in train_records:
        text = str(train.get("problem") or train.get("prompt") or "")
        normalized = normalize_text(text)
        if normalized in eval_exact and normalized:
            matches.append(
                {
                    "sample_id": train.get("sample_id"),
                    "match_type": "exact",
                    "score": 1.0,
                    "eval_id": eval_exact[normalized].get("sample_id"),
                }
            )
            continue
        if not config["contamination"].get("enable_near_duplicate", True):
            continue
        shingles = token_shingles(text, shingle_size)
        best_score = 0.0
        best_eval: dict[str, Any] | None = None
        for eval_row, candidate in eval_shingles:
            score = jaccard_similarity(shingles, candidate)
            if score > best_score:
                best_score = score
                best_eval = eval_row
        if best_eval is not None and best_score >= threshold:
            matches.append(
                {
                    "sample_id": train.get("sample_id"),
                    "match_type": "near",
                    "score": best_score,
                    "eval_id": best_eval.get("sample_id"),
                }
            )

    quarantine_ids = {str(match["sample_id"]) for match in matches}
    clean = [row for row in train_records if str(row.get("sample_id")) not in quarantine_ids]
    quarantine = [row for row in train_records if str(row.get("sample_id")) in quarantine_ids]
    extension = train_path.suffix
    clean_path = output_dir / f"train_clean{extension}"
    quarantine_path = output_dir / f"quarantine{extension}"
    if not clean:
        raise ValueError("Contamination audit removed every training record")
    write_records(clean_path, clean)
    if quarantine:
        write_records(quarantine_path, quarantine)
    report = {
        "audit_id": stable_hash({"train": str(train_path), "eval": list(map(str, eval_paths))}),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "match_count": len(matches),
        "clean_count": len(clean),
        "threshold": threshold,
        "matches": matches,
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    source_manifest = Path(config["paths"]["data_dir"]) / "manifests/data_prepare.json"
    upstream_id = verified_manifest_id(source_manifest)
    upstream_ids = [upstream_id]
    for eval_path in eval_paths:
        if eval_path.parent.name == "curated":
            continue
        eval_manifest_id = verified_artifact_manifest_id(eval_path)
        if eval_manifest_id not in upstream_ids:
            upstream_ids.append(eval_manifest_id)
    artifact_files = [clean_path, report_path]
    if quarantine:
        artifact_files.append(quarantine_path)
    manifest = build_manifest(
        artifact_type="decontaminated_prompt_dataset",
        stage="data.audit_contamination",
        config=config,
        files=artifact_files,
        record_count=len(train_records),
        success_count=len(clean),
        failure_count=len(quarantine),
        upstream_artifact_ids=upstream_ids,
        metadata={
            "match_count": len(matches),
            "clean_count": len(clean),
            "threshold": threshold,
        },
    )
    save_manifest(output_dir / "manifest.json", manifest)
    return report
