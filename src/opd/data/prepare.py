from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest
from opd.exceptions import DependencyError
from opd.hashing import stable_hash
from opd.schemas import PromptRecord
from opd.tableio import read_records, write_json, write_records


def _iter_source(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    source = config["data"]["source"]
    if source == "local":
        yield from read_records(config["data"]["input_path"])
        return
    if source != "huggingface":
        raise ValueError(f"Unsupported data source: {source}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DependencyError("Hugging Face input requires `pip install -e .[data]`") from exc
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"].get("dataset_subset"),
        split=config["data"].get("dataset_split", "train"),
        revision=config["data"].get("dataset_revision"),
    )
    yield from dataset


def _first_nonempty(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def convert_prompt_row(row: dict[str, Any], config: dict[str, Any]) -> PromptRecord | None:
    problem = _first_nonempty(row, ["problem", "prompt", "question"])
    answer = _first_nonempty(row, ["answer", "reference_answer", "final_answer"])
    if not problem or not answer:
        return None
    source_dataset = config["data"].get("dataset_name", "local-fixture")
    source_revision = config["data"].get("dataset_revision") or "local"
    source = _first_nonempty(row, ["source", "origin"], source_dataset)
    subject = _first_nonempty(
        row,
        ["problem_type", "subject", "category", "topic"],
        "unknown",
    )
    difficulty = _first_nonempty(row, ["level", "difficulty"], "unknown")
    identity = {"dataset": source_dataset, "source": source, "problem": problem}
    return PromptRecord(
        sample_id=stable_hash(identity, length=20),
        problem=problem,
        reference_answer=answer,
        reference_solution=_first_nonempty(row, ["solution", "reference_solution"], "") or None,
        subject=subject,
        difficulty=difficulty,
        source=source,
        source_dataset=source_dataset,
        source_revision=source_revision,
        split="unassigned",
        metadata={
            "synthetic": row.get("synthetic"),
            "problem_is_valid": row.get("problem_is_valid"),
            "solution_is_valid": row.get("solution_is_valid"),
        },
    )


def _passes_filters(row: dict[str, Any], config: dict[str, Any]) -> bool:
    filters = config["data"].get("filters", {})
    if not isinstance(filters, dict):
        raise ValueError("data.filters must be a mapping")
    minimum_difficulty = filters.get("minimum_difficulty")
    if minimum_difficulty is not None:
        raw_difficulty = next(
            (
                value
                for value in (row.get("difficulty"), row.get("level"))
                if value is not None and str(value).strip()
            ),
            None,
        )
        if raw_difficulty is None:
            return False
        try:
            difficulty = float(raw_difficulty)
        except (TypeError, ValueError):
            return False
        if difficulty < float(minimum_difficulty):
            return False
    return True


def _deduplicate(records: Iterable[PromptRecord]) -> tuple[list[PromptRecord], int]:
    seen: set[str] = set()
    unique: list[PromptRecord] = []
    duplicates = 0
    for record in records:
        key = stable_hash({"problem": record.problem.strip().lower()})
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicates


def prepare_dataset(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(config["paths"]["data_dir"]) / "curated"
    manifest_dir = Path(config["paths"]["data_dir"]) / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    extension = config["data"].get("format", "jsonl")

    source_rows = [dict(row) for row in _iter_source(config)]
    filtered_rows = [row for row in source_rows if _passes_filters(row, config)]
    converted = [convert_prompt_row(row, config) for row in filtered_rows]
    valid = [record for record in converted if record is not None]
    unique, duplicates = _deduplicate(valid)
    random.Random(int(config["project"]["seed"])).shuffle(unique)

    counts = config["data"]["counts"]
    requested = int(counts["smoke"]) + int(counts["validation"]) + int(counts["train"])
    if len(unique) < requested:
        if not config["data"].get("allow_smaller", False):
            raise ValueError(f"Need {requested} records but source only produced {len(unique)}")
        smoke_count = min(int(counts["smoke"]), len(unique))
        remaining = len(unique) - smoke_count
        validation_count = min(int(counts["validation"]), max(0, remaining // 4))
        train_count = remaining - validation_count
    else:
        smoke_count = int(counts["smoke"])
        validation_count = int(counts["validation"])
        train_count = int(counts["train"])

    splits = {
        "smoke": unique[:smoke_count],
        "validation": unique[smoke_count : smoke_count + validation_count],
        "train": unique[
            smoke_count + validation_count : smoke_count + validation_count + train_count
        ],
    }
    output_paths: dict[str, Path] = {}
    for split, records in splits.items():
        assigned = [record.model_copy(update={"split": split}) for record in records]
        path = output_dir / f"{split}.{extension}"
        write_records(path, [record.model_dump(mode="json") for record in assigned])
        output_paths[split] = path

    report = {
        "source_rows": len(source_rows),
        "filtered_rows": len(filtered_rows),
        "rows_removed_by_filters": len(source_rows) - len(filtered_rows),
        "valid_rows": len(valid),
        "duplicates_removed": duplicates,
        "split_counts": {name: len(records) for name, records in splits.items()},
        "subject_counts": dict(Counter(record.subject for record in unique)),
        "source_counts": dict(Counter(record.source for record in unique)),
    }
    report_path = output_dir / "data_card.json"
    write_json(report_path, report)
    files = [*output_paths.values(), report_path]
    manifest = build_manifest(
        artifact_type="prompt_dataset",
        stage="data.prepare",
        config=config,
        files=files,
        record_count=sum(len(records) for records in splits.values()),
        success_count=sum(len(records) for records in splits.values()),
        metadata=report,
    )
    save_manifest(manifest_dir / "data_prepare.json", manifest)
    output_paths["manifest"] = manifest_dir / "data_prepare.json"
    return output_paths
