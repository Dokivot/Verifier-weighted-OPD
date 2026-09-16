from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest
from opd.data.prepare import convert_prompt_row
from opd.exceptions import DependencyError
from opd.schemas import PromptRecord
from opd.tableio import read_records, write_records


def import_evaluation_data(config: dict[str, Any], *, name: str, input_path: str | Path) -> Path:
    allowed_characters = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not name or any(character not in allowed_characters for character in name):
        raise ValueError("Evaluation dataset name must use lowercase letters, digits, '-' or '_'")
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Evaluation source not found: {source_path}")

    conversion_config = deepcopy(config)
    conversion_config["data"]["dataset_name"] = f"evaluation/{name}"
    conversion_config["data"]["dataset_revision"] = (
        config.get("evaluation_data", {}).get(name, {}).get("revision", "local-import")
    )
    converted = [
        convert_prompt_row(dict(row), conversion_config) for row in read_records(source_path)
    ]
    records: list[PromptRecord] = [
        record.model_copy(update={"split": f"evaluation:{name}"})
        for record in converted
        if record is not None
    ]
    if not records:
        raise ValueError(
            f"Evaluation source produced no valid prompt/answer records: {source_path}"
        )

    extension = config["data"].get("format", "jsonl")
    output_dir = Path(config["paths"]["data_dir"]) / "eval"
    output_path = output_dir / f"{name}.{extension}"
    write_records(output_path, [record.model_dump(mode="json") for record in records])
    manifest = build_manifest(
        artifact_type="evaluation_dataset",
        stage="data.import_eval",
        config={**config, "evaluation_import": {"name": name, "input_path": str(source_path)}},
        files=[source_path, output_path],
        record_count=len(records),
        success_count=len(records),
        metadata={
            "name": name,
            "source_path": str(source_path),
            "source_rows": len(converted),
            "dropped_rows": len(converted) - len(records),
        },
    )
    save_manifest(output_path.with_suffix(".manifest.json"), manifest)
    return output_path


def fetch_evaluation_data(config: dict[str, Any], *, name: str) -> Path:
    source = config.get("evaluation_data", {}).get(name)
    if not isinstance(source, dict):
        available = ", ".join(sorted(config.get("evaluation_data", {}))) or "none"
        raise ValueError(f"Unknown evaluation dataset {name!r}; configured datasets: {available}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DependencyError("Evaluation download requires `pip install -e .[data]`") from exc

    dataset_name = str(source["dataset_name"])
    subset = source.get("dataset_subset")
    split = str(source.get("split", "test"))
    revision = str(source["revision"])
    dataset = load_dataset(
        dataset_name,
        subset,
        split=split,
        revision=revision,
    )
    conversion_config = deepcopy(config)
    conversion_config["data"]["dataset_name"] = dataset_name
    conversion_config["data"]["dataset_revision"] = revision
    converted = [convert_prompt_row(dict(row), conversion_config) for row in dataset]
    records = [
        record.model_copy(update={"split": f"evaluation:{name}"})
        for record in converted
        if record is not None
    ]
    if not records:
        raise ValueError(f"Downloaded evaluation dataset produced no valid records: {dataset_name}")

    extension = config["data"].get("format", "jsonl")
    output_path = Path(config["paths"]["data_dir"]) / "eval" / f"{name}.{extension}"
    write_records(output_path, [record.model_dump(mode="json") for record in records])
    manifest = build_manifest(
        artifact_type="evaluation_dataset",
        stage="data.fetch_eval",
        config={
            **config,
            "evaluation_fetch": {
                "name": name,
                "dataset_name": dataset_name,
                "subset": subset,
                "split": split,
                "revision": revision,
            },
        },
        files=[output_path],
        record_count=len(records),
        success_count=len(records),
        metadata={
            "name": name,
            "dataset_name": dataset_name,
            "subset": subset,
            "split": split,
            "revision": revision,
            "source_rows": len(converted),
            "dropped_rows": len(converted) - len(records),
        },
    )
    save_manifest(output_path.with_suffix(".manifest.json"), manifest)
    return output_path
