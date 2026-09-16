from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from opd.tableio import read_records

RecordModel = TypeVar("RecordModel", bound=BaseModel)


def chunks(records: list[RecordModel], size: int) -> Iterable[tuple[int, list[RecordModel]]]:
    if size <= 0:
        raise ValueError("shard size must be positive")
    for start in range(0, len(records), size):
        yield start // size, records[start : start + size]


def load_complete_shard(
    path: Path,
    *,
    model: type[RecordModel],
    expected_count: int,
    is_complete: Callable[[RecordModel], bool],
) -> list[RecordModel] | None:
    if not path.exists():
        return None
    try:
        records = [model.model_validate(row) for row in read_records(path)]
    except (OSError, ValueError):
        return None
    if len(records) != expected_count or not all(is_complete(record) for record in records):
        return None
    return records


def shard_path(directory: Path, index: int, extension: str) -> Path:
    return directory / f"part-{index:05d}.{extension}"
