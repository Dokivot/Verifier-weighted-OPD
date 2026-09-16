from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from opd.exceptions import DependencyError


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def atomic_write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, target)
    return target


def write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    if count == 0:
        raise ValueError(f"Refusing to commit empty JSONL artifact: {target}")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            records.append(value)
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    if target.suffix == ".jsonl":
        return write_jsonl(target, records)
    if target.suffix != ".parquet":
        raise ValueError(f"Unsupported record format: {target.suffix}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DependencyError("Parquet output requires `pip install -e .[data]`") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    pq.write_table(pa.Table.from_pylist(records), partial)
    os.replace(partial, target)
    return target


def read_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix == ".jsonl":
        return read_jsonl(target)
    if target.suffix != ".parquet":
        raise ValueError(f"Unsupported record format: {target.suffix}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DependencyError("Parquet input requires `pip install -e .[data]`") from exc
    return cast(list[dict[str, Any]], pq.read_table(target).to_pylist())
