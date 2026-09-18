#!/usr/bin/env python3
"""Reject formal SuRe K2 training until its required GPU and data gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required formal-training artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Required formal-training artifact is not a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    del arguments

    root = Path("artifacts/sure_k2_24h")
    contamination_manifest = root / "data/contamination/manifest.json"
    clean_train = root / "data/contamination/train_clean.parquet"
    smoke_summary = _load_json(root / "smoke/checkpoint/training_summary.json")
    pilot_summary = _load_json(root / "pilot/checkpoint/training_summary.json")
    pilot_budget = _load_json(root / "pilot/checkpoint/pilot_budget_report.json")
    if not contamination_manifest.exists() or not clean_train.exists():
        raise FileNotFoundError("Missing contamination manifest or clean training split")
    if smoke_summary.get("status") != "completed":
        raise RuntimeError("GPU smoke has not completed successfully")
    if pilot_summary.get("status") != "completed" or pilot_summary.get("global_step") != 2:
        raise RuntimeError("Two-step resumability pilot has not completed successfully")
    if pilot_budget.get("status") != "passed":
        raise RuntimeError("Pilot throughput does not satisfy the formal 24-hour budget gate")
    print("SuRe K2 readiness passed: decontamination audit, GPU smoke, and 2-step pilot are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
