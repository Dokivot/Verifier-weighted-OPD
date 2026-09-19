#!/usr/bin/env python3
"""Validate the measured steady-state pilot throughput before formal SuRe K2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opd.artifacts import build_manifest, load_manifest, save_manifest
from opd.config import config_hash, load_config
from opd.tableio import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = load_config(config_path)
    output_dir = Path(config["training"]["output_dir"])
    telemetry_path = output_dir / "telemetry" / "step_000002.json"
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Missing steady-state pilot telemetry: {telemetry_path}")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    step_seconds = float(telemetry["step_seconds"])
    max_steps = int(config["training"]["max_steps"])
    formal_stem = config_path.stem
    if formal_stem.endswith("_pilot"):
        formal_stem = formal_stem[: -len("_pilot")] + "_24h"
    formal_config = config_path.parent / f"{formal_stem}.yaml"
    formal = None
    if config_path.name.endswith("_pilot.yaml") and formal_config.exists():
        formal = load_config(formal_config)
        max_steps = int(formal["training"]["max_steps"])
    maximum = float(config.get("pilot", {}).get("max_steady_step_seconds", 1440))
    projected_hours = step_seconds * max_steps / 3600
    report = {
        "steady_state_step": 2,
        "steady_state_step_seconds": step_seconds,
        "formal_max_steps": max_steps,
        "projected_training_hours": projected_hours,
        "maximum_steady_step_seconds": maximum,
        "config_hash": config_hash(config),
        "formal_config_hash": config_hash(formal) if formal is not None else None,
        "run_id": telemetry.get("run_id"),
        "hardware_fingerprint": telemetry.get("hardware_fingerprint"),
        "status": "passed" if step_seconds <= maximum else "failed",
    }
    report_path = output_dir / "pilot_budget_report.json"
    write_json(report_path, report)
    manifest_path = output_dir / "manifest.json"
    existing_manifest = load_manifest(manifest_path)
    files = [
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    ]
    refreshed_manifest = build_manifest(
        artifact_type=existing_manifest.artifact_type,
        stage=existing_manifest.stage,
        config=config,
        files=files,
        record_count=existing_manifest.record_count,
        success_count=existing_manifest.success_count,
        failure_count=existing_manifest.failure_count,
        upstream_artifact_ids=existing_manifest.upstream_artifact_ids,
        metadata={**existing_manifest.metadata, "pilot_budget_report": str(report_path)},
    )
    save_manifest(manifest_path, refreshed_manifest)
    print(json.dumps(report, indent=2))
    if step_seconds > maximum:
        raise RuntimeError(
            f"Pilot is too slow for the 24-hour plan: {step_seconds:.1f}s > {maximum:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
