#!/usr/bin/env python3
"""Validate the measured steady-state pilot throughput before formal SuRe K2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    output_dir = Path(config["training"]["output_dir"])
    telemetry_path = output_dir / "telemetry" / "step_000002.json"
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Missing steady-state pilot telemetry: {telemetry_path}")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    step_seconds = float(telemetry["step_seconds"])
    max_steps = int(config["training"]["max_steps"])
    formal_config = config_path.parent / "sure_k2_24h.yaml"
    if config_path.name == "sure_k2_pilot.yaml" and formal_config.exists():
        with formal_config.open(encoding="utf-8") as handle:
            formal = yaml.safe_load(handle) or {}
        max_steps = int(formal["training"]["max_steps"])
    maximum = float(config.get("pilot", {}).get("max_steady_step_seconds", 1440))
    projected_hours = step_seconds * max_steps / 3600
    report = {
        "steady_state_step": 2,
        "steady_state_step_seconds": step_seconds,
        "formal_max_steps": max_steps,
        "projected_training_hours": projected_hours,
        "maximum_steady_step_seconds": maximum,
        "status": "passed" if step_seconds <= maximum else "failed",
    }
    report_path = output_dir / "pilot_budget_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if step_seconds > maximum:
        raise RuntimeError(
            f"Pilot is too slow for the 24-hour plan: {step_seconds:.1f}s > {maximum:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
