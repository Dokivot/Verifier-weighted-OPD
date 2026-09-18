#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/sure_k2_pilot.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_pilot/$(date +%Y%m%d_%H%M%S)}"

readarray -t CONFIG_VALUES < <(
  uv run --no-sync python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from opd.config import load_config


config = load_config(sys.argv[1])
print(config["training"]["input_path"])
print(Path(config["training"]["output_dir"]) / "rolling")
print(Path(config["paths"]["artifact_dir"]) / "data/contamination/manifest.json")
PY
)
if (( ${#CONFIG_VALUES[@]} != 3 )); then
  echo "ERROR: pilot config must provide training and artifact paths" >&2
  exit 2
fi
INPUT="${CONFIG_VALUES[0]}"
ROLLING="${CONFIG_VALUES[1]}"
CONTAMINATION_MANIFEST="${CONFIG_VALUES[2]}"

mkdir -p "$LOG_DIR"
if [[ ! -f "$INPUT" || ! -f "$CONTAMINATION_MANIFEST" ]]; then
  echo "ERROR: decontaminated pilot data is missing; run stages 10–13 of run_sure_k2_24h.sh first" >&2
  exit 2
fi

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/00_budget_before.log"
scripts/train.sh "$CONFIG" 2>&1 | tee "$LOG_DIR/10_step_1.log"
scripts/train.sh "$CONFIG" "$ROLLING" 2>&1 | tee "$LOG_DIR/20_resume_step_2.log"
uv run --no-sync python scripts/check_sure_k2_pilot.py --config "$CONFIG" \
  | tee "$LOG_DIR/30_budget_projection.log"
uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget_after.log"
