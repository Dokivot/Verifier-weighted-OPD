#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/sure_k2_smoke.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_smoke/$(date +%Y%m%d_%H%M%S)}"

readarray -t CONFIG_VALUES < <(
  uv run --no-sync python - "$CONFIG" <<'PY'
import sys

from opd.config import load_config


config = load_config(sys.argv[1])
print(config["training"]["input_path"])
PY
)
if (( ${#CONFIG_VALUES[@]} != 1 )); then
  echo "ERROR: smoke config must provide training.input_path" >&2
  exit 2
fi
INPUT="${CONFIG_VALUES[0]}"

mkdir -p "$LOG_DIR"
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: $INPUT is missing; run scripts/prepare_data.sh $CONFIG first" >&2
  exit 2
fi

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/00_budget_before.log"
scripts/train.sh "$CONFIG" 2>&1 | tee "$LOG_DIR/10_train.log"
uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget_after.log"
