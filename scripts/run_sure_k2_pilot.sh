#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/sure_k2_pilot.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_pilot/$(date +%Y%m%d_%H%M%S)}"
ROLLING="artifacts/sure_k2_24h/pilot/checkpoint/rolling"
INPUT="artifacts/sure_k2_24h/data/contamination/train_clean.parquet"

mkdir -p "$LOG_DIR"
if [[ ! -f "$INPUT" || ! -f "artifacts/sure_k2_24h/data/contamination/manifest.json" ]]; then
  echo "ERROR: decontaminated pilot data is missing; run stages 10–13 of run_sure_k2_24h.sh first" >&2
  exit 2
fi

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/00_budget_before.log"
scripts/train.sh "$CONFIG" 2>&1 | tee "$LOG_DIR/10_step_1.log"
scripts/train.sh "$CONFIG" "$ROLLING" 2>&1 | tee "$LOG_DIR/20_resume_step_2.log"
uv run --no-sync python scripts/check_sure_k2_pilot.py --config "$CONFIG" \
  | tee "$LOG_DIR/30_budget_projection.log"
uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget_after.log"
