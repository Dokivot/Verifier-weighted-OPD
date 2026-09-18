#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/sure_k2_smoke.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_smoke/$(date +%Y%m%d_%H%M%S)}"
INPUT="artifacts/sure_k2_24h/data/curated/smoke.parquet"

mkdir -p "$LOG_DIR"
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: $INPUT is missing; run scripts/prepare_data.sh configs/sure_k2_24h.yaml first" >&2
  exit 2
fi

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/00_budget_before.log"
scripts/train.sh "$CONFIG" 2>&1 | tee "$LOG_DIR/10_train.log"
uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget_after.log"

