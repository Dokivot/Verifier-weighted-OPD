#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
EXPERIMENT="${2:-opd_seed42}"
COST_OUTPUT="${3:-}"

uv run --no-sync opd report build --experiment "$EXPERIMENT" --config "$CONFIG"
if [[ -n "$COST_OUTPUT" ]]; then
  uv run --no-sync opd report cost --output-dir "$COST_OUTPUT" --config "$CONFIG"
fi
