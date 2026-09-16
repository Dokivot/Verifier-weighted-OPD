#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
SUITE="${2:-regression}"
CHECKPOINT="${3:-}"
uv run --no-sync opd budget --config "$CONFIG"
if [[ -n "$CHECKPOINT" ]]; then
  uv run --no-sync opd evaluate --suite "$SUITE" --checkpoint "$CHECKPOINT" --config "$CONFIG"
else
  uv run --no-sync opd evaluate --suite "$SUITE" --config "$CONFIG"
fi
uv run --no-sync opd budget --config "$CONFIG"
