#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?usage: scripts/evaluate_lighteval.sh CHECKPOINT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-artifacts/lighteval}"
CONFIG="${3:-configs/main.yaml}"
TASKS="${LIGHEVAL_TASKS:-}"

uv run --no-sync opd budget --config "$CONFIG"
if [[ -n "$TASKS" ]]; then
  uv run --no-sync opd benchmark run \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --tasks "$TASKS" \
    --config "$CONFIG"
else
  uv run --no-sync opd benchmark run \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG"
fi
uv run --no-sync opd budget --config "$CONFIG"
