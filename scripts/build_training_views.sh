#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
ROUND="${2:-0}"
for METHOD in vanilla-opd verifier-opd confidence-opd weighted-opd; do
  uv run --no-sync opd data build-view --round "$ROUND" --method "$METHOD" --config "$CONFIG"
done
