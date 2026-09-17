#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
ROUND="${2:-0}"
uv run --no-sync opd data select-annotations --round "$ROUND" --config "$CONFIG"
