#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/teacher.yaml}"
ROUND="${2:-0}"
uv run --no-sync opd budget --config "$CONFIG"
uv run --no-sync opd teacher annotate --round "$ROUND" --config "$CONFIG"
uv run --no-sync opd budget --config "$CONFIG"
