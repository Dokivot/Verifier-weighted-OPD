#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/data.yaml}"
uv run --no-sync opd data prepare --config "$CONFIG"
