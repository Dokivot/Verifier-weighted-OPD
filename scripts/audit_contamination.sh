#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
uv run --no-sync opd data audit-contamination --config "$CONFIG"
