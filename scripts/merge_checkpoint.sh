#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?usage: scripts/merge_checkpoint.sh CONFIG ADAPTER OUTPUT}"
ADAPTER="${2:?usage: scripts/merge_checkpoint.sh CONFIG ADAPTER OUTPUT}"
OUTPUT="${3:?usage: scripts/merge_checkpoint.sh CONFIG ADAPTER OUTPUT}"
uv run --no-sync opd checkpoint merge --adapter "$ADAPTER" --output "$OUTPUT" --config "$CONFIG"
