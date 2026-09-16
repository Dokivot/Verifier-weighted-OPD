#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?usage: scripts/promote_checkpoint.sh CONFIG BASELINE CANDIDATE CHECKPOINT OUTPUT}"
BASELINE="${2:?usage: scripts/promote_checkpoint.sh CONFIG BASELINE CANDIDATE CHECKPOINT OUTPUT}"
CANDIDATE="${3:?usage: scripts/promote_checkpoint.sh CONFIG BASELINE CANDIDATE CHECKPOINT OUTPUT}"
CHECKPOINT="${4:?usage: scripts/promote_checkpoint.sh CONFIG BASELINE CANDIDATE CHECKPOINT OUTPUT}"
OUTPUT="${5:?usage: scripts/promote_checkpoint.sh CONFIG BASELINE CANDIDATE CHECKPOINT OUTPUT}"

uv run --no-sync opd checkpoint promote \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --config "$CONFIG"
