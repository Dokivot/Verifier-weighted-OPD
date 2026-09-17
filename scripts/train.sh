#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:?usage: scripts/train.sh CONFIG [RESUME_CHECKPOINT]}"
RESUME_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${2:-}}"
uv run --no-sync opd budget --config "$CONFIG"
TRAIN_COMMAND=(uv run --no-sync accelerate launch --num_processes 1 -m opd train --config "$CONFIG")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_COMMAND+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
fi
"${TRAIN_COMMAND[@]}"
uv run --no-sync opd budget --config "$CONFIG"
