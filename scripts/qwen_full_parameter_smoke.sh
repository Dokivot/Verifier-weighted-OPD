#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/qwen_full_parameter_smoke.yaml}"

test -f artifacts/qwen_gpu_smoke/data/annotations/round_0/teacher.parquet || {
  echo "ERROR: run scripts/qwen_gpu_smoke.sh first to create smoke annotations" >&2
  exit 2
}

uv run --no-sync opd data build-view \
  --round 0 \
  --method vanilla-opd \
  --config "$CONFIG"
scripts/train.sh "$CONFIG"
scripts/merge_checkpoint.sh \
  "$CONFIG" \
  artifacts/qwen_gpu_smoke/checkpoints/full_parameter_seed42/best \
  artifacts/qwen_gpu_smoke/merged/full_parameter
uv run --no-sync opd evaluate \
  --suite smoke \
  --checkpoint artifacts/qwen_gpu_smoke/merged/full_parameter \
  --config "$CONFIG"

echo "Full-parameter Qwen GPU smoke passed."
