#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/tiny_gpu_smoke.yaml}"

uv run --no-sync opd doctor --config "$CONFIG"
uv run --no-sync opd audit sparse-kl --output artifacts/tiny_gpu_smoke/audits/sparse_kl.json --config "$CONFIG"
uv run --no-sync opd data prepare --config "$CONFIG"
uv run --no-sync opd data audit-contamination --config "$CONFIG"
uv run --no-sync opd rollout generate --round 0 --config "$CONFIG"
uv run --no-sync opd verify math --round 0 --config "$CONFIG"
uv run --no-sync opd teacher annotate --round 0 --config "$CONFIG"
uv run --no-sync opd data build-view --round 0 --method weighted-opd --config "$CONFIG"
uv run --no-sync accelerate launch --num_processes 1 -m opd train --config "$CONFIG"
uv run --no-sync opd checkpoint merge \
  --adapter artifacts/tiny_gpu_smoke/checkpoints/weighted_opd_seed42/best \
  --output artifacts/tiny_gpu_smoke/merged/weighted_opd \
  --config "$CONFIG"
uv run --no-sync opd evaluate \
  --suite smoke \
  --checkpoint artifacts/tiny_gpu_smoke/merged/weighted_opd \
  --config "$CONFIG"
uv run --no-sync opd checkpoint promote \
  --baseline artifacts/tiny_gpu_smoke/evaluation/weighted_opd/smoke/summary.json \
  --candidate artifacts/tiny_gpu_smoke/evaluation/weighted_opd/smoke/summary.json \
  --checkpoint artifacts/tiny_gpu_smoke/checkpoints/weighted_opd_seed42/best \
  --output artifacts/tiny_gpu_smoke/promotion/decision.json \
  --config "$CONFIG"
uv run --no-sync opd report failures \
  --predictions artifacts/tiny_gpu_smoke/evaluation/weighted_opd/smoke/predictions.jsonl \
  --output artifacts/tiny_gpu_smoke/reports/failures.json \
  --config "$CONFIG"
uv run --no-sync opd report build --experiment tiny_gpu_smoke --config "$CONFIG"
uv run --no-sync opd report cost --output-dir artifacts/tiny_gpu_smoke/reports/quality_cost --config "$CONFIG"
