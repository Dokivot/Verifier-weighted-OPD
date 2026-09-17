#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"
CONFIG="${1:-configs/smoke.yaml}"

uv run --no-sync python -m opd doctor --config "$CONFIG"
uv run --no-sync python -m opd audit sparse-kl \
  --output artifacts/smoke/audits/sparse_kl.json --config "$CONFIG"
uv run --no-sync python -m opd data prepare --config "$CONFIG"
uv run --no-sync python -m opd data audit-contamination --config "$CONFIG"
uv run --no-sync python -m opd rollout generate --round 0 --config "$CONFIG"
uv run --no-sync python -m opd verify math --round 0 --config "$CONFIG"
uv run --no-sync python -m opd data select-annotations --round 0 --config "$CONFIG"
uv run --no-sync python -m opd teacher annotate --round 0 --config "$CONFIG"
uv run --no-sync python -m opd data build-view \
  --round 0 --method weighted-opd --config "$CONFIG"
uv run --no-sync python -m opd train --config "$CONFIG"
uv run --no-sync python -m opd evaluate --suite smoke --config "$CONFIG"
uv run --no-sync python -m opd checkpoint promote \
  --baseline artifacts/smoke/evaluation/weighted_opd/smoke/summary.json \
  --candidate artifacts/smoke/evaluation/weighted_opd/smoke/summary.json \
  --checkpoint artifacts/smoke/checkpoints/weighted_opd_seed42/checkpoint.json \
  --output artifacts/smoke/promotion/decision.json \
  --config "$CONFIG"
uv run --no-sync python -m opd report failures \
  --predictions artifacts/smoke/evaluation/weighted_opd/smoke/predictions.jsonl \
  --output artifacts/smoke/reports/failures.json \
  --config "$CONFIG"
uv run --no-sync python -m opd report build --experiment smoke --config "$CONFIG"
uv run --no-sync python -m opd report cost \
  --output-dir artifacts/smoke/reports/quality_cost --config "$CONFIG"
