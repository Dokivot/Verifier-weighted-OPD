#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for CONFIG in \
  configs/dense_opd.yaml \
  configs/random_budget_b50.yaml \
  configs/verifier_filtered_b50.yaml \
  configs/vfs_b50.yaml \
  configs/vfs_b25.yaml; do
  scripts/select_annotations.sh "$CONFIG" 0
done

# Compute each Teacher state once. Budget methods reuse exact deterministic annotations
# through their selection manifests instead of repeating the same 7B forward passes.
scripts/annotate_teacher.sh configs/dense_opd.yaml 0

for CONFIG in \
  configs/dense_opd.yaml \
  configs/random_budget_b50.yaml \
  configs/verifier_filtered_b50.yaml \
  configs/vfs_b50.yaml \
  configs/vfs_b25.yaml; do
  uv run --no-sync opd data build-view \
    --round 0 \
    --method vanilla-opd \
    --config "$CONFIG"
done
