#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPERIMENT_NAME=dense_vanilla_lora_b100 \
LOG_DIR="${LOG_DIR:-logs/dense_vanilla_lora_mvp/$(date +%Y%m%d_%H%M%S)}" \
START_STAGE="${START_STAGE:-13}" \
scripts/run_dense_vanilla_mvp.sh configs/dense_vanilla_lora_mvp.yaml
