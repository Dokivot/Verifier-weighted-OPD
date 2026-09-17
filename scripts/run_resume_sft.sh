#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/resume_mvp_sft.yaml}"
LOG_DIR="${LOG_DIR:-logs/resume_mvp/$(date +%Y%m%d_%H%M%S)}"
START_STAGE="${START_STAGE:-30}"
ADAPTER="artifacts/resume_mvp/checkpoints/sft_seed42/best"
MERGED="artifacts/resume_mvp/merged/sft"

case "$START_STAGE" in
  30|31|32|33) ;;
  *)
    echo "ERROR: START_STAGE must be one of 30,31,32,33" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

run_logged() {
  local stage="$1"
  local log_name="$2"
  shift 2
  if ((10#$stage < 10#$START_STAGE)); then
    printf 'Skipping stage %s (%s); START_STAGE=%s\n' "$stage" "$log_name" "$START_STAGE"
    return
  fi
  "$@" 2>&1 | tee "$LOG_DIR/$log_name.log"
}

run_logged 30 30_train_sft scripts/train.sh "$CONFIG"
run_logged 31 31_merge_sft scripts/merge_checkpoint.sh "$CONFIG" "$ADAPTER" "$MERGED"
run_logged 32 32_regression_sft scripts/evaluate.sh "$CONFIG" regression "$MERGED"
run_logged 33 33_benchmark_sft scripts/evaluate_lighteval.sh \
  "$MERGED" \
  artifacts/resume_mvp/lighteval/sft \
  "$CONFIG"
