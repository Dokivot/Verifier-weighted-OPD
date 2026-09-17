#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/dense_vanilla_mvp.yaml}"
LOG_DIR="${LOG_DIR:-logs/dense_vanilla_mvp/$(date +%Y%m%d_%H%M%S)}"
START_STAGE="${START_STAGE:-13}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dense_vanilla_full_b100}"
ADAPTER="artifacts/resume_mvp/checkpoints/${EXPERIMENT_NAME}_seed42/best"
MERGED="artifacts/resume_mvp/merged/${EXPERIMENT_NAME}"

case "$START_STAGE" in
  13|14|20|21|22|23|24) ;;
  *)
    echo "ERROR: START_STAGE must be one of 13,14,20,21,22,23,24" >&2
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

run_logged 13 13_teacher scripts/annotate_teacher.sh "$CONFIG" 0
run_logged 14 14_build_view uv run --no-sync opd data build-view \
  --round 0 \
  --method vanilla-opd \
  --config "$CONFIG"
run_logged 20 20_train scripts/train.sh "$CONFIG"
run_logged 21 21_merge scripts/merge_checkpoint.sh "$CONFIG" "$ADAPTER" "$MERGED"
run_logged 22 22_regression scripts/evaluate.sh "$CONFIG" regression "$MERGED"
run_logged 23 23_benchmark scripts/evaluate_lighteval.sh \
  "$MERGED" \
  "artifacts/resume_mvp/lighteval/${EXPERIMENT_NAME}" \
  "$CONFIG"
run_logged 24 24_base_benchmark scripts/evaluate_lighteval.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  artifacts/resume_mvp/lighteval/base_dense_protocol \
  "$CONFIG"
