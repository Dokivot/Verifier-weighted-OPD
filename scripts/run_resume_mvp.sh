#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/vfs_weighted_mvp.yaml}"
LOG_DIR="${LOG_DIR:-logs/resume_mvp/$(date +%Y%m%d_%H%M%S)}"
START_STAGE="${START_STAGE:-0}"
ADAPTER="artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/best"
MERGED="artifacts/resume_mvp/merged/vfs_weighted_b50"

case "$START_STAGE" in
  0|1|2|10|11|12|13|14|20|21|22|23|24) ;;
  *)
    echo "ERROR: START_STAGE must be one of 0,1,2,10,11,12,13,14,20,21,22,23,24" >&2
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

run_logged 0 00_prepare scripts/prepare_data.sh "$CONFIG"
run_logged 1 01_fetch_eval scripts/fetch_eval_data.sh "$CONFIG"
run_logged 2 02_contamination scripts/audit_contamination.sh "$CONFIG"
run_logged 10 10_rollout scripts/generate_rollouts.sh "$CONFIG" 0
run_logged 11 11_verify scripts/verify.sh "$CONFIG" 0
run_logged 12 12_select scripts/select_annotations.sh "$CONFIG" 0
run_logged 13 13_teacher scripts/annotate_teacher.sh "$CONFIG" 0
run_logged 14 14_build_view uv run --no-sync opd data build-view \
  --round 0 \
  --method weighted-opd \
  --config "$CONFIG"
run_logged 20 20_train scripts/train.sh "$CONFIG"
run_logged 21 21_merge scripts/merge_checkpoint.sh "$CONFIG" "$ADAPTER" "$MERGED"
run_logged 22 22_regression scripts/evaluate.sh "$CONFIG" regression "$MERGED"
run_logged 23 23_benchmark scripts/evaluate_lighteval.sh \
  "$MERGED" \
  artifacts/resume_mvp/lighteval/vfs_weighted_b50 \
  "$CONFIG"
run_logged 24 24_base_benchmark scripts/evaluate_lighteval.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  artifacts/resume_mvp/lighteval/base \
  "$CONFIG"
