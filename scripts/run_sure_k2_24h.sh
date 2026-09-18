#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/sure_k2_24h.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_24h/$(date +%Y%m%d_%H%M%S)}"
START_STAGE="${START_STAGE:-10}"
RESUME_FROM="${RESUME_FROM:-}"
FINAL="artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/final"
BASE_EVAL="artifacts/sure_k2_24h/evaluation/base"
SURE_EVAL="artifacts/sure_k2_24h/evaluation/sure_k2"
REPORT_DIR="artifacts/sure_k2_24h/reports"

case "$START_STAGE" in
  10|11|12|13|20|30|31|40|41|50) ;;
  *)
    echo "ERROR: START_STAGE must be one of 10,11,12,13,20,30,31,40,41,50" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR" "$REPORT_DIR"

run_logged() {
  local stage="$1"
  local name="$2"
  shift 2
  if ((10#$stage < 10#$START_STAGE)); then
    printf 'Skipping stage %s (%s); START_STAGE=%s\n' "$stage" "$name" "$START_STAGE"
    return
  fi
  "$@" 2>&1 | tee "$LOG_DIR/${stage}_${name}.log"
}

run_logged 10 prepare scripts/prepare_data.sh "$CONFIG"
run_logged 11 fetch_math500 uv run --no-sync opd data fetch-eval \
  --name math500 --config "$CONFIG"
run_logged 12 fetch_amc23 uv run --no-sync opd data fetch-eval \
  --name amc23 --config "$CONFIG"
run_logged 13 audit_contamination uv run --no-sync opd data audit-contamination --config "$CONFIG"

if ((10#$START_STAGE <= 20)); then
  if [[ "${SKIP_READINESS_CHECK:-0}" != "1" ]]; then
    uv run --no-sync python scripts/validate_sure_k2_readiness.py --config "$CONFIG"
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    run_logged 20 train scripts/train.sh "$CONFIG" "$RESUME_FROM"
  else
    run_logged 20 train scripts/train.sh "$CONFIG"
  fi
fi

run_logged 30 base_math500 uv run --no-sync opd evaluate \
  --suite math500 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_logged 31 base_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_logged 40 sure_math500 uv run --no-sync opd evaluate \
  --suite math500 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_logged 41 sure_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_logged 50 compare_math500 uv run --no-sync opd compare \
  --baseline "$BASE_EVAL/math500/summary.json" \
  --candidate "$SURE_EVAL/math500/summary.json" \
  --output "$REPORT_DIR/math500_base_vs_sure.json" \
  --config "$CONFIG"
run_logged 50 compare_amc23 uv run --no-sync opd compare \
  --baseline "$BASE_EVAL/amc23/summary.json" \
  --candidate "$SURE_EVAL/amc23/summary.json" \
  --output "$REPORT_DIR/amc23_base_vs_sure.json" \
  --config "$CONFIG"

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget.log"
