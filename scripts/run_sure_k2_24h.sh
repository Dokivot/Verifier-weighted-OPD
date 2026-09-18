#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/sure_k2_24h.yaml}"
LOG_DIR="${LOG_DIR:-logs/sure_k2_24h/$(date +%Y%m%d_%H%M%S)}"
START_STAGE="${START_STAGE:-10}"
RESUME_FROM="${RESUME_FROM:-}"

CONFIG_VALUES_FILE="$(mktemp)"
trap 'rm -f "$CONFIG_VALUES_FILE"' EXIT
uv run --no-sync python - "$CONFIG" >"$CONFIG_VALUES_FILE" <<'PY'
from pathlib import Path
import sys

from opd.config import load_config


config = load_config(sys.argv[1])
training = config["training"]
evaluation = config["evaluation"]
benchmark = config["benchmark"]
report = config["report"]

values = [
    str(Path(training["output_dir"]) / "final"),
    str(evaluation["output_dir"]),
    str(evaluation["candidate_output_dir"]),
    str(benchmark["base_output_dir"]),
    str(benchmark["candidate_output_dir"]),
    str(benchmark["base_math_output_dir"]),
    str(benchmark["candidate_math_output_dir"]),
    str(report["output_dir"]),
    str(report["experiment_id"]),
    str(config["models"]["student"]["name"]),
]
print("\n".join(values))
PY
mapfile -t CONFIG_VALUES <"$CONFIG_VALUES_FILE"
if (( ${#CONFIG_VALUES[@]} != 10 )); then
  echo "ERROR: config must provide all SuRe K2 pipeline paths and identities" >&2
  exit 2
fi

FINAL="${CONFIG_VALUES[0]}"
BASE_EVAL="${CONFIG_VALUES[1]}"
SURE_EVAL="${CONFIG_VALUES[2]}"
BASE_OOD="${CONFIG_VALUES[3]}"
SURE_OOD="${CONFIG_VALUES[4]}"
BASE_MATH="${CONFIG_VALUES[5]}"
SURE_MATH="${CONFIG_VALUES[6]}"
REPORT_DIR="${CONFIG_VALUES[7]}"
EXPERIMENT_ID="${CONFIG_VALUES[8]}"
BASE_CHECKPOINT="${CONFIG_VALUES[9]}"

case "$START_STAGE" in
  10|11|12|13|14|15|16|17|20|40|41|42|43|50) ;;
  *)
    echo "ERROR: START_STAGE must be one of 10,11,12,13,14,15,16,17,20,40,41,42,43,50" >&2
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

run_logged 14 base_math500 uv run --no-sync opd evaluate \
  --suite math500 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_logged 15 base_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_logged 16 base_ifeval uv run --no-sync opd benchmark run \
  --checkpoint "$BASE_CHECKPOINT" --output-dir "$BASE_OOD" --tasks ifeval --config "$CONFIG"
run_logged 17 base_math500_official uv run --no-sync opd benchmark run \
  --checkpoint "$BASE_CHECKPOINT" --output-dir "$BASE_MATH" --tasks math500 --config "$CONFIG"

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

run_logged 40 sure_math500 uv run --no-sync opd evaluate \
  --suite math500 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_logged 41 sure_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_logged 42 sure_ifeval uv run --no-sync opd benchmark run \
  --checkpoint "$FINAL" --output-dir "$SURE_OOD" --tasks ifeval --config "$CONFIG"
run_logged 43 sure_math500_official uv run --no-sync opd benchmark run \
  --checkpoint "$FINAL" --output-dir "$SURE_MATH" --tasks math500 --config "$CONFIG"
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
run_logged 50 ood_regression uv run --no-sync opd report ood \
  --baseline "$BASE_OOD" \
  --candidate "sure_k2=$SURE_OOD" \
  --output "$REPORT_DIR/ood_regression.json" \
  --config "$CONFIG"
run_logged 50 build_report uv run --no-sync opd report build \
  --experiment "$EXPERIMENT_ID" --config "$CONFIG"

uv run --no-sync opd budget --config "$CONFIG" | tee "$LOG_DIR/99_budget.log"
