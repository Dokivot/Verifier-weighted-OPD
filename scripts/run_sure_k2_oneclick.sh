#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/sure_k2_24h.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/sure_k2_smoke.yaml}"
PILOT_CONFIG="${PILOT_CONFIG:-configs/sure_k2_pilot.yaml}"
FROM_STAGE="${FROM_STAGE:-0}"
FORCE_FROM="${FORCE_FROM:-}"
NO_BOOTSTRAP=0
SKIP_CHECK=0

usage() {
  cat <<'EOF'
Usage: scripts/run_sure_k2_oneclick.sh [options]

Options:
  --config PATH          Formal config (default: configs/sure_k2_24h.yaml)
  --from-stage N         Do not run stages below N
  --force-from N         Rerun stages N and above without deleting artifacts
  --no-bootstrap         Skip dependency installation
  --skip-check           Skip make check (not recommended)
  -h, --help             Show this help

The script resumes from passed stage markers by default. On failure it writes a
failure bundle under logs/sure_k2_oneclick/ and exits non-zero.
EOF
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --from-stage)
      FROM_STAGE="$2"
      shift 2
      ;;
    --force-from)
      FORCE_FROM="$2"
      shift 2
      ;;
    --no-bootstrap)
      NO_BOOTSTRAP=1
      shift
      ;;
    --skip-check)
      SKIP_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$FROM_STAGE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --from-stage must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "$FORCE_FROM" ]] && ! [[ "$FORCE_FROM" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --force-from must be a non-negative integer" >&2
  exit 2
fi

LOG_ROOT="$ROOT/logs/sure_k2_oneclick"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
STATE_DIR="$LOG_ROOT/state"
mkdir -p "$RUN_LOG_DIR" "$STATE_DIR"

CURRENT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
CONFIG_HASH="unknown"
LAST_STAGE="startup"
LAST_LOG=""

write_runtime_snapshot() {
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'stage=%s\n' "$LAST_STAGE"
    printf 'config=%s\n' "$CONFIG"
    printf 'config_hash=%s\n' "$CONFIG_HASH"
    printf 'git_commit=%s\n' "$CURRENT_COMMIT"
    printf 'run_log_dir=%s\n' "$RUN_LOG_DIR"
    printf 'last_log=%s\n' "$LAST_LOG"
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  } >"$STATE_DIR/latest_state.env"
}

write_failure_bundle() {
  local stage="$1"
  local name="$2"
  local exit_code="$3"
  local bundle="$RUN_LOG_DIR/failure_${stage}_${name}"
  mkdir -p "$bundle"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'stage=%s\n' "$stage"
    printf 'name=%s\n' "$name"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'config=%s\n' "$CONFIG"
    printf 'config_hash=%s\n' "$CONFIG_HASH"
    printf 'git_commit=%s\n' "$CURRENT_COMMIT"
    printf 'log=%s\n' "$LAST_LOG"
    printf 'retry_command=cd %q && scripts/run_sure_k2_oneclick.sh --config %q\n' "$ROOT" "$CONFIG"
  } >"$bundle/summary.env"
  git status --short >"$bundle/git_status.txt" 2>&1 || true
  git log -5 --oneline >"$bundle/git_log.txt" 2>&1 || true
  (nvidia-smi || true) >"$bundle/nvidia_smi.txt" 2>&1
  (df -h "$ROOT" || true) >"$bundle/disk.txt" 2>&1
  (free -h || true) >"$bundle/memory.txt" 2>&1
  (pgrep -af 'run_sure_k2|accelerate|opd|lighteval|vllm' || true) >"$bundle/processes.txt" 2>&1
  if [[ -n "$LAST_LOG" && -f "$LAST_LOG" ]]; then
    tail -n 240 "$LAST_LOG" >"$bundle/log_tail.txt" 2>&1 || true
  fi
  cp "$bundle/summary.env" "$STATE_DIR/latest_failure.env"
  rm -rf "$STATE_DIR/latest_failure"
  mkdir -p "$STATE_DIR/latest_failure"
  cp "$bundle"/* "$STATE_DIR/latest_failure/"
  cat <<EOF

Stage $stage ($name) failed with exit code $exit_code.
Failure bundle: $bundle
Latest failure: $STATE_DIR/latest_failure
The stage was not marked successful. Fix the error, pull the corrected code,
then rerun the same command; passed stages will be skipped automatically.
EOF
}

marker_path() {
  printf '%s/stage_%03d.ok' "$STATE_DIR" "$1"
}

stage_is_passed() {
  local stage="$1"
  local marker
  marker="$(marker_path "$stage")"
  [[ -f "$marker" ]] || return 1
  if [[ -n "$FORCE_FROM" ]] && ((stage >= FORCE_FROM)); then
    return 1
  fi
  if ((stage <= 1)); then
    grep -Fxq "git_commit=$CURRENT_COMMIT" "$marker"
  else
    grep -Fxq "config_hash=$CONFIG_HASH" "$marker"
  fi
}

mark_stage_passed() {
  local stage="$1"
  local name="$2"
  local marker
  marker="$(marker_path "$stage")"
  {
    printf 'stage=%s\n' "$stage"
    printf 'name=%s\n' "$name"
    printf 'config=%s\n' "$CONFIG"
    printf 'config_hash=%s\n' "$CONFIG_HASH"
    printf 'git_commit=%s\n' "$CURRENT_COMMIT"
    printf 'log=%s\n' "$LAST_LOG"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
  } >"$marker"
  printf '%s\n' "$stage" >"$STATE_DIR/last_passed_stage"
}

run_stage() {
  local stage="$1"
  local name="$2"
  shift 2
  LAST_STAGE="$stage:$name"
  LAST_LOG="$RUN_LOG_DIR/${stage}_${name}.log"
  write_runtime_snapshot
  if ((stage < FROM_STAGE)); then
    printf 'Skipping stage %s (%s): below --from-stage=%s\n' "$stage" "$name" "$FROM_STAGE"
    return 0
  fi
  if stage_is_passed "$stage"; then
    printf 'Skipping stage %s (%s): successful marker is present\n' "$stage" "$name"
    return 0
  fi
  printf '\n=== Stage %s: %s ===\n' "$stage" "$name" | tee "$LAST_LOG"
  set +e
  ( "$@" ) 2>&1 | tee -a "$LAST_LOG"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if ((exit_code != 0)); then
    write_failure_bundle "$stage" "$name" "$exit_code"
    return "$exit_code"
  fi
  mark_stage_passed "$stage" "$name"
  write_runtime_snapshot
  printf 'Stage %s passed.\n' "$stage" | tee -a "$LAST_LOG"
}

run_reports() {
  uv run --no-sync opd compare \
    --baseline "$BASE_EVAL/math500/summary.json" \
    --candidate "$SURE_EVAL/math500/summary.json" \
    --output "$REPORT_DIR/math500_base_vs_sure.json" \
    --config "$CONFIG"
  uv run --no-sync opd compare \
    --baseline "$BASE_EVAL/amc23/summary.json" \
    --candidate "$SURE_EVAL/amc23/summary.json" \
    --output "$REPORT_DIR/amc23_base_vs_sure.json" \
    --config "$CONFIG"
  uv run --no-sync opd report ood \
    --baseline "$BASE_OOD" \
    --candidate "sure_k2=$SURE_OOD" \
    --output "$REPORT_DIR/ood_regression.json" \
    --config "$CONFIG"
  uv run --no-sync opd report build \
    --experiment "$EXPERIMENT_ID" \
    --config "$CONFIG"
}

run_training() {
  if [[ -d "$TRAINING_OUTPUT/rolling" ]]; then
    scripts/train.sh "$CONFIG" "$TRAINING_OUTPUT/rolling"
  else
    scripts/train.sh "$CONFIG"
  fi
}

if ((NO_BOOTSTRAP == 1)); then
  printf 'Bootstrap was explicitly skipped.\n'
else
  run_stage 0 bootstrap scripts/remote_bootstrap.sh
fi

if ((SKIP_CHECK == 1)); then
  printf 'CPU checks were explicitly skipped.\n'
else
  run_stage 1 check bash -c 'set -o pipefail; make check'
fi

CONFIG_HASH="$(uv run --no-sync python - "$CONFIG" <<'PY'
import sys
from opd.config import config_hash, load_config

print(config_hash(load_config(sys.argv[1])))
PY
)"

CONFIG_VALUES=()
while IFS= read -r value; do
  CONFIG_VALUES+=("$value")
done < <(
  uv run --no-sync python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from opd.config import load_config

config = load_config(sys.argv[1])
training = config["training"]
evaluation = config["evaluation"]
benchmark = config["benchmark"]
report = config["report"]
for value in (
    training["output_dir"],
    Path(training["output_dir"]) / "final",
    evaluation["output_dir"],
    evaluation["candidate_output_dir"],
    benchmark["base_output_dir"],
    benchmark["candidate_output_dir"],
    benchmark["base_math_output_dir"],
    benchmark["candidate_math_output_dir"],
    report["output_dir"],
    report["experiment_id"],
    config["models"]["student"]["name"],
):
    print(value)
PY
)
if ((${#CONFIG_VALUES[@]} != 11)); then
  echo "ERROR: formal config did not provide all expected paths" >&2
  exit 2
fi

TRAINING_OUTPUT="${CONFIG_VALUES[0]}"
FINAL="${CONFIG_VALUES[1]}"
BASE_EVAL="${CONFIG_VALUES[2]}"
SURE_EVAL="${CONFIG_VALUES[3]}"
BASE_OOD="${CONFIG_VALUES[4]}"
SURE_OOD="${CONFIG_VALUES[5]}"
BASE_MATH="${CONFIG_VALUES[6]}"
SURE_MATH="${CONFIG_VALUES[7]}"
REPORT_DIR="${CONFIG_VALUES[8]}"
EXPERIMENT_ID="${CONFIG_VALUES[9]}"
BASE_CHECKPOINT="${CONFIG_VALUES[10]}"

run_stage 10 prepare scripts/prepare_data.sh "$CONFIG"
run_stage 11 fetch_math500 uv run --no-sync opd data fetch-eval --name math500 --config "$CONFIG"
run_stage 12 fetch_amc23 uv run --no-sync opd data fetch-eval --name amc23 --config "$CONFIG"
run_stage 13 audit_contamination uv run --no-sync opd data audit-contamination --config "$CONFIG"
run_stage 18 smoke scripts/run_sure_k2_smoke.sh "$SMOKE_CONFIG"
run_stage 19 pilot scripts/run_sure_k2_pilot.sh "$PILOT_CONFIG"
run_stage 14 base_math500 uv run --no-sync opd evaluate \
  --suite math500 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_stage 15 base_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --output-dir "$BASE_EVAL" --config "$CONFIG"
run_stage 16 base_ifeval uv run --no-sync opd benchmark run \
  --checkpoint "$BASE_CHECKPOINT" --output-dir "$BASE_OOD" --tasks ifeval --config "$CONFIG"
run_stage 17 base_math500_official uv run --no-sync opd benchmark run \
  --checkpoint "$BASE_CHECKPOINT" --output-dir "$BASE_MATH" --tasks math500 --config "$CONFIG"
run_stage 20 readiness uv run --no-sync python scripts/validate_sure_k2_readiness.py --config "$CONFIG"
run_stage 30 train run_training
run_stage 40 sure_math500 uv run --no-sync opd evaluate \
  --suite math500 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_stage 41 sure_amc23 uv run --no-sync opd evaluate \
  --suite amc23 --checkpoint "$FINAL" --output-dir "$SURE_EVAL" --config "$CONFIG"
run_stage 42 sure_ifeval uv run --no-sync opd benchmark run \
  --checkpoint "$FINAL" --output-dir "$SURE_OOD" --tasks ifeval --config "$CONFIG"
run_stage 43 sure_math500_official uv run --no-sync opd benchmark run \
  --checkpoint "$FINAL" --output-dir "$SURE_MATH" --tasks math500 --config "$CONFIG"
run_stage 50 reports run_reports
run_stage 51 budget uv run --no-sync opd budget --config "$CONFIG"

cat <<EOF

SuRe K2 experiment completed.
Run logs: $RUN_LOG_DIR
State: $STATE_DIR
Final checkpoint: $FINAL
Report directory: $REPORT_DIR
EOF
