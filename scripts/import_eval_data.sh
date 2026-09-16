#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?usage: scripts/import_eval_data.sh CONFIG NAME INPUT}"
NAME="${2:?usage: scripts/import_eval_data.sh CONFIG NAME INPUT}"
INPUT="${3:?usage: scripts/import_eval_data.sh CONFIG NAME INPUT}"
uv run --no-sync opd data import-eval --name "$NAME" --input "$INPUT" --config "$CONFIG"
