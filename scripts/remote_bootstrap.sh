#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || {
  echo "Install uv before bootstrapping this host." >&2
  exit 2
}

source "$ROOT/scripts/autodl_env.sh"
uv sync --frozen --extra data --extra gpu --extra eval --extra tracking --extra dev
uv run --no-sync opd doctor --config configs/main.yaml
nvidia-smi
"$ROOT/scripts/autodl_preflight.sh"
