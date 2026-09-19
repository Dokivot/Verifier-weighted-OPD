#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || {
  if [[ -n "${OPD_OFFLINE_WHEELHOUSE:-}" ]] && command -v python3 >/dev/null; then
    python3 -m pip install --user --no-index \
      --find-links "$OPD_OFFLINE_WHEELHOUSE" uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
}
command -v uv >/dev/null || {
  echo "Install uv before bootstrapping this host." >&2
  exit 2
}

source "$ROOT/scripts/autodl_env.sh"
if [[ -n "${OPD_OFFLINE_WHEELHOUSE:-}" ]]; then
  if [[ ! -d "$OPD_OFFLINE_WHEELHOUSE" ]]; then
    echo "Offline wheelhouse does not exist: $OPD_OFFLINE_WHEELHOUSE" >&2
    exit 2
  fi
  echo "Installing from offline wheelhouse: $OPD_OFFLINE_WHEELHOUSE"
  uv sync --frozen --offline --no-index --no-python-downloads --python 3.11 \
    --find-links "$OPD_OFFLINE_WHEELHOUSE" \
    --extra data --extra gpu --extra eval --extra tracking --extra dev
else
  uv sync --frozen --extra data --extra gpu --extra eval --extra tracking --extra dev
fi
uv run --no-sync opd doctor --config configs/main.yaml
nvidia-smi
"$ROOT/scripts/autodl_preflight.sh"
