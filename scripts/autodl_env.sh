#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this file with: source scripts/autodl_env.sh" >&2
  exit 2
fi

export OPD_STORAGE_ROOT="${OPD_STORAGE_ROOT:-/root/autodl-tmp/opd-lab-storage}"
mkdir -p "$OPD_STORAGE_ROOT"/{hf,uv,tmp}
export HF_HOME="${HF_HOME:-$OPD_STORAGE_ROOT/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OPD_STORAGE_ROOT/uv}"
export TMPDIR="${TMPDIR:-$OPD_STORAGE_ROOT/tmp}"
export TOKENIZERS_PARALLELISM=false

echo "OPD_STORAGE_ROOT=$OPD_STORAGE_ROOT"
echo "HF_HOME=$HF_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
