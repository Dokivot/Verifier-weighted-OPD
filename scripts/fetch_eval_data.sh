#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/main.yaml}"
for NAME in math500 aime_2024 aime_2025; do
  uv run --no-sync opd data fetch-eval --name "$NAME" --config "$CONFIG"
done
