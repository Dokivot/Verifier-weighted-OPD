#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?usage: scripts/train.sh CONFIG}"
uv run --no-sync opd budget --config "$CONFIG"
uv run --no-sync accelerate launch --num_processes 1 -m opd train --config "$CONFIG"
uv run --no-sync opd budget --config "$CONFIG"
