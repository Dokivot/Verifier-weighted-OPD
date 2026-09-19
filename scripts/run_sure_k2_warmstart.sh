#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOKE_CONFIG=configs/sure_k2_warmstart_smoke.yaml \
PILOT_CONFIG=configs/sure_k2_warmstart_pilot.yaml \
exec scripts/run_sure_k2_oneclick.sh \
  --config configs/sure_k2_warmstart_24h.yaml \
  "$@"
