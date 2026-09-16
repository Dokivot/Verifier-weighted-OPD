#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/qwen_gpu_smoke.yaml}"

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi is required for the Qwen GPU smoke" >&2
  exit 2
}
GPU_MEMORY_MIB="$(
  nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ ! "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || [[ "$GPU_MEMORY_MIB" -lt 40000 ]]; then
  echo "ERROR: Qwen3-14B BF16 smoke requires a 40GB-class GPU; found ${GPU_MEMORY_MIB:-unknown} MiB" >&2
  echo "Run make tiny-gpu-smoke on a 24GB card instead." >&2
  exit 2
fi

scripts/prepare_data.sh "$CONFIG"
scripts/fetch_eval_data.sh "$CONFIG"
scripts/audit_contamination.sh "$CONFIG"
uv run --no-sync opd audit sparse-kl \
  --output artifacts/qwen_gpu_smoke/audits/sparse_kl.json \
  --config "$CONFIG"
scripts/generate_rollouts.sh "$CONFIG" 0
scripts/verify.sh "$CONFIG" 0
scripts/annotate_teacher.sh "$CONFIG" 0
uv run --no-sync opd data build-view --round 0 --method weighted-opd --config "$CONFIG"
scripts/train.sh "$CONFIG"
scripts/merge_checkpoint.sh \
  "$CONFIG" \
  artifacts/qwen_gpu_smoke/checkpoints/weighted_opd_seed42/best \
  artifacts/qwen_gpu_smoke/merged/weighted_opd
uv run --no-sync opd evaluate \
  --suite smoke \
  --checkpoint artifacts/qwen_gpu_smoke/merged/weighted_opd \
  --config "$CONFIG"
scripts/evaluate_lighteval.sh \
  artifacts/qwen_gpu_smoke/merged/weighted_opd \
  artifacts/qwen_gpu_smoke/lighteval/weighted_opd \
  "$CONFIG"

echo "Qwen GPU smoke passed. Inspect artifacts/qwen_gpu_smoke before starting 15k rollout."
