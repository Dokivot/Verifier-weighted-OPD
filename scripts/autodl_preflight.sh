#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "$(uname -s)" == "Linux" ]] || fail "AutoDL run requires Linux"
[[ "$(uname -m)" == "x86_64" ]] || fail "Expected x86_64, got $(uname -m)"
command -v uv >/dev/null || fail "uv is missing"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is missing"
gpu_inventory="$(nvidia-smi -L)"
[[ "$gpu_inventory" == *"GPU "* ]] || fail "No NVIDIA GPU is visible"

storage_root="${OPD_STORAGE_ROOT:-/root/autodl-tmp/opd-lab-storage}"
mkdir -p "$storage_root"
available_gb="$(df -Pk "$storage_root" | awk 'NR==2 {print int($4/1024/1024)}')"
[[ "$available_gb" -ge 70 ]] || fail "Need at least 70 GiB free under $storage_root; found ${available_gb} GiB"

uv run --no-sync python - <<'PY'
import importlib.metadata
import platform
import shutil
import sys

import torch

required = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "langdetect",
    "lighteval",
    "peft",
    "pyarrow",
    "torch",
    "transformers",
    "vllm",
    "xxhash",
}
expected_versions = {
    "lighteval": "0.9.2",
    "vllm": "0.10.1.1",
    "xxhash": "3.8.1",
}
print(f"python={sys.version.split()[0]} platform={platform.platform()}")
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("ERROR: use Python 3.11 or 3.12")
for name in sorted(required):
    try:
        version = importlib.metadata.version(name)
        print(f"{name}={version}")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"ERROR: missing package {name}") from None
    expected = expected_versions.get(name)
    if expected is not None and version != expected:
        raise SystemExit(f"ERROR: {name}=={expected} is required; found {version}")
if shutil.which("lighteval") is None:
    raise SystemExit("ERROR: missing executable: lighteval")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch cannot see CUDA")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: selected GPU does not support bfloat16")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    memory_gib = properties.total_memory / 1024**3
    print(
        f"gpu[{index}]={properties.name} memory={memory_gib:.1f}GiB "
        f"capability={properties.major}.{properties.minor}"
    )
    if memory_gib < 23:
        raise SystemExit("ERROR: tiny smoke requires at least a 24 GiB GPU")
print(f"torch_cuda={torch.version.cuda}")
PY

uv run --no-sync opd doctor --config configs/main.yaml
echo "AutoDL preflight passed. Use an 80 GiB GPU for the 14B BF16 teacher stage."
