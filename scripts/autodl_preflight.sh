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
import subprocess
import sys

import torch
from bitsandbytes import functional as bnb_functional

from opd.runtime_profile import validate_rtx_pro_6000_runtime

required = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "langdetect",
    "lighteval",
    "peft",
    "pyarrow",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "triton",
    "vllm",
    "xformers",
    "xxhash",
}
expected_versions = {
    "bitsandbytes": "0.50.2",
    "lighteval": "0.9.2",
    "torch": "2.7.1+cu128",
    "torchaudio": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "transformers": "4.57.6",
    "triton": "3.3.1",
    "vllm": "0.10.1.1",
    "xformers": "0.0.31",
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
if torch.cuda.device_count() != 1:
    raise SystemExit(f"ERROR: RTX PRO 6000 profile expects exactly one GPU; found {torch.cuda.device_count()}")
driver_version = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()[0]
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    memory_gib = properties.total_memory / 1024**3
    print(
        f"gpu[{index}]={properties.name} memory={memory_gib:.1f}GiB "
        f"capability={properties.major}.{properties.minor}"
    )
    try:
        validate_rtx_pro_6000_runtime(
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            driver_version=driver_version,
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            memory_gib=memory_gib,
            compiled_architectures=torch.cuda.get_arch_list(),
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
device = torch.device("cuda:0")
left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
matmul = left @ right
query = torch.randn((1, 8, 256, 64), device=device, dtype=torch.bfloat16)
attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
quantization_input = torch.randn((4096,), device=device, dtype=torch.bfloat16)
quantized, quantization_state = bnb_functional.quantize_4bit(
    quantization_input,
    quant_type="nf4",
)
dequantized = bnb_functional.dequantize_4bit(quantized, quantization_state)
torch.cuda.synchronize()
kernel_outputs = (matmul, attention, dequantized)
if not all(bool(torch.isfinite(output).all().item()) for output in kernel_outputs):
    raise SystemExit("ERROR: Blackwell runtime kernel smoke produced non-finite values")
try:
    import vllm._C  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ERROR: vLLM CUDA extension failed to load: {exc}") from None
print(f"torch_cuda={torch.version.cuda}")
print(f"driver={driver_version}")
print(f"compiled_architectures={torch.cuda.get_arch_list()}")
print("blackwell_bf16_kernel_smoke=passed")
print("blackwell_bitsandbytes_nf4_smoke=passed")
PY

uv run --no-sync opd doctor --config configs/main.yaml
echo "AutoDL preflight passed for one RTX PRO 6000 Blackwell 96GB GPU with CUDA 12.8."
