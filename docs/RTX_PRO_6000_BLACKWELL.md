# RTX PRO 6000 Blackwell 96GB 运行说明

本分支专门面向单张 NVIDIA RTX PRO 6000 Blackwell 96GB。它与 `main` 的实验配置、模型、数据、
seed 和评测方法保持一致，只替换 GPU 软件栈并增加 Blackwell 运行门禁，便于把实验结果与 A800
版本进行质量比较。

## 固定软件栈

- Linux x86_64；
- Python 3.11 或 3.12；
- NVIDIA driver `>=570.26`；
- PyTorch `2.7.1+cu128`；
- torchvision `0.22.1+cu128`；
- torchaudio `2.7.1+cu128`；
- Triton `3.3.1`、xFormers `0.0.31`、bitsandbytes `0.50.2`；
- vLLM `0.10.1.1`，其官方 wheel 使用 CUDA 12.8 编译；
- Transformers `4.57.6`；
- LightEval `0.9.2`。

不要手工把 Torch 升到 2.8/2.9，也不要改成 CUDA 12.6。vLLM 包含预编译 CUDA 扩展，Torch、
CUDA 和 vLLM wheel 必须匹配。基础镜像推荐 PyTorch 2.8.0 + CUDA 12.8；项目实际运行环境由
`.venv` 和 `uv.lock` 决定，镜像内置 Torch 不参与实验。

## 服务器首次安装

```bash
cd /root/autodl-tmp
git clone -b rtx-pro-6000-blackwell \
  https://github.com/Dokivot/Verifier-weighted-OPD.git OPDProj
cd OPDProj

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
source scripts/autodl_env.sh
set -o pipefail

scripts/remote_bootstrap.sh 2>&1 | tee bootstrap.log
```

`remote_bootstrap.sh` 会拒绝以下环境：

- 不是单张 RTX PRO 6000 Blackwell；
- 可见显存不足 90GiB；
- compute capability 低于 12.0；
- NVIDIA driver 低于 570.26；
- Torch 不是 `2.7.1+cu128` 或 `torch.version.cuda` 不是 `12.8`；
- Torch wheel 不包含 `sm_120`/`compute_120`；
- vLLM CUDA 扩展无法加载；
- BF16 matmul、scaled-dot-product attention 或 bitsandbytes NF4 kernel 运行失败。

成功输出必须包含：

```text
blackwell_bf16_kernel_smoke=passed
blackwell_bitsandbytes_nf4_smoke=passed
AutoDL preflight passed for one RTX PRO 6000 Blackwell 96GB GPU with CUDA 12.8.
```

## 分级验证

依次执行，任一步失败都不要启动正式 MVP：

```bash
mkdir -p logs
make check 2>&1 | tee logs/00_make_check.log
make smoke 2>&1 | tee logs/01_cpu_smoke.log
make tiny-gpu-smoke 2>&1 | tee logs/02_tiny_gpu_smoke.log
scripts/qwen_gpu_smoke.sh 2>&1 | tee logs/03_qwen_gpu_smoke.log
```

Qwen smoke 完整通过后执行：

```bash
scripts/run_resume_mvp.sh
```

## 必须保留的环境证据

```bash
mkdir -p reports/environment
git rev-parse HEAD | tee reports/environment/git_commit.txt
git branch --show-current | tee reports/environment/git_branch.txt
nvidia-smi | tee reports/environment/nvidia_smi.txt
uv pip freeze | tee reports/environment/python_packages.txt

uv run --no-sync python - <<'PY' | tee reports/environment/blackwell_runtime.txt
import torch
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("compiled architectures:", torch.cuda.get_arch_list())
print("memory GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

## 结果比较边界

RTX PRO 6000 与 A800 可以比较模型质量，因为模型 revision、数据、seed、采样参数和 benchmark
保持不变；不要把两种 GPU 的 wall time 或 GPU hours 直接当作同硬件性能对比。不同架构上的并行
kernel 可能产生细小浮点差异，单 seed 也不代表跨 seed 稳定性。
