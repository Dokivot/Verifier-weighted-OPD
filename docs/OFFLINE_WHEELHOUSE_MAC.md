# Mac Docker 离线 wheel 仓库

AutoDL 运行的是 Linux x86_64、Python 3.11 和 CUDA 12.8。不要直接在 macOS 上执行 `pip download`，否则可能下载 macOS 或 arm64 wheel。项目提供 `scripts/build_linux_wheelhouse.sh:1`，在 Docker 的 `linux/amd64` 容器中构建正确的 wheel。

## 1. Mac 构建

安装并启动 Docker Desktop，然后在项目根目录执行：

```bash
cd /path/to/OPDProj
git checkout rtx-pro-6000-blackwell
git pull --ff-only origin rtx-pro-6000-blackwell
scripts/build_linux_wheelhouse.sh "$PWD/opd-wheelhouse"
```

构建器使用 Python 3.11，并依据冻结的 `uv.lock` 下载 `data`、`gpu`、`eval`、`tracking` 和 `dev` extras。普通依赖默认从 PyPI 获取，CUDA 12.8 PyTorch 依赖从 PyTorch CUDA index 获取。

如果 PyPI 连接较慢，可以使用镜像：

```bash
PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
scripts/build_linux_wheelhouse.sh "$PWD/opd-wheelhouse"
```

不要把 PyTorch CUDA index 替换成普通 PyPI；`torch==2.7.1+cu128`、`torchaudio==2.7.1+cu128` 和 `torchvision==0.22.1+cu128` 必须来自 CUDA 12.8 index。

## 2. 本地检查

构建完成后应存在：

```text
opd-wheelhouse/wheels/*.whl
opd-wheelhouse/requirements-linux-cp311.txt
opd-wheelhouse/manifest.txt
```

检查：

```bash
cat opd-wheelhouse/manifest.txt
find opd-wheelhouse/wheels -maxdepth 1 -type f -name '*.whl' | wc -l
find opd-wheelhouse/wheels -maxdepth 1 -type f \\
  \( -name 'torch-2.7.1+cu128-*.whl' -o -name 'vllm-0.10.1.1-*.whl' \\
  -o -name 'aiohttp-3.14.3-*.whl' -o -name 'openai_harmony-0.0.8-*.whl' \)
```

`aiohttp` 必须是 `cp311` wheel，不能使用之前遇到的 `cp312` wheel。`manifest.txt` 中的 machine 应为 `x86_64`，不是 Mac 的 `arm64`。

## 3. 上传到 AutoDL

推荐上传到数据盘：

```text
/root/autodl-tmp/opd-wheelhouse/wheels
/root/autodl-tmp/opd-wheelhouse/requirements-linux-cp311.txt
/root/autodl-tmp/opd-wheelhouse/manifest.txt
```

也可以先打包：

```bash
tar -czf opd-wheelhouse-linux-cp311.tar.gz opd-wheelhouse
```

服务器解压后检查：

```bash
cd /root/autodl-tmp/OPDProj
find /root/autodl-tmp/opd-wheelhouse/wheels -maxdepth 1 -name '*.whl' | wc -l
cat /root/autodl-tmp/opd-wheelhouse/manifest.txt
```

## 4. 离线安装

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
export OPD_OFFLINE_WHEELHOUSE=/root/autodl-tmp/opd-wheelhouse/wheels
scripts/remote_bootstrap.sh 2>&1 | tee logs/00_bootstrap_offline.log
```

当 `OPD_OFFLINE_WHEELHOUSE` 存在时，`remote_bootstrap.sh` 会执行 `uv sync --frozen --offline --no-index`，只从 wheelhouse 安装，不访问 PyPI。

安装后检查：

```bash
make check 2>&1 | tee logs/01_make_check_offline.log
uv run --no-sync python -c 'import torch, vllm, lighteval, more_itertools; print(torch.__version__, torch.version.cuda)'
```

如果出现 `No matching distribution`，检查 Python 是否为 3.11、wheelhouse 是否为 `linux/amd64`，以及报错包是否完整上传；不要切换到 macOS wheel。

## 5. 继续一键实验

```bash
export OPD_OFFLINE_WHEELHOUSE=/root/autodl-tmp/opd-wheelhouse/wheels
tmux new -s sure-k2
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
scripts/run_sure_k2_oneclick.sh 2>&1 | tee logs/sure_k2_oneclick_terminal.log
```

一键脚本会复用已成功阶段；安装或后续阶段失败时，查看 `logs/sure_k2_oneclick/state/latest_failure/`，修复代码后重新执行同一命令即可。
