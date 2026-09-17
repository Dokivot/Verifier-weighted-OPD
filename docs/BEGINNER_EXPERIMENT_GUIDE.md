# OPD-Lab 小白实验运行指南

本文只描述当前推荐的首轮简历 MVP，不把历史 Qwen3 方案或 7,500 prompts 的 Phase B/C
扩展实验混入主流程。目标是让第一次使用 Linux、AutoDL 和远端 GPU 的读者也能得到一套可复查、
可继续训练、可用于面试说明的真实实验结果。

## 1. 先知道最终要得到什么

首轮实验固定比较三个模型：

| 模型 | 用途 |
|---|---|
| `Qwen2.5-1.5B-Instruct` Base | 训练前基线 |
| VFS-Weighted OPD B50 | 项目主方法 |
| SFT | 判断提升是否只是普通监督微调带来的 |

主方法使用：

- Student：`Qwen/Qwen2.5-1.5B-Instruct`；
- Teacher：`Qwen/Qwen2.5-Math-7B-Instruct`；
- 数据：`open-r1/OpenR1-Math-220k`；
- 候选题：6,000 条；
- 正式 rollout：前 3,000 道题，每题 2 个候选；
- Teacher：只标注 VFS 在 50% Teacher-token 预算中选中的 states；
- Sparse KL：保存 Teacher top-64 log-prob、tail mass 和 entropy；
- 训练：4-bit QLoRA，最多 300 optimizer steps；
- seed：只使用 `42`；
- 正式 benchmark：MATH-500、AIME 2024、IFEval。

长度策略固定为：

- vLLM rollout 上下文上限：8,192 tokens；
- 单个 response 最多生成：4,096 tokens；
- Teacher annotation 和训练总长度：4,096 tokens。

因此 vLLM 有空间同时容纳 prompt 和 response；Teacher/训练只使用 prompt 加 response 的前 4,096
tokens，以控制显存和训练成本。不要只改其中一个长度。若以后要训练完整 8k trajectory，需要重新做
显存 smoke、成本评估和配置登记。

首轮产物统一写入 `artifacts/resume_mvp/`。历史大规模方案在 `plans/`，首轮不要运行。

## 2. 什么已经验证，什么必须在 AutoDL 验证

本地可以证明：

- 单元测试、Ruff、mypy strict 和锁文件检查通过；
- 所有 shell 脚本语法正确；
- 所有 YAML 能加载；
- mock 端到端 pipeline 能产生可晋级结果。

没有 NVIDIA GPU 的电脑不能证明：

- CUDA、bitsandbytes、vLLM 是否能在租用实例加载；
- 真实 QLoRA、adapter merge 是否成功；
- 7B Teacher annotation 是否会 OOM；
- LightEval 与当前驱动是否正常工作。

所以 tiny GPU smoke 和 Qwen 正式模型 smoke 是强制门禁，不能跳过。

## 3. 选择 AutoDL 实例和镜像

推荐配置：

- 单张 H100 80GB 或 A800 80GB；
- x86_64 Linux；
- 数据盘建议 200GB，最低 100GB；
- Python 3.11 或 3.12；
- 网络可以访问 Git 仓库和 Hugging Face。

如果 AutoDL 只有“PyTorch 2.8.0”和“PyTorch 2.12.1”镜像，选择 **PyTorch 2.8.0 + CUDA
12.8**。原因不是项目直接使用镜像中的 PyTorch，而是这个镜像通常提供更成熟的 Python/CUDA
基础环境。`scripts/remote_bootstrap.sh` 会在项目 `.venv` 中按 `uv.lock` 安装实际依赖，目前 Linux
锁定结果是：

- PyTorch `2.7.1`；
- CUDA 12.6 runtime wheels；
- vLLM `0.10.1.1`；
- LightEval `0.9.2`；
- Transformers `4.57.6`。

因此看到镜像标题写 2.8.0，而虚拟环境中显示 PyTorch 2.7.1，是正常现象。NVIDIA 驱动只需能
向后兼容 CUDA 12.6。最终以 `autodl_preflight.sh` 的检测结果为准，不以镜像名字为准。

显存建议：

- 24GB：只用于 tiny/Qwen smoke，不建议跑完整 MVP；
- 40GB/48GB：可能运行，但 annotation 较慢且余量较小；
- 80GB：正式 MVP 推荐。

磁盘建议：

- 200GB：最省心；
- 100GB：可以尝试，但 bootstrap 完成后必须至少剩余 70GiB，并及时删除可重建的 merged 模型；
- 永远保留至少 15GiB 空闲空间。

## 4. 在本地把源码提交到远端仓库

服务器只能拉取已经 commit 并 push 的文件。先在本地执行：

```bash
cd /Users/dokivot/PyProj/OPDProj
git status --short
git diff --check
```

确认没有 token、密码、`.env`、模型权重或实验数据后执行：

```bash
git add -A
git diff --cached --stat
git commit -m "Harden AutoDL MVP pipeline"
git push origin main
git rev-parse HEAD
```

保存最后输出的 commit SHA。若分支不是 `main`，把 push 命令中的分支名换成实际分支。

以下内容由 `.gitignore` 排除，不需要提交：`artifacts/`、`checkpoints/`、`logs/`、`backups/`、
`bootstrap.log`、模型权重和环境快照。

## 5. 登录服务器并使用 tmux

AutoDL 控制台会给出类似命令：

```bash
ssh -p PORT root@HOST
```

登录后立即创建 tmux 会话：

```bash
tmux new -s opd
```

常用操作：

- 暂时离开但不中止任务：按 `Ctrl+B`，松开后按 `D`；
- 重新进入：`tmux attach -t opd`；
- 查看会话：`tmux ls`。

所有下载、rollout、训练和评测都应在 tmux 中运行。

## 6. 克隆代码到数据盘

```bash
cd /root/autodl-tmp
git clone YOUR_REPOSITORY_URL OPDProj
cd OPDProj
git rev-parse HEAD
git status --short
```

检查：

1. SHA 与本地保存的 SHA 相同；
2. `git status --short` 没有输出。

私有仓库优先使用 SSH key。不要把访问 token 写入脚本、README 或命令历史。

## 7. 安装 uv，并把缓存放到数据盘

安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

加载项目环境变量：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail
```

你应该看到缓存位于：

```text
/root/autodl-tmp/opd-lab-storage
```

每次重新 SSH 登录或新开 tmux 后，都要重新执行：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail
```

## 8. 安装锁定依赖并运行 AutoDL 门禁

```bash
scripts/remote_bootstrap.sh 2>&1 | tee bootstrap.log
```

该脚本会：

- 创建项目 `.venv` 并严格按 `uv.lock` 安装依赖；
- 检查 Linux x86_64、Python 3.11/3.12；
- 检查 NVIDIA GPU、BF16、CUDA 可见性；
- 检查 vLLM、LightEval、xxhash 等兼容版本；
- 检查数据盘可用空间；
- 运行项目 doctor。

成功标志是最后出现：

```text
AutoDL preflight passed.
```

如果失败，不要继续。先保留 `bootstrap.log`，再根据最后一个 `ERROR` 修复。

确认虚拟环境实际版本：

```bash
uv run --no-sync python - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

## 9. 登录 Hugging Face

在 Hugging Face 设置页创建 read token，然后执行：

```bash
uv run --no-sync hf auth login
```

出现 `Login successful` 即可继续。`Cannot authenticate through git-credential` 只表示没有配置 Git
凭据助手，不影响 Hugging Face 模型和数据下载。

## 10. 建立实验记录

```bash
mkdir -p logs backups reports/environment
date -Iseconds | tee reports/environment/start_time.txt
git rev-parse HEAD | tee reports/environment/git_commit.txt
nvidia-smi | tee reports/environment/nvidia_smi.txt
df -h /root/autodl-tmp | tee reports/environment/disk_before.txt
uv pip freeze | tee reports/environment/python_packages.txt
```

这些文件默认不提交 Git，但必须在关机前打包下载。

## 11. 第一关：CPU 工程门禁

```bash
make check 2>&1 | tee logs/00_make_check.log
make smoke 2>&1 | tee logs/01_cpu_smoke.log
```

成功标准：

- 所有单元测试通过；
- Ruff check 和 format check 通过；
- mypy strict 通过；
- `uv lock --check` 通过；
- mock pipeline 最终显示 `promotable: true`。

Mock accuracy 不是模型效果，只证明工程链路能走通。

## 12. 第二关：tiny GPU smoke

```bash
make tiny-gpu-smoke 2>&1 | tee logs/02_tiny_gpu_smoke.log
```

它会真实运行小模型下载、rollout、Teacher top-k annotation、一个 QLoRA step、adapter merge 和
评测。检查：

```bash
test -f artifacts/tiny_gpu_smoke/checkpoints/weighted_opd_seed42/training_summary.json
test -f artifacts/tiny_gpu_smoke/merged/weighted_opd/manifest.json
test -f artifacts/tiny_gpu_smoke/evaluation/weighted_opd/smoke/summary.json
echo "tiny GPU smoke passed"
```

只有三条 `test` 都没有输出错误，最后才会打印通过信息。

## 13. 第三关：正式模型 smoke

```bash
scripts/qwen_gpu_smoke.sh 2>&1 | tee logs/03_qwen_gpu_smoke.log
```

该步骤使用正式 1.5B Student 和 7B Teacher，但只处理 32 条训练数据、训练 2 steps、评测 2 条
MATH-500。成功标准：

```bash
cat artifacts/qwen_gpu_smoke/checkpoints/weighted_opd_seed42/training_summary.json
find artifacts/qwen_gpu_smoke/lighteval/weighted_opd -type f | sort
df -h /root/autodl-tmp
```

必须确认：

- rollout 和 Teacher annotation 没有批量失败；
- loss、gradient norm 都不是 `NaN`/`Inf`；
- `best/` adapter 成功 merge；
- merged 模型能重新加载；
- LightEval 生成 `results/`、`details/`、`command.json` 和 `manifest.json`；
- 数据盘仍至少有 15GiB 空闲。

## 14. 正式首轮：运行 VFS-Weighted MVP

前三关全部通过后执行：

```bash
scripts/run_resume_mvp.sh
```

脚本会自动创建 `logs/resume_mvp/时间戳/`，依次运行：

| Stage | 工作 |
|---:|---|
| 0 | 下载、清洗并切分训练数据 |
| 1 | 下载固定 revision 的评测数据 |
| 2 | 污染审计 |
| 10 | 3,000 prompts × 2 Student rollout |
| 11 | 数学 verifier |
| 12 | VFS 状态选择 |
| 13 | 7B Teacher annotation |
| 14 | 构建 weighted OPD training view |
| 20 | QLoRA 训练 |
| 21 | 合并 best adapter |
| 22 | 内部 regression |
| 23 | 主方法 MATH-500/AIME 2024/IFEval |
| 24 | Base MATH-500/AIME 2024/IFEval |

不要同时启动第二个相同脚本，也不要并行运行 SFT。

## 15. 运行期间如何监控

新开一个 tmux 窗口，执行：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
watch -n 30 nvidia-smi
```

rollout 阶段监控：

```bash
watch -n 30 'cat artifacts/resume_mvp/data/rollouts/round_0/quality_gate.json'
```

状态含义：

- `collecting`：成功记录还不足 800；
- `passed`：截断率不高于 20%，继续生成；
- `failed`：截断率超过 20%，脚本自动停止，不要进入 Teacher 阶段。

查看当前日志：

```bash
find logs/resume_mvp -type f -name '*.log' -print | sort
tail -f logs/resume_mvp/实际时间戳/当前阶段.log
```

监控磁盘：

```bash
df -h /root/autodl-tmp
du -h -d 3 artifacts/resume_mvp "$HF_HOME" | sort -h | tail -30
```

GPU 暂时为 0% 不一定是故障，下载、校验、加载权重和写文件时可能只使用 CPU。若日志 10 分钟
没有变化，再用下列命令检查进程：

```bash
pgrep -af 'opd|vllm|lighteval|accelerate'
```

## 16. 失败后如何继续

先保留失败日志，不要直接删除整个 `artifacts/resume_mvp/`。修复原因后，从失败 stage 重新开始：

```bash
START_STAGE=13 scripts/run_resume_mvp.sh
```

允许的 stage 是：

```text
0 1 2 10 11 12 13 14 20 21 22 23 24
```

Rollout 和 Teacher annotation 使用配置 hash 分片；使用完全相同的配置重跑时，完整 shard 会复用。

若 stage 20 训练中断，并且以下目录存在：

```bash
test -f artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/latest/checkpoint_state.json
```

使用：

```bash
START_STAGE=20 \
RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/latest \
scripts/run_resume_mvp.sh
```

不要只写 `START_STAGE=20` 后声称“恢复训练”；没有 `RESUME_FROM_CHECKPOINT` 时会从头初始化训练。

如果 stage 21 之后失败，best adapter 已存在，不需要重训。例如只重跑 benchmark：

```bash
START_STAGE=23 scripts/run_resume_mvp.sh
```

脚本会拒绝不存在的 stage 编号，避免拼写错误后什么也没运行却返回成功。

## 17. 检查主方法是否真正完成

```bash
cat artifacts/resume_mvp/data/rollouts/round_0/quality_gate.json
cat artifacts/resume_mvp/data/annotation_selection/vfs_weighted_b50/round_0/selection_report.json
cat artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/training_summary.json
find artifacts/resume_mvp/lighteval/vfs_weighted_b50 -type f | sort
find artifacts/resume_mvp/lighteval/base -type f | sort
```

必须满足：

- rollout quality gate 是 `passed`；
- selection 未超过 Teacher-token budget；
- Teacher 成功记录数与 selected records 一致；
- `global_step > 0`，loss/gradient norm 有限；
- `best_step > 0` 且存在 `best/`；
- Base 和主方法都生成 LightEval results/details；
- 每个阶段存在 manifest 和 `job_metrics.json`。

训练现在会在 epoch 提前结束且未到 `eval_steps` 时补做一次最终 validation，因此小数据集也会生成
`best/`，不会在 merge 阶段才发现缺失。

## 18. 运行 SFT 基线

主方法和 Base benchmark 完整后再执行：

```bash
scripts/run_resume_sft.sh
```

SFT 会复用主流程生成的清洗训练集，依次执行 stage `30/31/32/33`：训练、merge、内部 regression、
正式 benchmark。若训练中断：

```bash
START_STAGE=30 \
RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/sft_seed42/latest \
scripts/run_resume_sft.sh
```

若只需重跑 SFT benchmark：

```bash
START_STAGE=33 scripts/run_resume_sft.sh
```

## 19. 最终如何比较结果

正式分数以 LightEval 原生 `results_*.json` 为准，逐题分析使用 `details_*.parquet`。最终表至少包含：

| 方法 | MATH-500 | AIME 2024 | IFEval | Teacher tokens | GPU hours |
|---|---:|---:|---:|---:|---:|
| Base | 实测 | 实测 | 实测 | 0 | 实测 |
| SFT | 实测 | 实测 | 实测 | 0 | 实测 |
| VFS-Weighted B50 | 实测 | 实测 | 实测 | 实测 | 实测 |

第一版只能报告：

- 相对 Base 和 SFT 的实际变化；
- 主方法使用的 Teacher tokens 与 GPU hours；
- 截断率、验证状态分布和失败案例；
- 单 seed 42 下的观察结果。

在 Random-B50 和 Dense-B100 尚未完成前，不得声称“VFS 比随机选择更好”；在组件消融完成前，
不得声称提升一定来自 entropy weighting 或 verifier weighting。paired bootstrap 只能表示 benchmark
题目抽样不确定性，不能代替多 seed 训练。

## 20. 磁盘不足时怎么处理

先检查，不要盲删：

```bash
df -h /root/autodl-tmp
du -h -d 3 artifacts/resume_mvp "$HF_HOME" .venv | sort -h | tail -40
```

优先保留：

- `checkpoints/*/best/`；
- `training_summary.json`；
- manifests 和 `job_metrics.json`；
- LightEval results/details/command；
- 配置、Git SHA 和日志。

完成评测并备份后，可删除可重建的 merged 模型：

```bash
rm -rf artifacts/resume_mvp/merged/vfs_weighted_b50
rm -rf artifacts/resume_mvp/merged/sft
```

删除前必须再次执行 `pwd` 和 `ls -lh` 确认路径。不要删除 best adapter、未完成阶段依赖的 shard，
也不要在仍需加载模型时清空 `HF_HOME`。

uv 下载缓存可在环境已经安装完成且磁盘紧张时清理：

```bash
uv cache clean
```

这不会删除 `.venv`，但后续重新安装依赖时需要再次下载。

## 21. 备份面试证据

记录结束环境：

```bash
date -Iseconds | tee reports/environment/end_time.txt
nvidia-smi | tee reports/environment/nvidia_smi_final.txt
df -h /root/autodl-tmp | tee reports/environment/disk_after.txt
uv run --no-sync opd budget --config configs/vfs_weighted_mvp.yaml \
  | tee reports/environment/final_budget.json
```

打包元数据、日志和评测：

```bash
tar -czf backups/opd_mvp_evidence_$(date +%Y%m%d_%H%M).tar.gz \
  configs docs/RESUME_MVP.md README.md PROJECT_PLAN.md MODEL_CARD.md DATA_CARD.md \
  uv.lock logs reports/environment \
  artifacts/resume_mvp/data/annotation_selection \
  artifacts/resume_mvp/checkpoints/*/training_summary.json \
  artifacts/resume_mvp/lighteval
```

分别备份 best adapter，排除可重建的 optimizer state：

```bash
tar --exclude='accelerate_state' -czf backups/vfs_best_seed42.tar.gz \
  artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/best

tar --exclude='accelerate_state' -czf backups/sft_best_seed42.tar.gz \
  artifacts/resume_mvp/checkpoints/sft_seed42/best
```

在本地电脑下载：

```bash
scp -P PORT -r root@HOST:/root/autodl-tmp/OPDProj/backups ./opd-backups
```

下载后验证压缩包：

```bash
ls -lh opd-backups
tar -tzf opd-backups/vfs_best_seed42.tar.gz | head
```

确认本地至少有两份可解压备份后，才能释放 AutoDL 实例。

## 22. 常见问题

### `uv: command not found`

```bash
source "$HOME/.local/bin/env"
uv --version
```

### `opd: command not found`

不要运行裸 `opd`，使用 `uv run --no-sync opd ...`，并确认已经执行 bootstrap。

### `torch cannot see CUDA`

停止实验，保存 `bootstrap.log` 和 `nvidia-smi`。通常是实例、驱动或镜像不兼容。不要用重新安装
随机版本 PyTorch 的方式覆盖锁文件。

### OOM

保存日志和 `nvidia-smi`，确认使用的是 80GB 正式实例。不要未经记录就修改 batch、量化、长度或
模型；这些修改会改变实验协议。若必须修改，创建新配置并保留原配置与失败证据。

### Hugging Face 401/403

```bash
uv run --no-sync hf auth whoami
uv run --no-sync hf auth login
```

### SSH 断开

重新登录后执行 `tmux attach -t opd`。若进程已退出，检查日志，再按 stage 恢复。

### `git pull` 前看到日志文件

最新版 `.gitignore` 已忽略 `logs/`、`backups/`、`bootstrap.log` 和环境快照。源码本身有修改时，
先用 `git diff` 判断是否是你有意改过的配置；不要直接 `git reset --hard`。

## 23. 最短检查清单

```text
[ ] 本地源码已 commit/push，并保存 commit SHA
[ ] AutoDL 选择 PyTorch 2.8.0 + CUDA 12.8 基础镜像
[ ] 进入 tmux，clone 到 /root/autodl-tmp/OPDProj
[ ] source scripts/autodl_env.sh
[ ] remote_bootstrap 最后显示 AutoDL preflight passed
[ ] Hugging Face 登录成功
[ ] make check 通过
[ ] make smoke 通过
[ ] tiny GPU smoke 通过
[ ] Qwen 正式模型 smoke 通过
[ ] run_resume_mvp 完成 stages 0—24
[ ] rollout quality gate passed
[ ] Base 与主方法 LightEval results/details 存在
[ ] run_resume_sft 完成 stages 30—33
[ ] 三个模型的正式分数和成本已整理
[ ] best adapters、日志、配置、SHA、manifests、results/details 已下载
[ ] 本地压缩包可解压后再释放服务器
```

第一次操作时一次只执行一个阶段。遇到错误先保存日志并定位最后一个明确报错，不要连续粘贴多条
“可能修复”的命令。
