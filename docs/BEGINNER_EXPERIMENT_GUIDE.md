# OPD-Lab 小白实验运行指南

这是一份从零开始的操作手册。目标不是“让命令开始运行”，而是让你最终得到一套可以复查、
可以写进简历、可以在面试时解释的 OPD 实验结果。

> 重要结论：本项目的 CPU 工程链路已经在本地验证，但真实 QLoRA、Qwen3-14B Teacher、vLLM
> 和 LightEval 必须在 NVIDIA GPU 服务器上完成两级 smoke test 后，才能开始正式 15k 实验。

---

## 1. 先理解你要完成什么

项目使用：

- Student：`Qwen/Qwen3-8B`；
- Teacher：`Qwen/Qwen3-14B`；
- 训练数据：`open-r1/OpenR1-Math-220k`；
- 训练 seed：只使用 `42`；
- 训练方式：4-bit QLoRA；
- 核心方法：Verifier + Teacher confidence 加权的 On-Policy Distillation；
- 正式评测：MATH-500、AIME 2024、GPQA Diamond、IFEval；
- 主要对照：Base、SFT、Vanilla OPD、Verifier OPD、Confidence OPD、Weighted OPD。

完整流程是：

```text
本地提交代码
  → AutoDL 环境检查
  → CPU mock smoke
  → tiny GPU smoke
  → Qwen3 正式模型 smoke
  → Round 0 数据与 rollout
  → Teacher annotation
  → 五种方法训练
  → 内部回归评测
  → LightEval 正式 benchmark
  → 可选 Round 1
  → 备份结果并关机
```

不要跳过 smoke test。它们的作用是用很少的 GPU 时间发现环境、显存、依赖和路径问题。

---

## 2. 你需要准备什么

### 2.1 账号与工具

准备：

1. 一个 GitHub、Gitee 或其他可从服务器克隆的 Git 仓库；
2. 一个 AutoDL 账号；
3. 一个 Hugging Face 账号；
4. 本地电脑上的终端；
5. 用于备份结果的本地磁盘或对象存储。

GPQA Diamond 可能要求你在 Hugging Face 网页接受数据集条款。没有授权时，其他 benchmark
仍可以运行，但必须把 GPQA 记录为“未运行”，不能用其他数据替代。

### 2.2 推荐服务器

如果希望从 smoke 一直跑到正式实验，最省事的选择是：

- 单张 H100 80GB；
- Ubuntu x86_64；
- 支持 CUDA 12.8 的驱动和镜像；
- Python 3.11 或 3.12；
- 100GB 数据盘；
- 稳定网络。

显存规则：

- 24GB：只能可靠执行 Qwen2.5 tiny smoke；
- 40/48GB：可以尝试 32 条 Qwen3 smoke，但不建议直接跑正式 14B Teacher；
- 80GB：正式实验推荐配置。

项目预算目标是约 `90 H100 GPU 小时`，硬上限是 `110 H100 GPU 小时`。这不是运行时长保证，
实际时间取决于 GPU、网络、输出长度和有效样本数。

### 2.3 100GB 数据盘规则

100GB 可以完成最小可信版本，但必须遵守：

- 仓库和缓存都放在 `/root/autodl-tmp` 数据盘；
- 不把模型缓存放到系统盘；
- 一次只保留一个 merged 8B 模型；
- adapter、manifest、训练摘要和评测结果优先保留；
- 始终保留至少 15GiB 空闲空间；
- 删除任何文件前先备份。

---

## 3. 本地电脑：先把代码提交到 Git

当前仓库的文件如果还是未跟踪状态，服务器将无法通过 `git clone` 获得它们。

在本地终端进入项目：

```bash
cd /Users/dokivot/PyProj/OPDProj
git status --short
```

如果看到很多以 `??` 开头的文件，这是“尚未被 Git 跟踪”，不是错误。执行：

```bash
git add -A
git commit -m "Initial engineering-grade OPD pipeline"
```

如果还没有远端仓库：

1. 在 GitHub/Gitee 网页创建一个空仓库；
2. 不要在网页端自动添加 README；
3. 复制仓库的 HTTPS 或 SSH 地址；
4. 执行下面的命令，把地址替换成你自己的。

```bash
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

如果 `git remote add origin` 提示 origin 已存在，先检查：

```bash
git remote -v
```

然后直接运行：

```bash
git push -u origin main
```

最终确认：

```bash
git status --short
git rev-parse HEAD
```

第一条命令应没有输出；第二条会输出一串 commit SHA。把 SHA 保存到实验记录中。

不要把 Hugging Face token、GitHub token、密码或 `.env` 文件提交到 Git。

---

## 4. AutoDL：创建实例

AutoDL 网页界面可能更新，但选择原则不变：

1. 创建 GPU 实例；
2. 优先选择单张 H100 80GB；
3. 选择较新的 Ubuntu/PyTorch/CUDA 12.8 镜像；
4. 数据盘选择 100GB；
5. 启动实例；
6. 从控制台复制 SSH 地址、端口和密码。

不要只看镜像名称判断环境是否正确。后面的 `autodl_preflight.sh` 才是最终门禁。

### 4.1 登录服务器

AutoDL 通常会给出类似命令：

```bash
ssh -p PORT root@HOST
```

把 `PORT` 和 `HOST` 换成控制台显示的值。第一次连接时输入 `yes`，然后输入密码。

终端中的密码不会显示星号，这是正常现象；输入完成后按回车。

### 4.2 使用 tmux 防止 SSH 断线中止任务

登录后先运行：

```bash
tmux new -s opd
```

常用操作：

- 暂时离开但不停止任务：依次按 `Ctrl+B`，松开后按 `D`；
- 重新进入：`tmux attach -t opd`；
- 查看会话：`tmux ls`。

所有长时间命令都应在 tmux 内执行。

---

## 5. 克隆项目到数据盘

在服务器执行：

```bash
cd /root/autodl-tmp
git clone YOUR_REPOSITORY_URL OPDProj
cd OPDProj
git rev-parse HEAD
git status --short
```

检查两件事：

1. commit SHA 与本地保存的 SHA 相同；
2. `git status --short` 没有输出。

如果是私有仓库，使用 GitHub deploy key、SSH key 或临时 personal access token。不要把 token
写入脚本、README 或 shell history；更推荐 SSH key。

---

## 6. 初始化环境

### 6.1 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

最后一条命令应显示 uv 版本。

### 6.2 把所有缓存放到数据盘

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail
```

你应该看到：

```text
OPD_STORAGE_ROOT=/root/autodl-tmp/opd-lab-storage
HF_HOME=/root/autodl-tmp/opd-lab-storage/hf
UV_CACHE_DIR=/root/autodl-tmp/opd-lab-storage/uv
TMPDIR=/root/autodl-tmp/opd-lab-storage/tmp
```

每次重新 SSH 登录或新开 tmux，都要重新执行：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
```

### 6.3 安装锁定依赖并运行 preflight

```bash
scripts/remote_bootstrap.sh 2>&1 | tee bootstrap.log
```

它会：

- 按 `uv.lock` 安装依赖；
- 检查 Linux x86_64；
- 检查 NVIDIA GPU、CUDA 和 BF16；
- 检查 Python 3.11/3.12；
- 检查 vLLM、Transformers、Accelerate、bitsandbytes、LightEval 等依赖；
- 检查 LightEval 0.9.2 与 vLLM 0.10.1.1 的兼容版本组合；
- 检查数据盘至少还有 70GiB；
- 检查 GPU 至少达到 tiny smoke 的 24GB 要求。

成功时最后会看到：

```text
AutoDL preflight passed.
```

如果这里失败，不要继续。根据错误修复实例、镜像、驱动、Python 或磁盘问题。

### 6.4 登录 Hugging Face

先在 Hugging Face 设置页创建只读 token，然后在服务器执行：

```bash
uv run --no-sync hf auth login
```

按提示粘贴 token。不要把 token 写进项目文件。

如果只运行公开模型和公开数据，很多步骤可能不要求登录；但 GPQA 授权通常需要账号。

---

## 7. 建立实验日志目录

```bash
mkdir -p logs backups reports/environment
set -o pipefail
```

记录环境：

```bash
date -Iseconds | tee reports/environment/start_time.txt
git rev-parse HEAD | tee reports/environment/git_commit.txt
nvidia-smi | tee reports/environment/nvidia_smi.txt
df -h | tee reports/environment/disk_before.txt
uv pip freeze | tee reports/environment/python_packages.txt
```

以后使用 `命令 2>&1 | tee logs/名称.log`，既能在屏幕看输出，也能保留日志。

---

## 8. 第一关：CPU 工程门禁

运行：

```bash
make check 2>&1 | tee logs/00_make_check.log
make smoke 2>&1 | tee logs/01_cpu_smoke.log
```

成功标准：

- 32 个单元测试全部通过；
- Ruff 通过；
- mypy strict 通过；
- `uv lock --check` 通过；
- mock 流水线输出 `promotable: true`；
- `artifacts/smoke/` 下存在 manifests、job metrics、评测和报告。

检查：

```bash
find artifacts/smoke -name manifest.json -o -name '*.manifest.json' | sort
find artifacts/smoke/reports -type f | sort
```

CPU smoke 的 accuracy 不是模型效果，只证明工程链路能连接起来。

---

## 9. 第二关：tiny GPU smoke

运行：

```bash
make tiny-gpu-smoke 2>&1 | tee logs/02_tiny_gpu_smoke.log
```

这个阶段使用 Qwen2.5-0.5B Student 和 Qwen2.5-1.5B Teacher，真实执行：

- GPU 模型下载与加载；
- rollout；
- Teacher top-k annotation；
- 一个 QLoRA optimizer step；
- adapter 保存与 merge；
- merged 模型重新加载和评测。

成功标准：

```bash
test -f artifacts/tiny_gpu_smoke/checkpoints/weighted_opd_seed42/training_summary.json && \
test -f artifacts/tiny_gpu_smoke/merged/weighted_opd/manifest.json && \
test -f artifacts/tiny_gpu_smoke/evaluation/weighted_opd/smoke/summary.json && \
echo "tiny smoke artifacts exist"
```

如果命令没有报错并打印最后一行，说明文件存在。

失败时保留：

- `logs/02_tiny_gpu_smoke.log`；
- `artifacts/tiny_gpu_smoke/`；
- `nvidia-smi` 输出。

不要在 tiny smoke 失败时直接启动正式实验。

---

## 10. 第三关：Qwen3 正式模型 smoke

运行：

```bash
scripts/qwen_gpu_smoke.sh 2>&1 | tee logs/03_qwen_gpu_smoke.log
```

该脚本会拒绝低于 40GB-class 的 GPU。它使用正式 Qwen3-8B/14B，但只训练 32 条数据、2 个
optimizer steps，并只抽 2 条 MATH-500 做 LightEval，因此费用远低于正式实验。

检查：

```bash
find artifacts/qwen_gpu_smoke -name manifest.json -o -name job_metrics.json | sort
find artifacts/qwen_gpu_smoke/lighteval -type f | sort
cat artifacts/qwen_gpu_smoke/checkpoints/weighted_opd_seed42/training_summary.json
df -h /root/autodl-tmp
```

必须同时满足：

1. rollout 和 Teacher annotation 没有批量失败；
2. Student 与 Teacher tokenizer fingerprint 匹配；
3. loss 和 gradient norm 都是有限数字，不是 `NaN` 或 `Inf`；
4. `best/` adapter 能成功 merge；
5. merged 8B 能被 vLLM 重新加载；
6. LightEval 产生 results、details、`command.json` 和 `manifest.json`；
7. 数据盘仍至少有 15GiB 空闲。

只有这一步完整成功，才能开始正式 15k 实验。

---

## 11. 正式 Round 0：准备数据

每个命令单独执行。上一个成功后再执行下一个。

### 11.1 下载并切分训练数据

```bash
scripts/prepare_data.sh configs/data.yaml 2>&1 | tee logs/10_prepare_data.log
```

期望文件：

```bash
ls -lh artifacts/data/curated/
cat artifacts/data/curated/data_card.json
cat artifacts/data/manifests/data_prepare.json
```

预期切分约为：

- smoke：128；
- validation：1000；
- train：15000。

### 11.2 自动下载固定版本 benchmark

```bash
scripts/fetch_eval_data.sh configs/main.yaml 2>&1 | tee logs/11_fetch_eval_data.log
```

检查：

```bash
ls -lh artifacts/data/eval/
```

应存在 MATH-500、AIME 2024 和 AIME 2025 的 parquet 与 manifest。

### 11.3 运行污染审计

```bash
scripts/audit_contamination.sh configs/main.yaml 2>&1 | tee logs/12_contamination.log
```

检查：

```bash
ls -lh artifacts/data/contamination/
cat artifacts/data/contamination/report.json
```

训练后续使用 `train_clean.parquet`，不要绕过污染审计直接使用原始 train。

### 11.4 运行 Sparse KL 数值审计

```bash
uv run --no-sync opd audit sparse-kl \
  --output artifacts/audits/sparse_kl.json \
  --config configs/main.yaml \
  2>&1 | tee logs/13_sparse_kl_audit.log
```

这一步不证明模型效果；它证明 top-k + tail bucket 的近似 loss 和梯度方向符合设计。

---

## 12. 正式 Round 0：生成 OPD 数据

### 12.1 Student rollout

```bash
scripts/generate_rollouts.sh configs/rollout.yaml 0 \
  2>&1 | tee logs/20_round0_rollout.log
```

主要输出：

```text
artifacts/data/rollouts/round_0/rollouts.parquet
artifacts/data/rollouts/round_0/manifest.json
artifacts/data/rollouts/round_0/job_metrics.json
```

任务中断后重新运行同一条命令。成功 shard 会复用，不需要从头生成。

### 12.2 数学 Verifier

```bash
scripts/verify.sh configs/main.yaml 0 \
  2>&1 | tee logs/21_round0_verifier.log
```

Verifier 输出 `pass/fail/unknown`。`unknown` 不等于错误，所以单独保留。

### 12.3 Teacher annotation

```bash
scripts/annotate_teacher.sh configs/teacher.yaml 0 \
  2>&1 | tee logs/22_round0_teacher.log
```

这是最需要 80GB GPU 的阶段之一。它保存 top-32 log-prob、tail mass 和 token entropy，而不是
完整 vocabulary logits。

任务中断后重新运行相同命令；成功 shard 会复用。

### 12.4 构建四种 OPD 训练视图

```bash
scripts/build_training_views.sh configs/main.yaml 0 \
  2>&1 | tee logs/23_round0_views.log
```

检查：

```bash
ls -lh artifacts/data/training_views/round_0/
```

应有：

- `vanilla_opd.parquet`；
- `verifier_opd.parquet`；
- `confidence_opd.parquet`；
- `weighted_opd.parquet`；
- 对应 manifest。

---

## 13. 正式 Round 0：按顺序训练

不要并行训练。每次只运行一个方法，完成后检查结果和磁盘。

```bash
scripts/train.sh configs/sft.yaml \
  2>&1 | tee logs/30_train_sft.log

scripts/train.sh configs/vanilla_opd.yaml \
  2>&1 | tee logs/31_train_vanilla.log

scripts/train.sh configs/verifier_opd.yaml \
  2>&1 | tee logs/32_train_verifier.log

scripts/train.sh configs/confidence_opd.yaml \
  2>&1 | tee logs/33_train_confidence.log

scripts/train.sh configs/weighted_opd.yaml \
  2>&1 | tee logs/34_train_weighted.log
```

每个训练目录都应至少包含：

```text
best/
latest/
training_summary.json
job_metrics.json
manifest.json
```

训练目录分别是：

```text
checkpoints/sft_seed42/
checkpoints/vanilla_opd_seed42/
checkpoints/verifier_opd_seed42/
checkpoints/confidence_opd_seed42/
checkpoints/weighted_opd_seed42/
```

### 13.1 如何判断训练是否正常

查看摘要，例如：

```bash
cat checkpoints/weighted_opd_seed42/training_summary.json
```

重点检查：

- `global_step` 大于 0；
- `history` 中 loss 和 gradient norm 是有限值；
- 出现 validation accuracy、format pass rate 和 composite score；
- `best_step` 大于 0；
- `best_validation_score` 不是负无穷；
- Early Stop 后仍有完整 `best/`。

### 13.2 训练中断如何恢复

不要删除 `latest/`。打开对应配置，例如：

```bash
nano configs/weighted_opd.yaml
```

在 `training:` 下加入：

```yaml
  resume_from_checkpoint: checkpoints/weighted_opd_seed42/latest
```

保存后重新运行同一训练脚本。恢复会加载模型、optimizer、scheduler、随机状态、epoch 和 batch
位置，不会从数据开头重复训练。

训练完成后可以删除这行，或改回：

```yaml
  resume_from_checkpoint: null
```

如果你修改过配置，执行 `git diff` 并把修改后的配置一起保存到实验备份。

---

## 14. Early Stop 和人工停止规则

默认每 200 optimizer steps 做一次 validation。复合分数是：

```text
0.8 × 数学准确率
+ 0.1 × 答案格式通过率
+ 0.1 × 指令回归分数
```

连续三次提升小于 `0.005` 时停止，并保留最佳 checkpoint。

出现以下情况立即暂停：

- loss 或 gradient norm 出现 NaN/Inf；
- OOM 重复出现；
- validation 明显持续下降；
- 平均输出长度相对 Base 变化超过 30%；
- GPQA 或 IFEval 回退超过 2 个百分点；
- 输出高度重复或 entropy 异常坍缩；
- 数据盘空闲低于 15GiB；
- 预算达到 110 H100h。

查看累计预算：

```bash
uv run --no-sync opd budget --config configs/main.yaml
```

---

## 15. 内部回归评测

先评测 Base：

```bash
uv run --no-sync opd evaluate \
  --suite regression \
  --config configs/main.yaml \
  2>&1 | tee logs/40_eval_base_regression.log
```

LoRA adapter 不能直接被 vLLM 当作完整模型使用。每个候选都必须先 merge。

以 Weighted OPD 为例：

```bash
scripts/merge_checkpoint.sh \
  configs/weighted_opd.yaml \
  checkpoints/weighted_opd_seed42/best \
  artifacts/merged/weighted \
  2>&1 | tee logs/41_merge_weighted.log

uv run --no-sync opd evaluate \
  --suite regression \
  --checkpoint artifacts/merged/weighted \
  --config configs/weighted_opd.yaml \
  2>&1 | tee logs/42_eval_weighted_regression.log
```

内部 paired comparison：

```bash
uv run --no-sync opd compare \
  --baseline artifacts/evaluation/base/regression/summary.json \
  --candidate artifacts/evaluation/weighted_opd/regression/summary.json \
  --output reports/weighted_vs_base_regression.json \
  --config configs/main.yaml
```

内部 regression 用于工程门禁、Early Stop 和快速比较，不能冒充 MATH-500/AIME 正式分数。

---

## 16. LightEval 正式 benchmark

MATH-500 和 AIME 的 LightEval 标准任务会为推理模型预留最多 32768 个生成 token。因此正式
benchmark 使用 Qwen3 的原生 40960 token 上下文；如果把 `benchmark.max_model_length` 设为
4096，LightEval 会在生成前因可用上下文为负数而直接失败。这个设置只影响正式 LightEval，
内部快速评测仍使用较短上下文以节省显存和时间。

项目固定使用 LightEval 0.9.2 和 vLLM 0.10.1.1。不要单独升级 vLLM；vLLM 0.10.2 及以后
删除了 LightEval 0.9.2 仍在调用的 `prompt_token_ids` 参数，会在模型加载完成后报
`LLM.generate() got an unexpected keyword argument 'prompt_token_ids'`。

### 16.1 先评测 Base

```bash
scripts/evaluate_lighteval.sh \
  Qwen/Qwen3-8B \
  artifacts/lighteval/base \
  configs/main.yaml \
  2>&1 | tee logs/50_lighteval_base.log
```

默认任务是：MATH-500、AIME 2024、GPQA Diamond、IFEval。

如果只想先测试 MATH-500：

```bash
LIGHEVAL_TASKS=math500 scripts/evaluate_lighteval.sh \
  Qwen/Qwen3-8B \
  artifacts/lighteval/base_math500 \
  configs/main.yaml \
  2>&1 | tee logs/50_lighteval_base_math500.log
```

### 16.2 评测候选模型

如果上一节已经生成 `artifacts/merged/weighted`：

```bash
scripts/evaluate_lighteval.sh \
  artifacts/merged/weighted \
  artifacts/lighteval/weighted \
  configs/weighted_opd.yaml \
  2>&1 | tee logs/51_lighteval_weighted.log
```

对 SFT、Vanilla、Verifier、Confidence 重复“merge → 评测”。100GB 数据盘不要同时保留多个
merged 8B 模型。

推荐的最低评测矩阵：

| 方法 | MATH-500 | AIME | GPQA | IFEval |
|---|---:|---:|---:|---:|
| Base | 必做 | 必做 | 必做或记录未授权 | 必做 |
| SFT | 必做 | 必做 | 建议 | 必做 |
| Vanilla OPD | 必做 | 必做 | 建议 | 必做 |
| Weighted OPD | 必做 | 必做 | 必做或记录未授权 | 必做 |
| Verifier/Confidence ablation | 必做 | 预算允许 | 可省略 | 建议 |

每个 LightEval 输出目录必须保留：

- 原生 results；
- 逐题 details；
- `command.json`；
- `job_metrics.json`；
- `manifest.json`。

查找结果：

```bash
find artifacts/lighteval -type f | sort
```

正式总分以 LightEval 原生 results/details 为准。

---

## 17. 100GB 磁盘下如何轮换 merged 模型

先检查空间：

```bash
du -h -d 2 artifacts checkpoints "$HF_HOME" | sort -h
df -h /root/autodl-tmp
```

安全顺序：

1. 确认 adapter 的 `best/`、训练摘要和 manifest 已备份；
2. merge 一个方法；
3. 完成内部评测和 LightEval；
4. 确认结果目录已经备份；
5. 删除这个可重建的 merged 模型；
6. 再 merge 下一个方法。

删除前必须确认当前路径：

```bash
pwd
ls -lh artifacts/merged
```

例如 Weighted 评测并备份完成后，才可以执行：

```bash
rm -rf artifacts/merged/weighted
```

不要删除：

- `checkpoints/*/best/`；
- `training_summary.json`；
- manifests；
- LightEval results/details；
- 当前仍会被后续 Round 1 使用的 merged Round 0 模型；
- 尚未完成后续阶段所需的 Hugging Face model snapshot。

---

## 18. 是否运行 Round 1

Round 1 不是无条件必做。只有同时满足以下条件才继续：

- Vanilla 或 Weighted 不差于 SFT；
- Weighted 相比 Vanilla 有可解释的收益；
- 剩余预算至少 20 H100h；
- 数据盘空间充足；
- Round 0 没有明显能力退化。

### 18.1 合并 Round 0 模型

```bash
scripts/merge_checkpoint.sh configs/vanilla_opd.yaml \
  checkpoints/vanilla_opd_seed42/best \
  artifacts/merged/vanilla_opd_round0

scripts/merge_checkpoint.sh configs/weighted_opd.yaml \
  checkpoints/weighted_opd_seed42/best \
  artifacts/merged/weighted_opd_round0
```

### 18.2 Vanilla Round 1

```bash
scripts/generate_rollouts.sh configs/round1_vanilla_opd.yaml 1
scripts/verify.sh configs/round1_vanilla_opd.yaml 1
scripts/annotate_teacher.sh configs/round1_vanilla_opd.yaml 1
uv run --no-sync opd data build-view \
  --round 1 \
  --method vanilla-opd \
  --config configs/round1_vanilla_opd.yaml
scripts/train.sh configs/round1_vanilla_opd.yaml
```

### 18.3 Weighted Round 1

```bash
scripts/generate_rollouts.sh configs/round1_weighted_opd.yaml 1
scripts/verify.sh configs/round1_weighted_opd.yaml 1
scripts/annotate_teacher.sh configs/round1_weighted_opd.yaml 1
uv run --no-sync opd data build-view \
  --round 1 \
  --method weighted-opd \
  --config configs/round1_weighted_opd.yaml
scripts/train.sh configs/round1_weighted_opd.yaml
```

两个分支写入不同数据目录，不会互相覆盖。完成后仍需 merge、内部评测和 LightEval。

如果 Round 0 已经回答研究问题，或 Weighted 没有超过 Vanilla，可以合理地停止 Round 1，并在
报告中写明停止依据。这比为了“做两轮”而浪费算力更专业。

---

## 19. 如何备份面试需要的证据

### 19.1 必须保留的内容

必须保留：

- Git commit SHA；
- 全部 YAML 配置；
- `uv.lock`；
- 环境、GPU 和磁盘信息；
- 每个命令的日志；
- 所有 manifest；
- 所有 `job_metrics.json`；
- 所有 `training_summary.json`；
- 每个方法的最佳 adapter；
- 内部 predictions/summary/comparison；
- LightEval results/details/command；
- 最终报告和图表；
- 失败实验的日志和停止理由。

### 19.2 打包元数据

```bash
mkdir -p backups
tar -czf backups/opd_metadata_$(date +%Y%m%d_%H%M).tar.gz \
  configs eval logs reports README.md PROJECT_PLAN.md DATA_CARD.md MODEL_CARD.md uv.lock \
  $(find artifacts checkpoints -type f \
    \( -name manifest.json -o -name '*.manifest.json' -o -name job_metrics.json \
       -o -name training_summary.json -o -name command.json \))
```

### 19.3 打包最佳 adapter

以 Weighted 为例，排除可重建且较大的 optimizer state：

```bash
tar --exclude='accelerate_state' \
  -czf backups/weighted_best_adapter_seed42.tar.gz \
  checkpoints/weighted_opd_seed42/best
```

对你最终要展示的其他模型也执行同样操作。

### 19.4 打包 LightEval 输出

```bash
tar -czf backups/lighteval_results_$(date +%Y%m%d_%H%M).tar.gz \
  artifacts/lighteval
```

### 19.5 下载到本地

可以使用 AutoDL 网页文件管理器下载 `backups/`，或在本地终端执行：

```bash
scp -P PORT -r root@HOST:/root/autodl-tmp/OPDProj/backups ./opd-backups
```

下载后在本地检查压缩包：

```bash
ls -lh opd-backups
tar -tzf opd-backups/weighted_best_adapter_seed42.tar.gz | head
```

至少保留两份副本，例如本地电脑一份、网盘/对象存储一份。

---

## 20. 如何整理最终实验结果

最终主表至少包含：

| 方法 | Round | MATH-500 | AIME 2024 | GPQA Diamond | IFEval | GPU 小时 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 实测 | 实测 | 实测/未授权 | 实测 | 实测 |
| SFT | 0 | 实测 | 实测 | 实测/未运行 | 实测 | 实测 |
| Vanilla OPD | 0 | 实测 | 实测 | 实测/未运行 | 实测 | 实测 |
| Weighted OPD | 0 | 实测 | 实测 | 实测/未授权 | 实测 | 实测 |

同时报告：

- 相对 Base 的绝对提升；
- 相对 SFT 和 Vanilla OPD 的提升；
- 输出长度与 unknown rate；
- paired bootstrap 区间；
- McNemar exact test；
- Teacher tokens、GPU hours 和质量—成本 Pareto 图；
- 典型成功与失败案例；
- Early Stop 的 best step；
- 所有已知限制。

单 seed 的正确表述是：

> 在训练 seed 42 和当前计算预算下，Weighted OPD 相比对照观察到……

不要写：

> 实验证明该方法在不同随机种子下稳定优于所有基线。

paired bootstrap 只衡量 benchmark 题目抽样不确定性，不能代替多 seed 训练。

---

## 21. 常见错误与处理

### 21.1 `uv: command not found`

```bash
source "$HOME/.local/bin/env"
uv --version
```

### 21.2 `opd: command not found`

不要直接运行裸 `opd`，使用：

```bash
uv run --no-sync opd --help
```

如果仍失败：

```bash
source scripts/autodl_env.sh
scripts/remote_bootstrap.sh
```

### 21.3 `No NVIDIA GPU is visible`

```bash
nvidia-smi
```

如果失败，通常是租到了 CPU 实例、GPU 未正确挂载或驱动环境错误。更换实例或镜像。

### 21.4 `torch cannot see CUDA`

说明 PyTorch、CUDA driver 或虚拟环境不兼容。不要继续训练；保留 bootstrap 日志并更换为支持
CUDA 12.8 的实例/镜像后重新 bootstrap。

### 21.5 OOM

先记录：

```bash
nvidia-smi
tail -200 logs/对应阶段.log
```

24GB 卡不要运行 Qwen3-14B smoke 或正式 Teacher。正式配置优先换 H100 80GB，不要在没有记录的
情况下随意缩短序列、改量化或更换模型，否则实验已经不是预注册方案。

### 21.6 磁盘不足

```bash
df -h /root/autodl-tmp
du -h -d 2 artifacts checkpoints "$HF_HOME" | sort -h
```

先上传 backups，再删除已经完成评测的 merged checkpoint 或可重建的旧 shard。不要直接清空
`HF_HOME`，因为后续阶段可能仍需 8B/14B 权重。

### 21.7 Hugging Face 401/403

```bash
uv run --no-sync hf auth whoami
uv run --no-sync hf auth login
```

GPQA 还需要在网页接受条款。若无法获得权限，记录为未运行。

### 21.8 SSH 断开

重新登录后：

```bash
tmux attach -t opd
```

如果 tmux 会话仍在，任务不会停止。如果进程已退出，查看日志和对应 stage manifest，再按项目的
resume 规则重跑。

### 21.9 Manifest checksum mismatch

说明某个已经登记的 artifact 被修改或删除。不要直接改 manifest。先备份故障目录，确认是哪一
阶段的产物被破坏，再删除该阶段输出并重新运行对应命令。

### 21.10 GPQA/IFEval 回退

数学分数提升但 GPQA/IFEval 明显下降，说明模型可能过度专门化。不要只汇报数学提升；按停止
规则保留 best checkpoint，并在模型卡中披露能力退化。

---

## 22. 关机前最终检查

运行：

```bash
date -Iseconds | tee reports/environment/end_time.txt
nvidia-smi | tee reports/environment/nvidia_smi_final.txt
df -h | tee reports/environment/disk_after.txt
uv run --no-sync opd budget --config configs/main.yaml \
  | tee reports/environment/final_budget.json
git rev-parse HEAD
git status --short
ls -lh backups
```

关机前逐项确认：

- [ ] 所有训练摘要已下载；
- [ ] 所有 LightEval results/details 已下载；
- [ ] 最佳 adapter 已下载；
- [ ] 配置和 Git SHA 已下载；
- [ ] 日志已下载；
- [ ] manifest 与 job metrics 已下载；
- [ ] 本地压缩包可以正常解压；
- [ ] 失败和未运行项目已记录；
- [ ] 没有把 mock accuracy 当成真实结果；
- [ ] 没有声称单 seed 代表跨 seed 稳定性。

全部确认后再停止或释放 AutoDL 实例。

---

## 23. 最短执行清单

当你已经理解前面的原因后，可以按下面顺序逐项打勾：

```text
[ ] 本地 git add / commit / push
[ ] AutoDL H100 80GB + 100GB 数据盘
[ ] clone 到 /root/autodl-tmp/OPDProj
[ ] source scripts/autodl_env.sh
[ ] scripts/remote_bootstrap.sh
[ ] Hugging Face 登录/GPQA 授权
[ ] make check
[ ] make smoke
[ ] make tiny-gpu-smoke
[ ] scripts/qwen_gpu_smoke.sh
[ ] prepare + fetch eval + contamination + sparse KL audit
[ ] Round 0 rollout + verifier + teacher + views
[ ] SFT + 四种 OPD 训练
[ ] Base 和各方法内部 regression
[ ] Base/SFT/Vanilla/Weighted 正式 LightEval
[ ] 根据结果决定是否运行 Round 1
[ ] 生成对比、失败分析和质量—成本报告
[ ] 备份 adapter、results/details、logs、manifests、配置、Git SHA
[ ] 本地验证备份后关机
```

第一次操作时不要一次粘贴整份清单。一次执行一个阶段，确认成功标准，再进入下一阶段。
