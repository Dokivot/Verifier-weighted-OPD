# AutoDL Remote GPU Runbook

> 当前执行以根目录 [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) 的推荐方案一为准：
> `Qwen2.5-1.5B-Instruct` Student + `Qwen2.5-Math-7B-Instruct` Teacher。本文中旧的
> 旧 Qwen3-8B/14B 方案只保存在 `plans/`，本文命令均以当前 Qwen2.5 方案为准。

本手册按“先门禁、再小跑、最后正式实验”的顺序执行。所有命令都从仓库根目录运行；脚本使用 `uv run --no-sync`，因此每台新服务器必须先成功执行一次 bootstrap。

当前默认正式任务是 [`RESUME_MVP.md`](RESUME_MVP.md) 的 3,000 prompts 方案。7.5k 多方法矩阵
保留为后续 Phase B/C，不应在首轮结果产生前一次性启动。

## 1. 租用与磁盘

- 系统：Ubuntu x86_64，Python 由 `uv` 固定为 3.11/3.12。
- 正式 Teacher 阶段：推荐单张 24GB–80GB GPU；24GB 卡可执行方案一 smoke 和小规模正式任务，80GB 卡更适合完整 annotation。
- 数据盘：100GB 时把仓库克隆到 `/root/autodl-tmp/OPDProj`，不要放系统盘。
- AutoDL 基础镜像推荐 PyTorch 2.8.0 + CUDA 12.8；`uv.lock` 会在 `.venv` 中安装实际使用的
  PyTorch 2.7.1 + CUDA 12.6 runtime。preflight 会实际检查 CUDA 与 BF16。
- 开始前必须提交 Git commit。正式 promotion 默认拒绝 `git_commit=unknown`。

```bash
cd /root/autodl-tmp
git clone YOUR_REPOSITORY_URL OPDProj
cd OPDProj
git rev-parse HEAD
git status --short
```

`git status --short` 应为空。仓库必须包含 `src/opd/checkpoints/`；旧版本的 `.gitignore` 曾错误忽略该源码目录。

## 2. 环境初始化

安装 `uv` 后，先加载数据盘环境变量，再 bootstrap：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
source scripts/autodl_env.sh
scripts/remote_bootstrap.sh
```

每次 SSH 新会话都重新执行：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
```

`scripts/autodl_preflight.sh` 会拒绝错误架构、不可见 GPU、不支持 BF16、缺失依赖或数据盘剩余空间低于 70GiB 的环境。随后运行 CPU 门禁：

```bash
make check
make smoke
```

## 3. 两级 GPU 门禁

先用小模型验证 CUDA、QLoRA、Teacher top-k、合并与评测：

```bash
make tiny-gpu-smoke
```

再用正式 Qwen2.5-1.5B/Math-7B 做 32 条真实链路，并运行 2 条 MATH-500 LightEval：

```bash
scripts/qwen_gpu_smoke.sh
```

该脚本会在启动前读取显存并拒绝低于 24GB-class 的 GPU。24GB 可完成门禁，完整 Teacher
annotation 仍推荐 H100/A800 80GB。

只有以下项目全部成立，才允许启动正式 MVP rollout：

1. `artifacts/qwen_gpu_smoke/` 下各阶段存在 `manifest.json` 和 `job_metrics.json`；
2. rollout/annotation 无批量失败，Teacher 与 Student tokenizer fingerprint 一致；
3. loss、gradient norm 均为有限值，`best/` adapter 可合并并重新加载；
4. LightEval 生成 `results/`、`details/`、`command.json` 和 manifest；
5. 峰值显存、tokens/s、剩余磁盘符合预算。

失败时保留整个 `artifacts/qwen_gpu_smoke/`，不要直接扩容重跑正式实验。

## 4. 简历 MVP

数据和固定 benchmark 会从 Hugging Face 指定 revision 自动获取，无需手工制作文件。GPU 门禁
通过后直接运行：

```bash
set -o pipefail
scripts/run_resume_mvp.sh
```

主方法和 Base benchmark 完成后运行 SFT：

```bash
scripts/run_resume_sft.sh
```

MVP 使用 6,000 条候选题、固定前 3,000 条 rollout，每题两个候选，vLLM context 为 8,192，
生成上限为 4,096 response tokens；Teacher/训练使用前 4,096 个总 tokens。程序会在累计 800 条
成功样本时检查截断率：

- `collecting`：尚未到 800 条；
- `passed`：截断率不高于 20%，程序继续完成剩余数据；
- `failed`：程序自动终止，禁止继续 Verifier 或 Teacher。

只有最终生成 `rollouts.parquet`、`manifest.json`，且 `quality_gate.json` 为 `passed`，脚本才会
进入后续阶段。旧的 1,536-token pilot shard 使用不同 hash，不会被复用，应保留为失败实验记录。

若任务失败，可按阶段恢复，例如从 Teacher annotation 开始：

```bash
START_STAGE=13 scripts/run_resume_mvp.sh
```

阶段号和检查项见 [`RESUME_MVP.md`](RESUME_MVP.md)。训练中断后设置：

```bash
START_STAGE=20 \
RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/latest \
scripts/run_resume_mvp.sh
```

Phase B/C 的 7.5k 多方法命令保留在 [`PROJECT_PLAN.md`](../PROJECT_PLAN.md)，MVP 完成前不要执行。

## 5. Round 1

首轮不运行 Round 1。先完成 Base、SFT、VFS-Weighted B50，再补 Random-B50 和 Dense-B100。
只有等预算主比较、artifact 和成本报告完整且剩余预算不少于 15 H100h 时，才为最佳方法新增
隔离的 Round 1 配置；不要复用旧的 Vanilla/Weighted Round 1 配置。

## 6. 正式评测

MVP 脚本会自动合并 adapter 并分别写入独立输出目录。以下手工命令只用于 Phase B/C：

```bash
scripts/merge_checkpoint.sh configs/sft.yaml checkpoints/sft_seed42/best artifacts/merged/sft
scripts/merge_checkpoint.sh configs/dense_opd.yaml checkpoints/dense_b100_seed42/best artifacts/merged/dense_b100
scripts/merge_checkpoint.sh configs/random_budget_b50.yaml checkpoints/random_b50_seed42/best artifacts/merged/random_b50
scripts/merge_checkpoint.sh configs/vfs_b50.yaml checkpoints/vfs_b50_seed42/best artifacts/merged/vfs_b50

scripts/evaluate_lighteval.sh Qwen/Qwen2.5-1.5B-Instruct artifacts/lighteval/base
scripts/evaluate_lighteval.sh artifacts/merged/sft artifacts/lighteval/sft
scripts/evaluate_lighteval.sh artifacts/merged/dense_b100 artifacts/lighteval/dense_b100
scripts/evaluate_lighteval.sh artifacts/merged/random_b50 artifacts/lighteval/random_b50
scripts/evaluate_lighteval.sh artifacts/merged/vfs_b50 artifacts/lighteval/vfs_b50
```

包装器已按 LightEval 0.9.2 使用 `model_name=`、chat template、逐题 details，并通过 `lighteval.tasks.extended.ifeval.main` 注册 IFEval。GPQA 首次下载可能要求先在 Hugging Face 接受条款并执行 `uv run --no-sync hf auth login`。

MATH-500、AIME、GPQA 与 IFEval 的正式总分以 LightEval 原生 results/details 为准；项目内
`opd compare` 只消费内部 evaluation schema，不能直接读取 LightEval JSON。需要显著性检验时，
先保留各模型逐题 details，再按相同题目 ID 对齐分析，不能把内部 regression 分数冒充正式榜单分数。

## 7. 100GB 清理与备份

任何删除前先上传：配置、Git SHA、所有 manifest、`job_metrics.json`、`training_summary.json`、LightEval results/details、最终 adapter 和最终 merged checkpoint。

```bash
du -h -d 2 artifacts checkpoints "$HF_HOME" | sort -h
tar -czf opd-metadata-$(date +%Y%m%d-%H%M).tar.gz \
  configs eval reports \
  $(find artifacts checkpoints -name manifest.json -o -name job_metrics.json -o -name training_summary.json)
```

建议清理顺序：

1. 已合并并备份后删除非最佳 `latest/accelerate_state`；
2. LightEval 完成并备份后删除当前 merged 模型，再合并下一个；
3. Round 0 完成后保留合并模型、manifest 和最终表，删除可重建的旧 shard；
4. 最后才运行 `uv cache clean` 或删除无用 Hugging Face snapshot；不要在模型仍被后续阶段引用时清缓存。

始终至少保留 15GiB 空闲空间，低于该值停止新阶段。

## 8. 停止规则

- 达到 40 H100h 后只完成必要评测；达到 60 H100h 不再启动新训练。
- 连续三次 validation 提升低于 0.005 时 early-stop。
- NaN/Inf、持续 OOM、输出长度变化超过 30%、GPQA/IFEval 回退超过 2 个百分点时暂停。
- 每阶段后执行 `uv run --no-sync opd budget --config configs/main.yaml` 并备份新增 artifacts。
