# AutoDL Remote GPU Runbook

本手册按“先门禁、再小跑、最后正式实验”的顺序执行。所有命令都从仓库根目录运行；脚本使用 `uv run --no-sync`，因此每台新服务器必须先成功执行一次 bootstrap。

## 1. 租用与磁盘

- 系统：Ubuntu x86_64，Python 由 `uv` 固定为 3.11/3.12。
- 正式 Teacher 阶段：推荐单张 H100 80GB；24GB 卡只用于 Qwen2.5 tiny smoke，Qwen3 smoke 至少需要 40GB-class GPU。
- 数据盘：100GB 时把仓库克隆到 `/root/autodl-tmp/OPDProj`，不要放系统盘。
- 镜像/驱动需支持锁文件中的 PyTorch 2.9.1 + CUDA 12.8；preflight 会实际检查 CUDA 与 BF16。
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

再用正式 Qwen3-8B/14B 做 32 条真实链路，并运行 2 条 MATH-500 LightEval：

```bash
scripts/qwen_gpu_smoke.sh
```

该脚本会在启动前读取显存并拒绝低于 40GB-class 的 GPU。40/48GB 仅用于小规模门禁；
7.5k 正式 Teacher annotation 仍推荐 H100/A800 80GB。

只有以下项目全部成立，才允许启动正式 7.5k rollout：

1. `artifacts/qwen_gpu_smoke/` 下各阶段存在 `manifest.json` 和 `job_metrics.json`；
2. rollout/annotation 无批量失败，Teacher 与 Student tokenizer fingerprint 一致；
3. loss、gradient norm 均为有限值，`best/` adapter 可合并并重新加载；
4. LightEval 生成 `results/`、`details/`、`command.json` 和 manifest；
5. 峰值显存、tokens/s、剩余磁盘符合预算。

失败时保留整个 `artifacts/qwen_gpu_smoke/`，不要直接扩容重跑正式实验。

## 4. Round 0

数据和固定 benchmark 会从 Hugging Face 指定 revision 自动获取，无需手工制作文件：

```bash
scripts/prepare_data.sh configs/data.yaml
scripts/fetch_eval_data.sh configs/main.yaml
scripts/audit_contamination.sh configs/main.yaml
uv run --no-sync opd audit sparse-kl \
  --output artifacts/audits/sparse_kl.json \
  --config configs/main.yaml
```

共享一次 Base on-policy 数据：

```bash
scripts/generate_rollouts.sh configs/rollout.yaml 0
cat artifacts/data/rollouts/round_0/quality_gate.json
scripts/verify.sh configs/main.yaml 0
scripts/annotate_teacher.sh configs/teacher.yaml 0
scripts/build_training_views.sh configs/main.yaml 0
```

数据准备仍保留 15,000 条清洗候选题；rollout 和每种训练方法只使用固定前 7,500 条。生成上限为
3,072 response tokens，共 38 个 shard。程序会在累计 800 条成功样本时检查截断率：

- `collecting`：尚未到 800 条；
- `passed`：截断率不高于 20%，程序继续完成剩余数据；
- `failed`：程序自动终止，禁止继续 Verifier 或 Teacher。

只有最终生成 `rollouts.parquet`、`manifest.json`，且 `quality_gate.json` 为 `passed`，才能执行
后续命令。旧的 1,536-token pilot shard 使用不同 hash，不会被复用，应保留为失败实验记录。

按顺序训练，避免同时占用磁盘和 GPU：

```bash
scripts/train.sh configs/sft.yaml
scripts/train.sh configs/vanilla_opd.yaml
scripts/train.sh configs/verifier_opd.yaml
scripts/train.sh configs/confidence_opd.yaml
scripts/train.sh configs/weighted_opd.yaml
```

训练中断后，在对应配置的 `training.resume_from_checkpoint` 指向 `.../latest` 再运行同一脚本。checkpoint 保存 optimizer、scheduler、随机状态、epoch/batch cursor 和完整 history，不会从数据开头重复训练。

## 5. Round 1

只有 Vanilla 或 Weighted 不差于 SFT 且剩余预算不少于 20 H100h 时继续。两条分支使用隔离的数据目录，不会互相覆盖。

先合并 Round 0：

```bash
scripts/merge_checkpoint.sh configs/vanilla_opd.yaml \
  checkpoints/vanilla_opd_seed42/best \
  artifacts/merged/vanilla_opd_round0
scripts/merge_checkpoint.sh configs/weighted_opd.yaml \
  checkpoints/weighted_opd_seed42/best \
  artifacts/merged/weighted_opd_round0
```

Vanilla Round 1：

```bash
scripts/generate_rollouts.sh configs/round1_vanilla_opd.yaml 1
scripts/verify.sh configs/round1_vanilla_opd.yaml 1
scripts/annotate_teacher.sh configs/round1_vanilla_opd.yaml 1
uv run --no-sync opd data build-view --round 1 --method vanilla-opd \
  --config configs/round1_vanilla_opd.yaml
scripts/train.sh configs/round1_vanilla_opd.yaml
```

Weighted Round 1：

```bash
scripts/generate_rollouts.sh configs/round1_weighted_opd.yaml 1
scripts/verify.sh configs/round1_weighted_opd.yaml 1
scripts/annotate_teacher.sh configs/round1_weighted_opd.yaml 1
uv run --no-sync opd data build-view --round 1 --method weighted-opd \
  --config configs/round1_weighted_opd.yaml
scripts/train.sh configs/round1_weighted_opd.yaml
```

## 6. 正式评测

先合并 adapter，再分别写入独立输出目录：

```bash
scripts/merge_checkpoint.sh configs/sft.yaml checkpoints/sft_seed42/best artifacts/merged/sft
scripts/merge_checkpoint.sh configs/vanilla_opd.yaml checkpoints/vanilla_opd_seed42/best artifacts/merged/vanilla
scripts/merge_checkpoint.sh configs/weighted_opd.yaml checkpoints/weighted_opd_seed42/best artifacts/merged/weighted

scripts/evaluate_lighteval.sh Qwen/Qwen3-8B artifacts/lighteval/base
scripts/evaluate_lighteval.sh artifacts/merged/sft artifacts/lighteval/sft
scripts/evaluate_lighteval.sh artifacts/merged/vanilla artifacts/lighteval/vanilla
scripts/evaluate_lighteval.sh artifacts/merged/weighted artifacts/lighteval/weighted
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

- 达到 90 H100h 后只完成必要评测；达到 110 H100h 不再启动新训练。
- 连续三次 validation 提升低于 0.005 时 early-stop。
- NaN/Inf、持续 OOM、输出长度变化超过 30%、GPQA/IFEval 回退超过 2 个百分点时暂停。
- 每阶段后执行 `uv run --no-sync opd budget --config configs/main.yaml` 并备份新增 artifacts。
