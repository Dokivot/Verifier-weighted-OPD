# 使用现有 6,000 条 Rollout 运行全参数 Vanilla OPD

本指南使用已经生成的 Round 0 数据训练标准 Vanilla OPD。所有 6,000 条 Student rollout 都进入
training view，Verifier 结果只用于后续分析，不参与 loss 加权；所有样本权重和 token 权重均为 `1.0`。
当前主配置是全参数 BF16 训练，不使用 LoRA。QLoRA 仍保留为独立的备用对照，见
`configs/dense_vanilla_lora_mvp.yaml`。
后续 verifier-guided 设计继续保存在 [`VERIFIER_DESIGN.md`](VERIFIER_DESIGN.md)，本实验不会覆盖它。

## 1. 本次复用什么

直接复用以下已有数据，不重新执行数据划分、污染检查、rollout、verify 或 B50 selection：

```text
artifacts/resume_mvp/data/rollouts/round_0/rollouts.parquet
artifacts/resume_mvp/data/verifications/round_0/math.parquet
artifacts/resume_mvp/data/annotations/vfs_weighted_b50/round_0/teacher.parquet
```

最后一个文件包含约 3,064 条已有 Teacher top-k logits。Stage 13 会按 `rollout_id` 复用其中兼容的
标注，仅为其余约 2,936 条 rollout 补算 Teacher logits，最终生成独立的 6,000 条全量标注文件。
程序会核对 Teacher 模型、revision、tokenizer fingerprint 和 `top_k=64`；不兼容时直接停止。

## 2. 先停止旧训练

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh

pgrep -af 'run_resume_mvp|scripts/train.sh|accelerate launch|opd train'
```

若有旧的 weighted OPD 进程，回到对应 tmux 窗口按 `Ctrl+C`。不要让两次训练同时占用 GPU。
旧 checkpoint 和日志不会被删除，新实验使用完全独立的目录。

正式训练前，如果这是该服务器第一次运行全参数分支，建议在已经完成的 Qwen GPU smoke 数据上
额外验证一次全参数加载、反向传播和 checkpoint 保存：

```bash
scripts/qwen_full_parameter_smoke.sh \
  2>&1 | tee logs/qwen_full_parameter_smoke.log
```

它不会重新生成正式 6,000 条 rollout，也不会重新计算正式 Teacher logits。

## 3. 检查三个输入文件

```bash
test -f artifacts/resume_mvp/data/rollouts/round_0/rollouts.parquet
test -f artifacts/resume_mvp/data/verifications/round_0/math.parquet
test -f artifacts/resume_mvp/data/annotations/vfs_weighted_b50/round_0/teacher.parquet
echo "三个输入文件均存在"
```

检查记录数量和唯一 ID：

```bash
uv run --no-sync python - <<'PY'
import pyarrow.parquet as pq

rollout_path = "artifacts/resume_mvp/data/rollouts/round_0/rollouts.parquet"
reuse_path = (
    "artifacts/resume_mvp/data/annotations/"
    "vfs_weighted_b50/round_0/teacher.parquet"
)
rollout_ids = pq.read_table(rollout_path, columns=["rollout_id"])["rollout_id"].to_pylist()
reuse_rows = pq.read_table(reuse_path, columns=["rollout_id", "status"]).to_pylist()
reusable_ids = {row["rollout_id"] for row in reuse_rows if row["status"] == "success"}

print("rollouts:", len(rollout_ids))
print("unique rollout IDs:", len(set(rollout_ids)))
print("reusable Teacher annotations:", len(reusable_ids & set(rollout_ids)))
print("Teacher annotations still needed:", len(set(rollout_ids) - reusable_ids))
assert len(rollout_ids) == 6000
assert len(set(rollout_ids)) == 6000
PY
```

## 4. 从 Stage 13 启动全量流程

建议在 tmux 中运行：

```bash
tmux new -s dense_vanilla
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail

scripts/run_dense_vanilla_mvp.sh
```

脚本按顺序执行：

```text
13 补齐 Teacher annotations
14 构建 6,000 条 vanilla_opd training view
20 训练一轮：6,000 / gradient_accumulation_steps 16 = 375 optimizer steps
21 重新保存/整理全参数 checkpoint（不会进行 LoRA merge）
22 运行本地 regression evaluation
23 运行 LightEval benchmark
24 用完全相同的协议评测未训练的 Student baseline
```

训练期间每 125 步保存一次 `latest`，但只在完整一轮的第 375 步运行验证并生成 `best`。这样最终
合并和评测的 `best` 一定见过全部 6,000 条记录，不会因为较早验证分数持平而退回只见过部分数据的
第 125 步 checkpoint。

Stage 21 的脚本名称仍沿用历史的 `merge_checkpoint.sh`，但在全参数配置下会直接从完整模型
checkpoint 复制/保存为可评测目录，不会加载或合并 LoRA adapter。

按 `Ctrl+B`，再按 `D` 可退出 tmux 而不停止任务；使用 `tmux attach -t dense_vanilla` 返回。

## 5. 监控 Stage 13

另开 SSH 终端：

```bash
cd /root/autodl-tmp/OPDProj
watch -n 30 nvidia-smi
```

查看最新日志：

```bash
tail -f "$(find logs/dense_vanilla_mvp -name '13_teacher.log' -print | sort | tail -1)"
```

查看已经完成的新 shard 数量；每个完整 shard 最多 200 条：

```bash
watch -n 30 \
'find artifacts/resume_mvp/data/annotations/dense_vanilla_b100/round_0/shards \
  -type f -name "*.parquet" 2>/dev/null | wc -l'
```

Stage 13 完成后检查复用和补算数量：

```bash
cat artifacts/resume_mvp/data/annotations/dense_vanilla_b100/round_0/manifest.json
```

首次完整执行时应看到：

```text
record_count: 6000
reused_annotation_records: 约 3064
generated_records: 约 2936
failure_count: 0
```

准确数量以 manifest 为准。若中断后重跑，已经写完的新 shard 会计入 `resumed_records`，因此无需从头算。

## 6. 检查 6,000 条 Training View

Stage 14 完成后执行：

```bash
uv run --no-sync python - <<'PY'
from collections import Counter

import pyarrow.parquet as pq

path = "artifacts/resume_mvp/data/training_views/round_0/dense_vanilla_b100.parquet"
rows = pq.read_table(path).to_pylist()
token_weights = {weight for row in rows for weight in row["confidence_weights"]}

print("records:", len(rows))
print("unique rollout IDs:", len({row["rollout_id"] for row in rows}))
print("methods:", Counter(row["method"] for row in rows))
print("verifier weights:", Counter(row["verifier_weight"] for row in rows))
print("token weights:", sorted(token_weights))

assert len(rows) == 6000
assert len({row["rollout_id"] for row in rows}) == 6000
assert {row["method"] for row in rows} == {"vanilla_opd"}
assert {row["verifier_weight"] for row in rows} == {1.0}
assert token_weights == {1.0}
PY
```

这一步通过，才表示 6,000 条数据全部进入可训练数据集，而不是只训练原 B50 子集。

## 7. 中断后恢复

Stage 13 或 14 中断，直接从对应阶段重跑：

```bash
START_STAGE=13 scripts/run_dense_vanilla_mvp.sh
```

训练 Stage 20 中断，先确认 checkpoint：

```bash
test -f \
artifacts/resume_mvp/checkpoints/dense_vanilla_full_b100_seed42/latest/checkpoint_state.json
```

再恢复：

```bash
START_STAGE=20 \
RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/dense_vanilla_full_b100_seed42/latest \
scripts/run_dense_vanilla_mvp.sh
```

如果训练、合并和 regression 都已完成，仅 benchmark 失败：

```bash
START_STAGE=23 scripts/run_dense_vanilla_mvp.sh
```

如果候选 benchmark 已完成、只缺同协议 baseline：

```bash
START_STAGE=24 scripts/run_dense_vanilla_mvp.sh
```

## 备用 LoRA 对照

主方案结束后，可以用完全相同的 rollout、Teacher annotations 和 training view 运行纯 LoRA：

```bash
scripts/run_dense_vanilla_lora_mvp.sh \
  2>&1 | tee logs/dense_vanilla_lora_mvp.log
```

它使用独立的 training view、checkpoint、merged model、evaluation 和 LightEval 目录，不会覆盖
全参数结果。纯 LoRA 保持 BF16 基座权重冻结，只训练 adapter；比较时只改变
`parameter_update_mode`，不改变数据、seed、训练步数和 benchmark。若要运行 4-bit QLoRA 备选，
使用 `configs/dense_vanilla_qlora_mvp.yaml`，并用相同的 runner 指定该配置。

## 8. 需要永久保留的结果

```text
configs/dense_vanilla_mvp.yaml
logs/dense_vanilla_mvp/
artifacts/resume_mvp/data/annotations/dense_vanilla_b100/round_0/manifest.json
artifacts/resume_mvp/data/training_views/round_0/dense_vanilla_b100.manifest.json
artifacts/resume_mvp/checkpoints/dense_vanilla_full_b100_seed42/training_summary.json
artifacts/resume_mvp/checkpoints/dense_vanilla_full_b100_seed42/manifest.json
artifacts/resume_mvp/merged/dense_vanilla_full_b100/manifest.json
artifacts/resume_mvp/evaluation/dense_vanilla_full_b100/
artifacts/resume_mvp/lighteval/dense_vanilla_full_b100/
artifacts/resume_mvp/lighteval/base_dense_protocol/
```

模型文件较大时，至少把全部 JSON manifest、training summary、benchmark results 和日志下载到本地。
