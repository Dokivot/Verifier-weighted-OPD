# SuRe K2 24h：AutoDL 小白运行指南

本文只对应当前 Qwen3 strict online OPD 主线。旧 Qwen2.5 指南不能与本实验混用。

## 1. 服务器与镜像

- GPU：单张 RTX PRO 6000 Blackwell 96GB；
- Python：3.11；
- PyTorch：仓库 lock 固定的 Linux CUDA 12.8 wheel；
- 数据盘：至少 200GB，开始前建议可用空间不低于 100GB；
- 正式命令全部在 `tmux` 中执行。

进入仓库并加载统一缓存目录：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
mkdir -p logs
```

每次重开 shell 都要重新执行 `source scripts/autodl_env.sh`。

## 2. 拉取代码与安装

```bash
git pull
scripts/remote_bootstrap.sh 2>&1 | tee logs/00_bootstrap_sure_k2.log
uv run --no-sync hf auth whoami
```

不要删除 `HF_HOME` 或 uv cache；模型和 wheel 会复用缓存。

## 3. 第一关：CPU 检查

```bash
set -o pipefail
make check 2>&1 | tee logs/01_make_check_sure_k2.log
```

必须同时看到 tests、ruff、format、mypy 和 lock 全部通过。任何一项失败都不要启动 GPU 正式训练。

## 4. 下载并固定数据

```bash
scripts/prepare_data.sh configs/sure_k2_24h.yaml \
  2>&1 | tee logs/02_prepare_deepmath.log

uv run --no-sync opd data fetch-eval \
  --name math500 --config configs/sure_k2_24h.yaml \
  2>&1 | tee logs/03_fetch_math500.log

uv run --no-sync opd data fetch-eval \
  --name amc23 --config configs/sure_k2_24h.yaml \
  2>&1 | tee logs/04_fetch_amc23.log
```

检查数量：

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path("artifacts/sure_k2_24h/data")
for path in [
    root / "curated/smoke.parquet",
    root / "curated/train.parquet",
    root / "eval/math500.parquet",
    root / "eval/amc23.parquet",
]:
    print(path, pq.read_metadata(path).num_rows)
PY
```

预期原始 train 为 `30720`，smoke 为 `8`。其中 55 个正式 batch 只需要 `28160` 条；额外的
`2560` 条是去污染后仍能满足固定 batch 训练的缓冲，不能把它们误解为额外训练 step。下载后的
manifest、Parquet 和 `data_card.json` 都要保留。

接着必须做去污染审计，正式训练和 pilot 只能读取 clean split：

```bash
uv run --no-sync opd data audit-contamination --config configs/sure_k2_24h.yaml
cat artifacts/sure_k2_24h/data/contamination/report.json
```

保留 `train_clean.parquet`、`quarantine.parquet`（如存在）、`report.json` 和 `manifest.json`。
若 clean split 少于 `28160` 条，停止并扩大原始 train split 后重新 prepare/audit；不要绕过去污染直接训练。

训练开始前会写入 `checkpoints/sure_k2_seed42/checkpoint_storage_plan.json`。它按实际 Student 参数量计算
rolling state 原子替换所需的双份空间，并额外预留 28 GiB 给尚未缓存的 8B Teacher、step artifacts、日志和
评测输出；低于 80 GiB 或估算峰值时会拒绝启动。不要删除 rolling 目录来绕过该检查。

## 5. 第二关：真实模型 smoke

```bash
set -o pipefail
scripts/run_sure_k2_smoke.sh \
  2>&1 | tee logs/05_sure_k2_gpu_smoke.log
```

它会加载真实 1.7B Student 和 8B Teacher，对 8 条样本完成在线生成、Teacher scoring、全参数反传、
optimizer step 和 checkpoint 保存。Student 使用 FP32 master 参数/优化器状态与 BF16 compute，避免
`1e-6` 更新在 BF16 参数本体上被舍入掉。smoke 只验证接口，不代表效果。

通过条件：

- 命令退出码为 0；
- `artifacts/sure_k2_24h/smoke/checkpoint/training_summary.json` 的 `status` 为 `completed`；
- loss、gradient norm 和 SuRe weights 均为有限值；
- `sure_weight_min >= 1`、`sure_weight_max <= 2`；
- `telemetry/step_000001_quality_gate.json` 中存在 `finish_reason_counts`、终止 token 列表和长度分位数；
- stop token 包含 Qwen 的 `<|im_end|>` 与 `<|endoftext|>`（两者都在当前 tokenizer 中时）；
- 没有 OOM、空 response 或 tokenizer mismatch。

## 6. 第三关：512-prompt pilot 与恢复

```bash
set -o pipefail
scripts/run_sure_k2_pilot.sh \
  2>&1 | tee logs/06_sure_k2_pilot.log
```

脚本第一次运行 step 1 并正常暂停，第二次从 rolling checkpoint 恢复并运行 step 2。它使用正式
`max_response_tokens=8192` 和 global batch 512，因此能暴露真实显存、截断率和速度问题。

查看两个 step：

```bash
cat artifacts/sure_k2_24h/pilot/checkpoint/telemetry/step_000001.json
cat artifacts/sure_k2_24h/pilot/checkpoint/telemetry/step_000002.json
cat artifacts/sure_k2_24h/pilot/checkpoint/training_summary.json
```

估时使用 step 2 的 `step_seconds`（step 1 可能包含首次初始化抖动）：

```text
预计训练小时 = step_2.step_seconds × 55 / 3600
```

pilot 脚本会自动写入 `pilot_budget_report.json`：以 step 2 的稳态耗时估计 55 steps；若单 step 超过
`1440` 秒（约 24 分钟）会失败。这个门槛为 55 step 预留约 2 小时给初始化、checkpoint 和评测，不能
通过就不要开始正式实验。若出现 OOM，先把正式、pilot 配置中的 `rollout_micro_batch_size` 降到 `1`；
不要改 global batch、模型、loss 或长度。若截断率超过 20%，不要开始正式实验，先保留 quality gate JSON
并分析 EOS/template。

## 7. 正式运行

数据准备、fetch 和去污染审计完成后，才可以从 stage 20 开始：

```bash
tmux new -s sure-k2
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail

START_STAGE=20 scripts/run_sure_k2_24h.sh \
  2>&1 | tee logs/10_sure_k2_24h.log
```

该脚本会拒绝开始训练，直到以下工件同时有效：去污染 manifest、真实模型 smoke 的 completed summary，
以及 2-step pilot 的 passed 预算报告。仅用于开发排障时才可设置 `SKIP_READINESS_CHECK=1`；该开关禁止
用于正式实验及其报告。

脚本顺序是：55-step SuRe 训练 → Base MATH-500 → Base AMC23 → SuRe MATH-500 → SuRe AMC23 →
两份比较报告。不要同时启动第二个正式训练进程。

按 `Ctrl+B`、再按 `D` 退出 tmux，不会停止任务。重新连接：

```bash
tmux attach -t sure-k2
```

## 8. 实时监控

新开一个终端：

```bash
cd /root/autodl-tmp/OPDProj
watch -n 5 nvidia-smi
```

查看已提交 step 和最新 telemetry：

```bash
watch -n 30 'find artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/steps \
  -name "step_*.parquet" 2>/dev/null | wc -l'

latest=$(find artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/telemetry \
  -name 'step_*.json' ! -name '*quality_gate*' | sort | tail -1)
cat "$latest"
```

重点检查 `truncation_rate`、`loss`、`gradient_norm`、`sure_weight_min/max`、`step_seconds` 和
`verifier_score`，以及 quality gate 中的 `finish_reason_counts` 和 `response_length_tokens`。训练 loss
不要求单调下降，但不能出现 NaN/Inf，weight 必须在 `[1,2]`；若 `<|im_end|>` 已作为 stop 但 length
结束比例仍高，禁止继续正式训练。

## 9. 暂停与恢复

最安全的主动暂停方式是等一个新 `step_XXXXXX.json` 和 rolling checkpoint 写完后，再在训练终端按
`Ctrl+C`。不要在屏幕显示正在保存 checkpoint 时关机。

恢复：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail

START_STAGE=20 \
RESUME_FROM=artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/rolling \
scripts/run_sure_k2_24h.sh \
  2>&1 | tee -a logs/10_sure_k2_24h.log
```

恢复会校验 data checksum、run ID、连续 step 文件和 policy hash chain。已提交 step 不会重跑。若上次
恰好在 step Parquet 写完、rolling 尚未提交时中断，程序会拒绝自动猜测；保留现场并分析，不要手删文件。

## 10. 单独重跑评测

训练已经完成但评测失败时，不要重训。按失败位置设置：

```bash
START_STAGE=30 scripts/run_sure_k2_24h.sh 2>&1 | tee logs/11_eval_retry.log
```

`30` 从 Base MATH 开始，`31` 从 Base AMC 开始，`40` 从 SuRe MATH 开始，`41` 从 SuRe AMC 开始，
`50` 只重建比较报告。

最终重点文件：

```text
artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/training_summary.json
artifacts/sure_k2_24h/evaluation/base/math500/summary.json
artifacts/sure_k2_24h/evaluation/base/amc23/summary.json
artifacts/sure_k2_24h/evaluation/sure_k2/math500/summary.json
artifacts/sure_k2_24h/evaluation/sure_k2/amc23/summary.json
artifacts/sure_k2_24h/reports/math500_base_vs_sure.json
artifacts/sure_k2_24h/reports/amc23_base_vs_sure.json
```

## 11. 关机前备份

AutoDL 系统盘可能随实例释放丢失。至少把以下内容打包到数据盘，再下载到 Mac 或上传对象存储：

```bash
tar -czf /root/autodl-tmp/sure_k2_results_$(date +%Y%m%d_%H%M%S).tar.gz \
  artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/training_summary.json \
  artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/telemetry \
  artifacts/sure_k2_24h/evaluation \
  artifacts/sure_k2_24h/reports \
  logs/sure_k2_24h \
  configs/sure_k2_24h.yaml \
  uv.lock
```

若要以后继续训练，还必须额外保留整个
`artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/rolling`。最终模型位于 `final`，step 34/55 模型位于
`checkpoints/step_000034` 和 `checkpoints/step_000055`。
