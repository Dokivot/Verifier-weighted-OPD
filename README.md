# OPD-Lab

面向 LLM 后训练实习作品集的工程级 **On-Policy Distillation（OPD）** 项目。当前主方法是
**VFS-Weighted OPD**：先验证同题多个 Student rollout，在固定 Teacher-token 预算内选择值得
标注的 on-policy states，再用 verifier 样本权重与 Teacher entropy token 权重训练。模型固定为
`Qwen2.5-1.5B-Instruct` Student 与 `Qwen2.5-Math-7B-Instruct` Teacher。

完整实验设计与 GPU 预算见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。更大规模的备用方案保存在 [`plans/`](plans/README.md)。

项目按“先完整结果、后公平基线、最后消融”推进。3,000 prompts 的首轮执行方案见
[`docs/RESUME_MVP.md`](docs/RESUME_MVP.md)。

## 项目创新

1. **Verifier-First State Acquisition**：在 Teacher forward 前完成 K=2 状态分组与预算选择。
2. **Dual-Granularity Reliability Weighting**：组合 trajectory 级 verifier 权重和 token 级 entropy 权重。
3. **Cost-Aware Reproducible System**：实现 sparse logits、可恢复 shards、artifact lineage 与真实成本评测。

这些是本项目的算法组合与系统设计贡献，不声称每一点都是全球首次提出。

## 研究问题

在相同 Student rollout 和实际 Teacher token 预算下，Verifier-first 的状态选择是否比随机选择
和简单 verifier 过滤获得更好的数学推理质量—成本表现？

完整研究矩阵最终比较：

| 方法 | 作用 |
|---|---|
| Base | 原始 1.5B Student |
| SFT | 使用 verified solution 的监督微调 |
| Dense OPD B100 | 标注全部 states 的原始 OPD 成本上界 |
| Random-Budget B50 | 相同 Teacher-token 预算的随机选择对照 |
| Verifier-Filtered B50 | 只用 pass/unknown 过滤的强基线 |
| VFS-Weighted OPD B50 | VFS 选择与双粒度加权的主方法 |
| VFS-OPD B25 | 极低 Teacher-token 预算的成本曲线 |

所有训练使用唯一预注册 seed `42`。首轮先完成 Base、SFT 和 VFS-Weighted B50；Random-B50、
Dense-B100 与组件消融在主 pipeline 得到结果后补充。Entropy/verifier weighting 已有相关研究，
本项目强调其与 pre-Teacher state selection 的完整工程组合。完整文献边界见
[`docs/OPD_INNOVATION_REVIEW.md`](docs/OPD_INNOVATION_REVIEW.md)。

## 架构

```text
OpenR1-Math-220k
        │
        ▼
prepare + deduplicate + contamination audit
        │
        ▼
1.5B Student on-policy rollout × 2 (vLLM, resumable shards)
        │
        ▼
Math verifier + boundary/uncertain/solved/failed grouping
        │
        ▼
deterministic state selection under Teacher-token budget
        │
        ▼
7B Math Teacher forcing annotation for selected states
top-k log-probs + tail mass + token entropy
        │
        ▼
weighted training view
trajectory verifier weight + token entropy weight
        │
        ▼
QLoRA + validation early-stop + best checkpoint
        │
        ▼
MATH-500 / AIME / GPQA Diamond / IFEval
```

## 工程特性

- **可恢复任务**：Rollout 与 Teacher annotation 按配置 hash 分片；成功 shard 复用，失败 shard 重跑。
- **Artifact lineage**：Manifest 记录配置 hash、SHA-256、模型 revision、上游 artifact ID、记录数和资源指标。
- **评测数据导入**：将 LightEval/Hugging Face 导出的 benchmark JSONL/Parquet 标准化为统一 schema，并纳入污染审计 lineage。
- **Teacher 信号压缩**：只保存 top-k log-prob、tail mass 和 entropy，避免完整 vocabulary logits 的存储成本。
- **正确 token 对齐**：Teacher 与 Student 都使用同一 chat template；causal logits 从 `response_start - 1` 开始对齐。
- **Sparse KL**：精确计算 top-k 项，把其余 vocabulary 聚合为 tail bucket。
- **Sparse/full audit**：固定 seed 的 32 组数值审计比较 loss、梯度 cosine 与单步更新方向。
- **Verifier-first 选择**：Teacher forward 前完成状态分组和预算选择，避免先计算后丢弃。
- **公平预算**：按 estimated/actual Teacher tokens，而不是按记录数比较方法。
- **实验缓存复用**：正式消融可共享一次 Dense annotation，再按 selection manifest 构造子集；
  线上 VFS 路径仍只标注被选 states。
- **训练保护**：4-bit QLoRA、Accelerate、checkpoint resume、最多一轮 epoch、validation early-stop。
- **独立评测**：保存逐题输出、切片指标、paired bootstrap 和 exact McNemar 结果。
- **成本硬停止**：`opd budget` 汇总 GPU hours，达到 `60 H100h` 时阻止训练脚本继续启动。

## 快速开始

本地 CPU 不下载模型即可运行完整 mock pipeline：

```bash
uv sync --extra dev
make test
make smoke
```

`make smoke` 使用 24 条仓库内算术 fixture，执行：

```text
prepare → contamination audit → rollout → verify → select annotations
→ teacher annotate → build view → train → evaluate
→ checkpoint promotion → failure/cost report
```

结果写入 `artifacts/smoke/`。Mock checkpoint 只验证工程正确性，不能作为模型效果结果。

真实 tiny-model GPU smoke 使用固定 revision 的 Qwen2.5-0.5B Student 与 1.5B Teacher：

```bash
make tiny-gpu-smoke
```

该命令会实际下载模型、生成 rollout、计算 Teacher top-k、运行一个 QLoRA optimizer step 并评测；建议至少使用 24GB NVIDIA GPU。

## 正式运行

AutoDL 上先加载数据盘环境、安装锁定依赖并执行环境门禁：

```bash
source scripts/autodl_env.sh
scripts/remote_bootstrap.sh
make check
make smoke
make tiny-gpu-smoke
scripts/qwen_gpu_smoke.sh
```

`rtx-pro-6000-blackwell` 分支固定使用 PyTorch `2.7.1+cu128`、CUDA 12.8 和 vLLM
`0.10.1.1`，并强制验证单张 RTX PRO 6000 Blackwell 96GB、SM 12.0 与真实 BF16 kernels。
专用安装说明见 [`docs/RTX_PRO_6000_BLACKWELL.md`](docs/RTX_PRO_6000_BLACKWELL.md)。

门禁通过后，优先运行隔离的简历 MVP：

```bash
scripts/run_resume_mvp.sh
scripts/run_resume_sft.sh
```

MVP 输出统一写入 `artifacts/resume_mvp/`。完整命令、断点续跑和结果口径见
[`docs/RESUME_MVP.md`](docs/RESUME_MVP.md)。

`main` 分支的 `tiny-gpu-smoke` 至少需要 24GB NVIDIA GPU；本 Blackwell 分支的 bootstrap
只接受单张 RTX PRO 6000 96GB。MVP 准备 6,000 条候选题、rollout 前 3,000 条 prompt，
每题生成 2 个 rollout，每个 rollout 最多生成 4,096 response tokens。

### Phase B/C：MVP 完成后再运行

以下 15,000 candidates / 7,500 prompts 全矩阵命令不是 MVP 的后续必执行步骤。只有 Base、SFT、
VFS-Weighted 的结果和 artifact 已完整保存后，才按预算补 Random、Dense 与消融：

```bash
scripts/prepare_data.sh configs/data.yaml
scripts/fetch_eval_data.sh configs/main.yaml
scripts/audit_contamination.sh configs/main.yaml

uv run --no-sync opd audit sparse-kl \
  --output artifacts/audits/sparse_kl.json \
  --config configs/main.yaml
scripts/generate_rollouts.sh configs/rollout.yaml 0
scripts/verify.sh configs/main.yaml 0
scripts/build_vfs_experiment_data.sh

scripts/train.sh configs/sft.yaml
scripts/train.sh configs/dense_opd.yaml
scripts/train.sh configs/random_budget_b50.yaml
scripts/train.sh configs/verifier_filtered_b50.yaml
scripts/train.sh configs/vfs_b50.yaml
scripts/train.sh configs/vfs_b25.yaml
```

`configs/base.yaml` 已固定并静态核验模型、训练数据和评测数据的 commit SHA。远端 smoke
仍必须验证这些 revision 在实际 AutoDL 环境可下载、Student/Teacher tokenizer 指纹一致，且许可证
满足你的发布方式；任一验证失败都不得启动正式 rollout。正式生成在 800 条成功样本后自动检查
截断率，只有不高于 20% 才继续。所有预算方法读取同一批 rollout 和 verification，选择结果保存
为独立 manifest，禁止手工改选中的记录。

污染审计不会在 benchmark 文件缺失时静默跳过；必须先运行自动下载脚本。
`scripts/import_eval_data.sh` 仅作为无法访问 Hugging Face 时的离线备用入口。GPQA/IFEval
使用 LightEval 正式评测，不进入数学训练污染过滤。

正式评测前先把候选 PEFT adapter 合并为 vLLM checkpoint，例如 VFS-B50：

```bash
scripts/merge_checkpoint.sh \
  configs/vfs_b50.yaml \
  checkpoints/vfs_b50_seed42/best \
  artifacts/merged/vfs_b50
```

零基础逐步操作见 [`docs/BEGINNER_EXPERIMENT_GUIDE.md`](docs/BEGINNER_EXPERIMENT_GUIDE.md)；
精简版远端命令、停止条件和上传策略见 [`docs/REMOTE_RUNBOOK.md`](docs/REMOTE_RUNBOOK.md)。

## 评测

内部回归入口：

```bash
uv run --no-sync opd evaluate \
  --suite regression \
  --checkpoint CHECKPOINT \
  --config configs/main.yaml
```

标准 benchmark 入口：

```bash
scripts/evaluate_lighteval.sh CHECKPOINT artifacts/lighteval/RUN_NAME
```

LoRA 模型必须先用 `scripts/merge_checkpoint.sh` 合并，Base 则可直接传配置中的 Student，包装器会自动使用固定 revision。脚本通过 `opd benchmark run` 调用 LightEval，并自动写入命令、job metrics 和 checksum manifest。任务注册表位于 [`eval/registry.yaml`](eval/registry.yaml)。项目锁定 `lighteval==0.9.2`；正式运行前仍需核对 GPQA 访问权限。若设置 `LIGHEVAL_TASKS`，其值应是逗号分隔的注册表名称，例如 `math500,aime2024`。

当前代码已通过本地 CPU 单元测试、静态检查和 mock 端到端链路；本机没有 NVIDIA GPU，
因此真实 QLoRA、merge、vLLM 与 LightEval 成功只能由上述 AutoDL 两级 smoke 确认。详细审计见
[`docs/AUTODL_VALIDATION.md`](docs/AUTODL_VALIDATION.md)。

逐题结果可比较：

```bash
uv run --no-sync opd compare \
  --baseline artifacts/evaluation/base/math500/summary.json \
  --candidate artifacts/evaluation/weighted/math500/summary.json \
  --output reports/weighted_vs_base_math500.json \
  --config configs/main.yaml
```

失败案例和质量—成本报告：

```bash
uv run --no-sync opd report failures \
  --predictions PREDICTIONS.jsonl \
  --output reports/failures.json \
  --config configs/main.yaml
uv run --no-sync opd report cost \
  --output-dir reports/quality_cost \
  --config configs/main.yaml
```

`report cost` 根据 `report.runs` 汇总实际 job metrics，并生成 JSON、Markdown 与可直接嵌入 README 的 SVG Pareto 图。

## Early Stop

训练默认每 200 optimizer steps 评测一次，使用 `0.8 × 数学准确率 + 0.1 × 答案格式通过率 + 0.1 × 轻量指令回归` 的预注册复合分数；连续三次提升小于 `0.005` 时停止，并保留 best checkpoint。Train loss 下降不能单独证明模型变好。

以下情况必须暂停任务：NaN/Inf、未恢复 OOM、validation 明显下降、GPQA/IFEval 回退超过 2 个百分点、平均输出长度变化超过 30%，或 entropy/重复率异常坍缩。

## 单 Seed 限制

为节省 GPU，本项目只执行一个训练 seed：`42`。Shard 派生 seed 只是避免各 shard 使用相同随机流，不是独立训练重复。

Paired bootstrap 衡量 benchmark 题目抽样不确定性，**不能估计训练 seed 方差**。最终报告必须表述为“在 seed 42 和当前预算下观察到……”，不能声称跨 seed 稳定。

## 目录

```text
configs/       单 seed 的数据、训练和 Round 1 配置
src/opd/       数据、rollout、Teacher、Verifier、训练、评测代码
tests/         单元测试与完整 CPU smoke 测试
scripts/       本地和远端阶段任务入口
eval/          标准 benchmark 注册表
docs/          远端运行与面试演示手册
plans/         推荐版和研究扩展版备用方案
```

## 质量门禁

```bash
make test
make lint
make typecheck
make smoke
```

当前 CPU 门禁覆盖 schema、mask/EOS/截断、sparse/full KL 数值审计、sample 权重、Tokenizer 指纹、Verifier 三态、confidence 归一化、artifact 原子写入、累计成本账单、shard resume、固定 seed、paired bootstrap、McNemar、checkpoint promotion 和端到端流水线。

## 项目卡片

- [`DATA_CARD.md`](DATA_CARD.md)：数据来源、过滤、污染和限制。
- [`MODEL_CARD.md`](MODEL_CARD.md)：方法、评测和单 seed 声明模板。
- [`docs/INTERVIEW_DEMO.md`](docs/INTERVIEW_DEMO.md)：五分钟面试演示流程。
