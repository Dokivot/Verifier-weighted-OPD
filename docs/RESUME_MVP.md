# OPD-Lab 简历 MVP 执行方案（历史 Qwen2.5 流程）

> **历史方案，不再用于新实验。** 当前推荐的 Qwen3 sampled-token K2 reverse-KL 方案见
> [`../plans/PROJECT_PLAN_REVERSE_KL.md`](../plans/PROJECT_PLAN_REVERSE_KL.md)。本文件仅用于解释或恢复
> 已产生的 Qwen2.5 artifact。

本方案的目标是先得到一条完整、可复现、可展示的真实 GPU 结果，再补公平基线和消融。它不是
最终论文实验矩阵，但足以形成第一版简历项目和面试演示。

## 1. 第一版要证明什么

使用固定 seed `42`，在 3,000 道数学题上让 `Qwen2.5-1.5B-Instruct` 每题生成两个 on-policy
rollout。先运行数学 verifier，再在 Dense annotation 估算量的 50% Teacher-token 预算内选择
states，只让固定 revision 的 `Qwen2.5-Math-7B-Instruct` 标注被选 states。最后使用
sample-level verifier 权重和 token-level Teacher entropy 权重执行 sparse-KL QLoRA。

第一版只要求完成三个模型：

| 模型 | 必要性 | 作用 |
|---|---:|---|
| Base Student | 必做 | 训练前能力基线 |
| VFS-Weighted OPD B50 | 必做 | 完整创新 pipeline |
| SFT | 必做 | 判断提升是否只是普通监督微调带来的 |

MVP 正式评测固定为 MATH-500、AIME 2024 和 IFEval。MATH-500 是主指标；AIME 检查高难数学；
IFEval 检查数学训练是否破坏通用指令遵循。所有方法保存逐题 details，不能只保留总分。

## 2. 三个简历创新点

### 2.1 Verifier-First State Acquisition

传统 OPD 往往先对所有 Student states 做昂贵 Teacher forward，再决定哪些样本值得训练。本项目
把数学验证和同题多 rollout 分组移动到 Teacher forward 之前，将状态分为
`boundary/uncertain/solved/failed`，并在固定 Teacher-token 预算内确定性选择 states。

### 2.2 Dual-Granularity Reliability Weighting

选中数据后，训练同时使用两种可靠性：sample-level verifier 权重控制整条 trajectory 的可信度，
token-level Teacher entropy 权重降低高不确定 token 的影响。相关思想已有论文覆盖，因此简历中应
称为“组合并工程化双粒度加权”，不称为全球首次提出。

### 2.3 Cost-Aware Reproducible OPD System

系统保存 top-k logits、tail mass 和 entropy，不缓存完整词表分布；rollout、Teacher annotation
均支持 hash 分片恢复；manifest 记录模型/data revision、上游 lineage、checksum、Teacher tokens
和 GPU hours。最终同时报告模型质量与成本，而不是只报告 accuracy。

建议使用“项目创新”或“系统设计贡献”。在消融未完成前，不声称每个组件独立有效。

## 3. 固定配置

- 主配置：`configs/vfs_weighted_mvp.yaml`
- SFT 配置：`configs/resume_mvp_sft.yaml`
- Student：`Qwen/Qwen2.5-1.5B-Instruct`
- Teacher：`Qwen/Qwen2.5-Math-7B-Instruct`
- 数据：`open-r1/OpenR1-Math-220k`
- Prompt：3,000
- Rollout：每题 `K=2`，8,192 context，最多 4,096 response tokens
- Teacher/训练：最多保留 prompt + response 的前 4,096 tokens
- Teacher budget：Dense 估算 tokens 的 50%
- OPD optimizer steps：最多 300，每 100 steps 验证并保存
- 训练 seed：仅 `42`

所有 MVP artifact 都写入 `artifacts/resume_mvp/`，不会与早期 Qwen3、7.5k 完整实验或 smoke
结果混用。Teacher revision 已固定，正式运行时仍要在 manifest 中核对实际 revision。

## 4. AutoDL 上的最快执行顺序

进入仓库和 tmux 后：

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
git pull --ff-only
make check
make smoke
scripts/qwen_gpu_smoke.sh
```

GPU smoke 通过后，运行主方法：

```bash
set -o pipefail
scripts/run_resume_mvp.sh
```

该脚本依次执行：prepare、benchmark 数据下载、污染审计、K=2 rollout、verify、VFS selection、
仅对 selected states 做 Teacher annotation、构建 weighted view、训练、merge、内部 regression、
主方法 benchmark 和 Base benchmark。每次运行使用新的时间戳日志目录。

若在某阶段失败，先保留日志并修复原因，再从该阶段继续。例如 Teacher annotation 前已完成选择：

```bash
START_STAGE=13 scripts/run_resume_mvp.sh
```

阶段号为 `0/1/2/10/11/12/13/14/20/21/22/23/24`。已完成 rollout 和 annotation 的 hash shard
会复用。训练若在 stage 20 中断，使用：

```bash
START_STAGE=20 \
RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/vfs_weighted_b50_seed42/latest \
scripts/run_resume_mvp.sh
```

不能仅依赖 `START_STAGE`，否则训练会从头初始化。即使数据不足以到达第一个 100-step 验证点，
训练结束时也会补做最终 validation 并保存 `best/`。

主方法得到完整 benchmark 后，再运行 SFT：

```bash
scripts/run_resume_sft.sh
```

SFT 训练中断时同样设置 `RESUME_FROM_CHECKPOINT=artifacts/resume_mvp/checkpoints/sft_seed42/latest`；
可用恢复 stage 为 `30/31/32/33`。

若希望先快速看到数学趋势，可临时只执行 MATH-500，但最终 MVP 必须补齐三个任务：

```bash
LIGHEVAL_TASKS=math500 scripts/run_resume_mvp.sh
```

## 5. 每个阶段必须检查什么

1. Rollout：`quality_gate.json` 最终为 `passed`，截断率不高于 20%。
2. Selection：`selected_estimated_teacher_tokens <= teacher_token_budget`。
3. Teacher：成功记录数与 selected records 一致，没有静默失败。
4. Training：无 NaN/Inf，保留 `training_summary.json`、best/latest 和 job metrics。
5. Benchmark：保留 results、逐题 details、`command.json`、manifest 和 job metrics。
6. 成本：记录实际 Teacher tokens 和累计 GPU hours，不用估算值替代最终成本。

实时监控：

```bash
watch -n 30 'cat artifacts/resume_mvp/data/rollouts/round_0/quality_gate.json'
watch -n 30 nvidia-smi
df -h /root/autodl-tmp
```

## 6. 首轮结果如何写

首轮主表：

| 方法 | MATH-500 | AIME 2024 | IFEval | Teacher tokens | GPU hours |
|---|---:|---:|---:|---:|---:|
| Base | 实测 | 实测 | 实测 | 0 | 实测 |
| SFT | 实测 | 实测 | 实测 | 0 | 实测 |
| VFS-Weighted B50 | 实测 | 实测 | 实测 | 实测 | 实测 |

在 Random-B50 尚未完成时，只能写“相对 Base/SFT 的观察结果”和系统贡献，不能写“VFS selection
优于随机选择”。同理，在去掉 verifier/entropy 的消融完成前，不能声称双粒度加权是提升来源。

## 7. 后续实验优先级

### Phase B：补齐核心证据

1. Random-Budget B50：与主方法使用相同 Teacher-token 预算，是验证 VFS selection 的必要对照。
2. Dense Vanilla OPD B100：给出原始 OPD 的质量上界和成本上界。

完成 Phase B 后，项目才能严谨回答“状态选择是否提高 Teacher-token 效率”。

### Phase C：消融与扩展

按价值排序：VFS 去掉 entropy weighting、Verifier-filtered B50、VFS B25、confidence-only、K=1/K=4、
Round 1。单 seed `42` 保持不变；如未来预算允许，多 seed 的优先级高于继续堆叠新组件。

## 8. 简历表述模板

在填入真实数字后使用：

> 构建面向数学推理的工程级 On-Policy Distillation 系统：对 1.5B Student 的同题多 rollout 先
> 执行 verifier 状态分组，再在固定 Teacher-token 预算内选择性调用 7B Math Teacher；组合
> trajectory 级验证权重与 token 级 entropy 权重，并实现 sparse-logit 缓存、可恢复分片、artifact
> lineage 和质量—成本评测。在 seed 42、3,000 prompts 下，MATH-500 相对 Base 提升 X.X 个点，
> Teacher annotation 使用 Y tokens / Z GPU 小时。

未测得的数字不得填入，单 seed 结果不得表述为跨 seed 稳定结论。
