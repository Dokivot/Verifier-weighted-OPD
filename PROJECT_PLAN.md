# OPD-Lab：Verifier-First State-Budgeted OPD

> 当前执行版本。固定单 seed `42`，使用 `Qwen2.5-1.5B-Instruct` 作为 Student、
> `Qwen2.5-Math-7B-Instruct` 作为 Teacher，在单张 RTX PRO 6000 Blackwell 96GB 和 200GB
> 数据盘条件下完成一条
> 可复现、可恢复、可评测的工程级 On-Policy Distillation（OPD）链路。主方法为
> VFS-Weighted OPD：Verifier-First 状态选择与双粒度可靠性加权。

旧的 Qwen3-8B/14B 大规模设计仍保存在 [`plans/`](plans/README.md)，仅作为备用升级方向，
不与本方案的 artifact 混用。

## 1. 先看结论

### 1.1 研究问题

在相同 Student rollout 和实际 Teacher token 预算下，先执行数学验证、再按同题多 rollout
状态选择 Teacher annotation，是否比随机预算选择和简单 verifier 过滤取得更好的质量—成本结果？

### 1.2 为什么选择方案一

- `Qwen2.5-1.5B-Instruct` 比当前的 Qwen3-8B 更弱，MATH-500 不容易出现“基线已经接近上限”的问题；
- `Qwen2.5-Math-7B-Instruct` 是同一 Qwen2.5 tokenizer 家族中的数学 Teacher，Teacher 与 Student
  的 response token 可以可靠对齐；
- Qwen2.5 没有 Qwen3 thinking/non-thinking 切换带来的长思考输出不确定性，便于控制 rollout 截断率；
- 1.5B Student 的 QLoRA 训练和 7B Teacher annotation 可在单张 RTX PRO 6000 Blackwell
  96GB 上保留充足余量，适合 200GB 数据盘；
- Student 足够小，仍保留真正的 on-policy rollout、teacher-forcing、sparse KL、Verifier、训练和
  benchmark 评测，不是仅在 `GSM8K` 上跑一个 toy demo。

### 1.3 重要边界

本方案只使用一个训练 seed。paired bootstrap 和 McNemar test 只能描述 benchmark 题目层面的
不确定性，不能证明跨 seed 的训练稳定性；最终报告必须明确这个限制。

本项目不把 entropy weighting、verifier gating 或 selective KD 声称为原创。这些方向已有
Entropy-Aware OPD、RG-OPD/OPDVR 和 Selective KD。完整检索与差异见
[`docs/OPD_INNOVATION_REVIEW.md`](docs/OPD_INNOVATION_REVIEW.md)。

### 1.4 三个项目创新点

1. **Verifier-First State Acquisition**：Teacher forward 前完成同题多 rollout 分组与预算选择；
2. **Dual-Granularity Reliability Weighting**：组合 trajectory verifier 权重与 token entropy 权重；
3. **Cost-Aware Reproducible OPD System**：sparse logits、可恢复 shards、artifact lineage、真实
   Teacher tokens/GPU hours 与质量—成本评测。

第一项是主要研究假设，后两项是已有思想的工程化组合与系统贡献。简历使用“项目创新”或
“系统设计贡献”，不使用“全球首次提出”。

## 2. 固定模型与版本

| 角色 | 模型 | 用途 |
|---|---|---|
| Student | `Qwen/Qwen2.5-1.5B-Instruct` | rollout、QLoRA、最终评测 |
| Teacher | `Qwen/Qwen2.5-Math-7B-Instruct` | 在 Student response 上 teacher-forcing annotation |
| 开发 Student | `Qwen/Qwen2.5-0.5B-Instruct` | tiny GPU smoke，不进入最终结果 |
| 开发 Teacher | `Qwen/Qwen2.5-1.5B-Instruct` | tiny GPU smoke，不进入最终结果 |

Student revision 已固定为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。Teacher revision 已固定为
`ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`。AutoDL smoke 必须确认两个 revision 可下载且
Student/Teacher tokenizer fingerprint 一致；不一致时不得开始正式 rollout。

训练使用 4-bit QLoRA，计算 dtype 优先 `bfloat16`。Student rollout 使用 vLLM，Teacher
annotation 使用 Transformers teacher-forcing；两者分成不同 job，不让 Teacher 参与反向传播。

## 3. 数据方案

### 3.1 训练数据

主数据集是 Hugging Face `open-r1/OpenR1-Math-220k` 的 `default`/`train`，使用配置中固定的
dataset revision。直接复用其中的题目、参考答案和 solution；不需要手工合成训练题。

字段用途：

| 字段 | 用途 |
|---|---|
| `problem` | Student rollout prompt |
| `solution` | SFT target 或质量分析 |
| `answer` | Math-Verify 的标准答案 |
| `problem_type`、`level` | 分题型和难度统计 |
| `source` | 数据溯源、污染审计和失败分析 |

### 3.2 固定规模

| Split | 数量 | 用途 |
|---|---:|---|
| smoke | 128 | CPU/tiny/GPU 完整链路检查 |
| validation | 1,000 | early-stop、checkpoint 选择和回归 |
| train candidate | 15,000 | 清洗、去重、污染审计后的候选池 |
| Round 0 prompt | 7,500 | 所有主实验共享，每题生成 2 个 on-policy states |
| Round 1 prompt | 7,500 | 由 Round 0 最佳 Student 重新生成 |

脚本通过固定 seed 先打乱再切分；正式 rollout 只取清洗后 train 的前 7,500 条。所有方法
共享同一批 prompt、同一 Student revision 和同一采样配置，不能为某个方法单独增加数据。
Phase A 的隔离配置把 candidate pool 缩为 6,000、rollout 缩为 3,000；上述 15k/7.5k 数字仅用于
Phase B/C 完整矩阵。

### 3.3 数据污染

在 rollout 前执行 MATH-500、AIME 2024、AIME 2025 的 exact hash 和近重复审计。疑似重复
样本写入 quarantine 并记录数量、阈值和来源，不静默删除。benchmark 只用于最终评测，不能
用于训练或反复调参。

### 3.4 是否合成数据

不合成新题目。OPD 必须自动产生两类运行时数据：

1. 当前 Student 的 on-policy response；
2. Teacher 对这些 response 的 token 分布、entropy 和 top-k log-prob。

这两类数据由脚本生成并写入带 manifest 的 Parquet；它们不是人工合成训练集。

## 4. 端到端数据流

```text
OpenR1-Math-220k
        │
        ▼
prepare → split → decontaminate
        │
        ▼
15k clean candidate / fixed 7.5k prompt manifest
        │
        ▼
Round 0: 1.5B Student rollout × 2
        │
        ▼
Math-Verify + 同题状态分组
        │
        ▼
fixed Teacher-token budget selection
        │
        ▼
7B Math Teacher teacher-forcing annotation
                    │
                    ▼
       top-k sparse distribution + entropy
                    │
                    ▼
          training views → SFT / OPD methods
                    │
                    ▼
          QLoRA train → validation → best checkpoint
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Round 0 benchmark       Round 1 rollout/training
```

每个阶段读取上游 manifest，而不是仅根据目录里“存在某个文件”来判断输入。大文件先写
`.partial`，通过 schema、数量和 checksum 后再原子提交；成功 shard 重跑时直接复用。

## 5. OPD 方法与分阶段实验

### 5.1 Phase A：先获得完整简历结果

使用 `configs/vfs_weighted_mvp.yaml` 在 3,000 prompts 上完成 Base、SFT 和
**VFS-Weighted OPD B50**。主方法先做 VFS selection，再同时使用 verifier sample weight 与
Teacher entropy token weight，最多训练 300 optimizer steps。三个模型统一评测 MATH-500、
AIME 2024 和 IFEval。该阶段的目标是尽快验证真实 pipeline、获得可量化结果和完整 artifact，
不是证明每个组件的独立贡献。

### 5.2 Phase B/C：核心基线与消融

完整矩阵如下：

| ID | 方法 | Teacher 预算 | 回答的问题 |
|---|---|---:|---|
| E0 | Base | 0 | 原始 1.5B Student 能力 |
| E1 | SFT | 0 Teacher forward | 标准监督基线 |
| E2 | Dense Vanilla OPD | B100 | 原始 OPD 的质量和成本上界 |
| E3 | Random-Budget OPD | B50 | 只减少预算会损失多少 |
| E4 | Verifier-Filtered OPD | B50 | 简单 pass/unknown 过滤是否足够 |
| E5 | **VFS-Weighted OPD** | **B50** | 状态选择与双粒度加权的整体效果 |
| E6 | VFS-OPD | B25 | 极低预算下是否仍有收益 |

全部训练使用 seed `42`。Phase B 优先补 E3 Random-Budget B50 与 E2 Dense B100；只有 E3
完成后才能声称 VFS selection 比随机选择更有效。Phase C 再补 E4、E6、去掉 entropy/verifier
权重、K 值和 Round 1。E2 是高成本上界，不是主对手。

### 5.3 Vanilla OPD

Student 先生成 `response`。Teacher 对 `prompt + response` 执行 teacher-forcing，在每个
response token 位置产生分布。训练时只对 response、非 padding、非异常截断 token 计算 loss，
所有有效 token 等权。

### 5.4 VFS-Weighted OPD

每题的两个 rollout 按 verification 结果组成四类状态：

| 状态组 | 定义 | 直觉 |
|---|---|---|
| `boundary` | pass 与 non-pass 并存 | Student 决策边界不稳定 |
| `uncertain` | 无 pass，但答案不同或有 unknown | 值得 Teacher 纠正 |
| `solved` | 两个都 pass | 已掌握，优先级较低 |
| `failed` | 两个都 fail 且答案一致 | 可能过难或有系统错误 |

选择顺序固定为：非截断优先；`pass > unknown > fail`；短 Teacher context 优先；最后使用由
seed `42` 派生的稳定 hash。subject/difficulty 间做 round-robin，累计 estimated Teacher tokens
达到 B50/B25 后停止。选择规则在 benchmark 前冻结，并保存逐条选择原因和预算前后余额。

MVP 在 VFS 选中 states 上使用 verifier sample weight 与 Teacher confidence/entropy token weight。
双粒度加权是集成贡献，不声称是全新算法；Phase C 通过去掉 entropy 权重和去掉 verifier 权重
判断各组件贡献。它们发生在 Teacher forward 后，不能替代 VFS 在 annotation 前减少 Teacher 调用。

Teacher 主存储使用 top-64 token ids、top-64 log-probs、tail mass、token entropy 和 response
mask。它是 `top-k sparse/approximate KL`，不是无损 full-vocabulary KL。使用 32–64 条样本做
full/sparse KL、梯度 cosine 和单步更新方向审计。

## 6. Rollout 长度与质量门禁

主正式 rollout 使用 `max_new_tokens: 4096`、`temperature: 0.6`、`top_p: 0.95`、每题两个
sample。这个上限比 1536/3072 更适合多步数学题，同时比无条件 8192 更节省显存、磁盘和
Teacher token 预算。两个 sample 是状态分组所需的最低 K 值，不扩展到 K=4。

开始正式生成后，至少完成 800 条成功样本才判断质量门禁：

- 截断定义为 `response_tokens >= max_new_tokens`；
- 截断率必须不高于 20%；
- 若失败，停止扩展，不要直接删除统计或盲目把上限改成 8192；
- 先检查 chat template、EOS、thinking 控制和 prompt/response 统计，再决定是否提高上限；
- 若确有大量正确答案在 4096 处结束，可单独做 8192 成本对照，但不能替换主结果而不记录。

每个 shard 记录成功、失败、empty、timeout、OOM 和 truncated 状态，实时写入
`quality_gate.json`，可以在另一个终端查看当前截断率。

## 7. 评测与结果判定

### 7.1 三层评测

1. **Smoke**：4–128 条，验证 schema、mask、推理、checkpoint 和报告链路；不作为效果结论。
2. **Validation/regression**：1,000 条固定 validation，用于 early-stop 和 checkpoint promotion。
3. **MVP benchmark**：先对 Base、SFT、VFS-Weighted B50 跑 MATH-500、AIME 2024、IFEval；
4. **完整研究 benchmark**：Phase B 对 Random B50、Dense B100 跑同一矩阵，Phase C 再扩展。

### 7.2 标准 benchmark

| Benchmark | 作用 | 指标 |
|---|---|---|
| MATH-500 | 主数学指标 | accuracy，附 level/subject 切片 |
| AIME 2024 | 高难数学 | accuracy/pass@1 |
| GPQA Diamond | 分布外科学推理 | accuracy |
| IFEval | 通用指令回归 | strict/loose accuracy |

所有模型使用相同 revision、chat template、greedy 解码（`temperature=0`、`num_samples=1`）、
answer extractor、benchmark revision 和 LightEval 版本。AIME 可额外记录 temperature 0.7
的 pass@8，但不能用它替代主表的 pass@1。

### 7.3 最终比较

主比较是 `VFS B50 vs Random B50`；辅助比较是 `VFS B50 vs Verifier-filtered B50`、
`VFS B50 vs Dense B100` 和 `Dense B100 vs Base/SFT`。报告每题 prediction、正确率、paired
bootstrap 95% CI、MATH-500 win/tie/loss、McNemar exact test、难度/题型/response 长度切片、
actual Teacher tokens、annotation GPU hours 和每提升 1 个百分点所需 Teacher tokens。

项目不要求一定超过公开 SOTA。若提升不显著，结论写为“在单 seed、当前数据量和预算下未观察到
可靠收益”，并分析 verifier pass rate、Teacher entropy、截断率和训练曲线，负结果同样可作为
简历项目的工程结论。

## 8. Early-stop 与停止规则

完整配置每 200 optimizer steps、MVP 每 100 steps 在固定 validation 上评估一次：

```text
composite = 0.80 × math_accuracy
          + 0.10 × format_pass_rate
          + 0.10 × instruction_regression_score
```

连续 3 次提升小于 `0.005` 时停止，恢复该轮 best checkpoint。train loss 只能作为诊断，不能
单独决定继续训练。每轮最多一个 epoch，当前配置最多 600 steps。

出现以下任一情况立即暂停并保留日志：NaN/Inf、未恢复 OOM、gradient norm 异常、validation
明显下降、IFEval/GPQA 回退超过 2 个百分点、平均输出长度变化超过 30%、空回答/重复率异常、
Teacher entropy 坍缩或质量门禁失败。

只有 Phase A/B 已完成、VFS-Weighted B50 至少不差于 SFT 且剩余预算充足时才进入 Round 1。
若 Round 1 validation 提升小于 0.5 个百分点或单位 Teacher token 收益变差，停止，不运行 Round 2。

## 9. 技术栈与代码边界

| 功能 | 技术 |
|---|---|
| Python/依赖 | Python 3.11、`uv`、锁定 `uv.lock` |
| 训练 | PyTorch、Transformers、PEFT、bitsandbytes、Accelerate |
| rollout | vLLM |
| 数据 | Datasets、PyArrow、Parquet |
| verifier | Math-Verify |
| benchmark | LightEval |
| 配置/Schema | YAML deep-merge、Pydantic |
| 测试/质量 | pytest、ruff、mypy |
| 记录 | manifest、JSON/Parquet、可选 W&B |

第一版不引入 Ray、Kubernetes、Airflow、数据库或微服务。阶段式 CLI + shell 脚本已经足以
支持 AutoDL 的单卡任务；只有需要多个并行 worker 时才考虑调度器。

## 10. Artifact 与目录约定

```text
artifacts/
├── data/raw/                 # 原始快照（只读）
├── data/curated/             # 固定 split
├── data/contamination/       # clean/quarantine/report
├── data/rollouts/round_N/    # Student rollout shards
├── data/verifications/       # verifier 结果
├── data/annotation_selection/# 选择决策、预算报告与 manifest
├── data/annotations/round_N/ # Teacher sparse annotation
├── data/training_views/      # 各方法训练视图
├── checkpoints/              # adapter、best、latest、merged
├── evaluation/               # regression 与内部评测
├── lighteval/                # 标准 benchmark 详情和结果
└── reports/                  # 汇总表、图、失败案例
```

每个 artifact 的 manifest 至少记录：Git commit、配置 hash、上游 artifact id、模型和 tokenizer
revision、tokenizer fingerprint、数据 revision、记录数、失败数、checksum、GPU 型号、wall time、
Teacher tokens 和 job metrics。Git 只提交代码、配置、schema 和小 fixture，不提交模型、Parquet、
checkpoint 或日志。

由于此前可能存在 Qwen3 旧结果，切换方案前应先把旧的 `artifacts/data/`、`checkpoints/` 和
`logs/` 复制到带日期的备份目录或对象存储。新的 manifest 必须使用新模型 revision，禁止把旧
Qwen3 rollout 与方案一 Teacher annotation 拼接。

## 11. GPU、磁盘与时间预算

配置中的目标为 `40 GPU hours`，硬上限为 `60 GPU hours`。下表沿用原 H100 粗估，只用于租卡
预算上界，不代表 RTX PRO 6000 的实测速度；实际报告必须使用本次 job metrics 记录的 GPU 型号、
wall time 和累计 GPU hours。每个脚本都会在达到 hard cap 时阻止新任务。

| 阶段 | 原 H100 粗估 GPU 小时 |
|---|---:|
| tiny/Qwen smoke 与稽核 | 2–5 |
| Phase A 3k Student rollout（K=2） | 2–5 |
| Phase A VFS-B50 Teacher annotation | 2–6 |
| Phase A SFT + VFS-Weighted 训练 | 3–7 |
| Round 0 内部评测与 promotion | 2–4 |
| Final benchmark、报告和重试余量 | 5–10 |
| **Phase A 估计合计** | **16–37** |
| 可选 Phase B / Round 1 | 不计入 Phase A，另行实测 |

若接近 `60`，按以下顺序降本：先取消 Round 1，再把 E4/E6 限制为 MATH-500，最后缩小 prompt
数量；不删除 Base、SFT、Dense、Random-B50、VFS-B50、validation 和逐题结果。正式运行前保留至少
30GB 可用磁盘。200GB 数据盘上只保留当前 merged checkpoint、必要 shard、manifest 和日志；
成功生成的临时模型缓存和旧 benchmark 详情可以在校验 checksum 后清理。

## 12. AutoDL 执行顺序

本分支必须使用 [`docs/RTX_PRO_6000_BLACKWELL.md`](docs/RTX_PRO_6000_BLACKWELL.md) 中的
CUDA 12.8 安装方案，不得复用 A800 分支的 CUDA 12.6 虚拟环境。

```bash
source scripts/autodl_env.sh
scripts/remote_bootstrap.sh
make check
make smoke
make tiny-gpu-smoke
scripts/qwen_gpu_smoke.sh
```

Qwen smoke 通过后，检查固定 Teacher revision、tokenizer fingerprint 和截断率。然后优先运行
简历 MVP：

```bash
scripts/run_resume_mvp.sh
scripts/run_resume_sft.sh
```

详细断点续跑与验收见 [`docs/RESUME_MVP.md`](docs/RESUME_MVP.md)。以下 7.5k 多方法命令属于
Phase B/C，不应在 MVP 出结果前启动：

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

`build_vfs_experiment_data.sh` 先生成五份不可变 selection manifest，再执行一次 Dense Teacher
annotation，最后按选择集合构建训练视图。Teacher-forcing 是确定性的，因此共享 Dense 缓存不会
改变任何方法的训练数据；成本报告按各 selection 中实际 `teacher_tokens` 求和。训练完成后先运行
regression 和 promotion，再对候选 checkpoint 做 LightEval。

## 13. 实施里程碑与验收

### M0：工程门禁

`make check`、CPU smoke、配置解析、manifest、mask、sparse KL 和 shard resume 全部通过。

### M1：模型 smoke

tiny GPU smoke 和方案一 Qwen smoke 成功；能下载两模型，tokenizer fingerprint 一致，能完成
一次 rollout、annotation、QLoRA step、merge 和 MATH smoke。

### M2：简历 MVP 数据

固定 3,000 prompt、每题 K=2 的 rollout、verification、selection、annotation 和 weighted view
完成；所有 shard 可恢复，截断率不高于 20%，每个方法的 Teacher tokens 和选择原因可统计。

### M3：Round 0 训练

Base、SFT、VFS-Weighted B50 完成统一 benchmark 并能恢复 best checkpoint；validation 曲线、
early-stop、GPU hours 和 artifact lineage 完整。随后再进入 Random-B50 与 Dense-B100。

### M4：最终评测与可选 Round 1

先完成统一 benchmark、逐题预测、统计比较、失败案例和质量—成本 Pareto 图。只有主比较已经
完成且预算充足时，才对最佳候选执行 Round 1。

### M5：可交付报告

报告必须包含模型/数据版本、训练配置、单 seed 限制、截断率、Verifier 分布、Teacher entropy、
各方法分数、成本、失败案例、已知限制和复现实验命令。

## 14. 简历项目表述

> 设计 Verifier-First State-Budgeted OPD：对 1.5B Student 的同题多 rollout 先执行数学验证
> 与状态分组，再在固定 Teacher-token 预算内选择性调用 7B Math Teacher；实现确定性 selection
> manifest、可恢复 sparse-logit annotation、QLoRA 和质量—成本 Pareto 评测，并在相同 Teacher
> tokens 下对比 random-budget、verifier-filtered 与 dense OPD。

面试时如实说明：方案一主动选择较弱 Student 是为了避免基线饱和，并不是声称 1.5B 模型达到
生产级数学能力；项目贡献是可审计的 OPD 工程链路和受控实验结论。
