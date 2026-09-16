# OPD-Lab 最小可信执行方案

> 当前执行版本。目标是在 **60–110 H100 GPU 小时**内，使用固定单 seed 完成一个具备完整数据、训练、评测和复现链路的 8B OPD 简历项目。

备用方案：

- [推荐完整版：300–500 H100 GPU 小时](plans/PROJECT_PLAN_RECOMMENDED.md)
- [研究扩展版：600–1000 H100 GPU 小时](plans/PROJECT_PLAN_RESEARCH_SCALE.md)
- [方案选择和升级条件](plans/README.md)

## 1. 项目要解决什么问题

### 1.1 项目名称

**Budget-Aware Verifier- and Confidence-Weighted On-Policy Distillation for an 8B Reasoning LLM**

中文名称：**面向 8B 推理模型的预算感知、验证器与置信度加权在线策略蒸馏**。

### 1.2 核心问题

> 在相同 Student、训练 token 和 Teacher annotation 预算下，使用数学 Verifier 过滤样本，并根据 Teacher 置信度调整 token loss，能否比 SFT 和等权 Vanilla OPD 获得更好的数学推理质量—成本表现？

### 1.3 为什么它不是 toy project

即使控制 GPU 规模，项目仍然保留完整的工程和实验链路：

- 使用 8B Student 和 14B Teacher；
- 准备 15k 真实数学问题候选池，并在固定 token 预算下使用其中 7.5k 条正式训练；
- rollout 来自当前 Student，符合 on-policy 定义；
- Teacher 在 Student response 上执行 teacher-forcing annotation；
- 使用确定性的 Math-Verify，而非只依赖 LLM judge；
- 比较 Base、SFT、Vanilla OPD 和改进 OPD；
- 所有方法使用同一个预先固定的 seed，并通过逐题配对统计、多个 benchmark 和完整训练曲线增强结论可信度；
- 使用独立 validation early-stop 和正式 benchmark；
- 记录 Teacher tokens、GPU hours、数据 lineage 和失败案例；
- 所有大任务支持分片、恢复和重复执行。

## 2. 控制范围

### 2.1 当前只做数学推理

第一版不加入代码、工具调用和通用对话训练。数学任务具有现成训练集、标准答案和自动 verifier，可以用较少 GPU 时间形成可信结论。

项目仍然使用 `GPQA Diamond` 和 `IFEval` 做外部分布和通用能力回归，检查数学训练是否伤害其他能力。

### 2.2 当前不做的内容

- 不自行生成新题目；
- 不训练 Reward Model；
- 不运行 32B Teacher；
- 不做 60k–100k prompts；
- 不运行多 seed 大规模复现实验；
- 不做超过 2 轮 OPD；
- 不引入 Kubernetes、Ray、Airflow 或微服务；
- 不做全参数 8B 训练；
- 不使用完整 vocabulary logits 作为主存储格式。

这些内容已经保存在备用方案中，当前阶段不消耗算力实现。

## 3. 固定模型配置

### 3.1 主模型

| 角色 | 配置 | 用途 |
|---|---|---|
| Student | `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218` | rollout、QLoRA 训练和最终评测 |
| Teacher | `Qwen/Qwen3-14B`，revision `40c069824f4251a91eefaf281ebe4c544efd3e18` | Teacher-forcing annotation |

模型 repository 与 revision 已在 `configs/base.yaml` 固定，并通过 Hugging Face 元数据静态核验。
首次远端 smoke 仍要确认 revision 可实际下载，并以 tokenizer fingerprint 确认 Student 与 Teacher
使用相同 vocabulary；未通过时禁止扩展到正式数据量。

### 3.2 开发模型

本地和低价 GPU 开发使用 0.5B–1.5B Student 与 3B–7B Teacher。开发模型只验证工程正确性，不进入最终主表。

### 3.3 训练方式

- Student：4-bit QLoRA；
- 计算 dtype：优先 bf16；
- Teacher：bf16 或 8-bit 推理，根据显存决定；
- Student rollout：vLLM；
- Teacher annotation：Transformers teacher-forcing；
- 每轮最多训练 1 epoch，并启用 early stopping。

## 4. 固定数据方案

### 4.1 训练数据

使用 `open-r1/OpenR1-Math-220k` 的 `default` 子集，revision
`e4e141ec9dea9f8326f4d347be56105859b2bd68`，不自行合成题目。

数据用途：

| 字段 | 用途 |
|---|---|
| `problem` | Student rollout prompt |
| `solution` / verified generation | SFT baseline target |
| `answer` | Math verifier 标准答案 |
| `problem_type` | 分领域采样和结果切片 |
| `source` | 数据溯源和污染审计 |
| correctness 字段 | 检查本项目 verifier 的一致性 |

### 4.2 固定规模

| Split | 数量 | 用途 |
|---|---:|---|
| smoke | 128 | 完整 pipeline 快速验证 |
| train | 15,000 | 清洗后的候选池；正式 SFT 和 OPD 均取固定前 7,500 条 |
| validation | 1,000 | early-stop、checkpoint selection |
| regression | 500 | 高频轻量回归，可与 validation 重叠但固定 |

选择 train/validation 前，先从候选数据中移除与最终 benchmark 疑似重复的样本。

### 4.3 最终 Benchmark

最小可信版固定使用：

| Benchmark | 作用 | 指标 |
|---|---|---|
| MATH-500 | 数学主结果 | Accuracy、level/subject 切片 |
| AIME 2024/2025 | 高难度数学 | Accuracy、pass@1；最终模型额外 pass@8 |
| GPQA Diamond | 分布外科学推理 | Accuracy |
| IFEval | 通用能力回归 | Strict/Loose Accuracy |

为了节省 GPU，完整 benchmark 只评测四个最终模型：Base、SFT、Vanilla OPD、Weighted OPD。

### 4.4 不需要人工合成的数据

项目使用现成问题、答案和 reasoning traces。唯一必须自动生成的数据是：

1. 当前 Student 的 rollout；
2. Teacher 对这些 rollout 的 token 分布与 entropy；
3. 更新后的 Student 在第二轮产生的新 rollout。

这些步骤全部由脚本完成，不需要人工编写题目或答案。

## 5. 最小实验矩阵

### 5.1 必做模型

| ID | 方法 | Seed | 是否完整 benchmark |
|---|---|---:|---|
| E0 | Base Student | 42 | 是 |
| E1 | SFT | 42 | 是 |
| E2 | Vanilla OPD | 42 | 是 |
| E3 | Verifier-only OPD | 42 | validation + MATH-500 消融 |
| E4 | Confidence-only OPD | 42 | validation + MATH-500 消融 |
| E5 | Verifier + Confidence Weighted OPD | 42 | 是 |

全项目只使用预先登记的 `seed=42`。E3/E4 只跑 validation 和 MATH-500，可减少完整 benchmark 推理成本。

### 5.2 第二轮 OPD

为了证明系统真正支持策略更新后的 on-policy 数据，只对以下模型的 Round 0 最佳 checkpoint 运行 Round 1：

- Vanilla OPD best checkpoint；
- Weighted OPD best checkpoint。

Round 1 使用相同固定 seed。它用于展示多轮工程闭环，并比较边际收益与成本。

### 5.3 公平条件

所有 OPD 方法共享 Round 0 Base Student rollout 和 Teacher annotation。必须固定：

- Base Student revision；
- 同一批 7,500 prompts；
- 每题 1 个 rollout；
- 最大 3,072 response tokens；
- Teacher revision；
- Teacher top-k；
- optimizer 和 scheduler；
- 最大训练 token；
- validation prompt 和解码参数。

不同方法只能改变 filtering/weighting，不得偷偷增加 Teacher 数据或训练步数。

## 6. OPD 数据流

```text
OpenR1-Math problems
        │
        ▼
Curate / Split / Decontaminate
        │
        ▼
15k Clean Candidate Pool
        │
        ▼
Fixed First 7.5k Prompt Manifest
        │
        ▼
Base Student rollout × 1
        │
        ├──► Math-Verify: pass / fail / unknown
        │
        └──► 14B Teacher annotation
                    │
                    ▼
          top-k logits + entropy
                    │
                    ▼
          Build training views
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Vanilla OPD   Verifier OPD   Weighted OPD
        │                       │
        ▼                       ▼
 validation + early-stop + checkpoint
        │                       │
        └── best checkpoint ────┘
                    │
                    ▼
              Round 1 rollout
```

## 7. 核心训练方法

### 7.1 SFT

使用现成的 verified reasoning trace：

```text
prompt → verified solution
```

只在 response token 上计算 causal language modeling loss。

### 7.2 Vanilla OPD

Student 先生成 response。Teacher 在以下序列上做 teacher-forcing：

```text
prompt + student response
```

在每个 response 位置，让 Student distribution 匹配 Teacher distribution。所有有效 token 等权。

### 7.3 Verifier 权重

第一版使用保守策略：

| Verifier | 权重 |
|---|---:|
| pass | 1.0 |
| unknown | 0.3 |
| fail | 0.0 |

因为最终答案 verifier 无法定位 reasoning 中第一个错误位置，所以主实验不强行蒸馏 fail rollout。`fail=0.1` 只作为后续可选消融，不在最小计划中执行。

### 7.4 Confidence 权重

Teacher annotation 时计算每个 response token 位置的 entropy：

```text
confidence_t = clip(
    1 - entropy_t / log(vocab_size),
    min_weight,
    max_weight
)
```

token 权重在 batch 内归一化到均值约为 1，防止不同方法因为总 loss scale 不同而获得不公平优势。

### 7.5 主方法 Loss

```text
L = sum(response_mask_t
        * verifier_weight
        * confidence_t
        * sparse_KL_t)
    / sum(response_mask_t
          * verifier_weight
          * confidence_t)
```

### 7.6 Sparse KL

完整 vocabulary logits 占用大量存储。主实验保存：

- top-32 Teacher token ids；
- top-32 logprobs；
- tail probability mass 或等价归一化信息；
- 每个位置的 Teacher entropy；
- response mask。

文档和简历中必须称为 `top-k sparse/approximate KL`，不能称为无损 full-vocabulary KL。

在 32–64 条样本上执行 full-logits audit：比较 sparse KL 与 full KL 的 loss、梯度 cosine similarity 和一次 optimizer step 后的参数变化趋势。

## 8. 技术栈

| 模块 | 框架 |
|---|---|
| Python 环境 | Python 3.11 + uv |
| 深度学习 | PyTorch |
| 模型加载 | Hugging Face Transformers |
| Rollout | vLLM |
| QLoRA | PEFT + bitsandbytes |
| 训练启动 | Accelerate |
| 数据 | Datasets + PyArrow/Parquet |
| 数学验证 | Math-Verify |
| 标准评测 | LightEval |
| 配置 | 分层 YAML + 确定性 deep merge |
| CLI | stdlib argparse（降低本地 smoke 依赖） |
| Schema | Pydantic |
| 实验记录 | W&B；无法联网时写本地 JSONL |
| 测试 | pytest + ruff + mypy |
| 环境固定 | Docker |
| 大文件 | S3/MinIO 或持久化磁盘 |

第一版使用阶段式 CLI 和 shell 脚本，不使用复杂工作流调度器。

## 9. 仓库结构

```text
OPDProj/
├── README.md
├── PROJECT_PLAN.md
├── plans/
│   ├── README.md
│   ├── PROJECT_PLAN_RECOMMENDED.md
│   └── PROJECT_PLAN_RESEARCH_SCALE.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Makefile
├── configs/
│   ├── data.yaml
│   ├── rollout.yaml
│   ├── teacher.yaml
│   ├── sft.yaml
│   ├── vanilla_opd.yaml
│   └── weighted_opd.yaml
├── src/opd/
│   ├── cli.py
│   ├── schemas.py
│   ├── artifacts.py
│   ├── data/
│   ├── rollout/
│   ├── teacher/
│   ├── verifier/
│   ├── training/
│   ├── evaluation/
│   └── monitoring/
├── scripts/
├── tests/
├── eval/
├── experiments/
└── reports/
```

## 10. 工程接口

项目最终提供以下命令：

```bash
opd doctor
opd data prepare --config configs/data.yaml
opd data import-eval --name math500 --input EXPORT.jsonl --config configs/main.yaml
opd data audit-contamination --config configs/data.yaml
opd audit sparse-kl --output artifacts/audits/sparse_kl.json --config configs/main.yaml
opd rollout generate --round 0 --config configs/rollout.yaml
opd verify math --round 0 --config configs/main.yaml
opd teacher annotate --round 0 --config configs/teacher.yaml
opd data build-view --method vanilla-opd --round 0
opd data build-view --method weighted-opd --round 0
opd train --config configs/sft.yaml
opd train --config configs/vanilla_opd.yaml
opd train --config configs/weighted_opd.yaml
opd evaluate --suite regression --checkpoint CHECKPOINT
opd benchmark run --checkpoint CHECKPOINT --output-dir OUTPUT_DIR
opd report build --experiment EXPERIMENT_ID
```

每个命令读取 manifest 并产生新的 manifest，不能依赖“某个目录里刚好存在什么文件”。

## 11. 数据与 Artifact 设计

### 11.1 数据目录

```text
data/
├── raw/
├── curated/
├── rollouts/
├── annotations/
├── verifications/
├── training_views/
├── eval/
└── manifests/
```

### 11.2 Manifest 必须记录

- artifact id；
- 上游 artifact id；
- Git commit；
- 完整配置 hash；
- 模型和 tokenizer revision；
- 数据集 revision；
- shard 数和总记录数；
- checksum；
- 创建时间；
- GPU 型号和运行时间；
- prompt/response/Teacher token 数；
-失败和跳过数量。

### 11.3 分片与断点续跑

- rollout shard：100–250 prompts；
- annotation shard：与 rollout shard 一一对应；
- 临时文件使用 `.partial`；
- schema、数量和 checksum 通过后再提交；
- 重启时只补缺失或失败 shard；
- 单条异常写入 error record，不终止整个任务。

## 12. Early Stopping

### 12.1 Validation 指标

```text
composite_score =
    0.80 * math_validation_accuracy
  + 0.10 * format_pass_rate
  + 0.10 * instruction_regression_score
```

权重在实验开始前固定。

### 12.2 单轮停止配置

```yaml
early_stopping:
  eval_steps: 200
  patience: 3
  min_delta: 0.005
  max_epochs_per_round: 1
  load_best_model_at_end: true
```

连续 3 次 evaluation 提升小于 0.5 个百分点时停止。不能根据 train loss 单独停止或继续。

### 12.3 硬停止

以下情况立即暂停：

- NaN/Inf；
- gradient norm 持续异常；
- validation accuracy 明显下跌；
- 通用回归下降超过 2 个百分点；
- 平均输出长度变化超过 30%；
- 重复、空回答或拒答率突然上升；
- Teacher/Student entropy 异常坍缩；
- 自动减小 batch 后仍连续 OOM。

### 12.4 是否执行 Round 1

Round 0 后，只有满足以下条件才继续：

- Vanilla 或 Weighted OPD 至少不差于 SFT；
- pipeline 没有 tokenizer/mask/verifier 问题；
- 剩余预算不少于 20 H100 小时；
- Round 0 validation 有明确可分析信号。

Round 1 后不再继续 Round 2，除非用户主动升级到推荐完整版。

## 13. 评测与比较

### 13.1 主比较

```text
Weighted OPD vs Vanilla OPD
Weighted OPD vs SFT
Vanilla OPD vs Base
```

### 13.2 固定解码

主结果：

```yaml
temperature: 0
num_samples: 1
```

AIME 最终扩展结果：

```yaml
temperature: 0.7
top_p: 0.95
num_samples: 8
```

### 13.3 单 Seed 统计策略

所有训练和生成固定使用 `seed=42`。报告：

- 每个 benchmark 的原始分数；
- 基于逐题结果的 paired bootstrap 95% confidence interval；
- MATH-500 逐题 win/tie/loss；
- 对成对正确/错误结果使用 McNemar test；
- 按 subject、difficulty、response length 和 verifier status 切片；
- validation checkpoint 曲线，而不是只展示最终一个点。

paired bootstrap 只能衡量评测样本的不确定性，不能替代多 seed 对训练随机性的估计。最终报告必须明确写出“本项目使用单训练 seed，训练方差未被完整测量”这一限制，不得把样本级置信区间描述成训练稳定性证明。

### 13.4 成功判据

主方法满足以下多数条件，可视为积极结果：

1. MATH validation 和 MATH-500 高于 Vanilla OPD；
2. paired bootstrap 差值置信区间不明显跨越 0；
3. MATH-500 的逐题净胜样本数为正；
4. AIME 不退化；
5. GPQA/IFEval 下降不超过预设阈值；
6. Teacher tokens 与 Vanilla OPD 相同；
7. 没有输出长度、重复率和格式退化。

如果差异不显著，项目仍然成立，但结论应写成“在当前预算和数据规模下，未观察到可靠收益”，并分析 verifier 过滤率、Teacher entropy 和数据新鲜度。

## 14. GPU 时间预算

### 14.1 总预算

```text
目标：60–90 H100 GPU 小时
硬上限：110 H100 GPU 小时
```

任何阶段达到硬上限都停止扩大实验，只完成已有 artifact 的评测和报告。

### 14.2 分阶段预算

| 阶段 | H100 GPU 小时 | 备注 |
|---|---:|---|
| 小模型与 8B smoke | 6–12 | loss、mask、verifier、恢复 |
| Base rollout：7.5k × 1 | 4–7 | response 上限 3,072；800 条截断门禁 |
| Round 0 Teacher annotation | 3–6 | top-32 + entropy |
| SFT | 4–8 | seed 42，early-stop |
| 四个 OPD 训练 run | 12–24 | Vanilla、Verifier、Confidence、Weighted 各一次 |
| Round 1：两个最佳 checkpoint | 12–22 | rollout、annotation、training |
| 最终 benchmark | 7–14 | 仅四个最终模型 |
| 失败重试和余量 | 6–12 | 不提前花掉 |
| **总计** | **54–105** | 目标控制在 60–90 |

### 14.3 降本规则

按以下顺序降本，不破坏实验可信度：

1. 使用 early stopping；
2. 消融只用 25%–50% train prompts；
3. 完整 benchmark 只评测最终模型；
4. Round 1 只运行两个最佳 checkpoint；
5. 保持 3,072 token 上限，优先减少正式 prompt 数，而不是制造截断样本；
6. 若 800 条门禁仍失败，停止并评估关闭 thinking mode，不盲目继续生成；
7. 若仍超预算，将 7.5k 正式 prompts 进一步下调，并在报告中明确记录。

不允许通过取消 Base、SFT、Vanilla OPD 或独立 validation 来省钱。

## 15. 实施里程碑

### M0：工程骨架，2–3 天，0 GPU 小时

- Python 项目、目录和配置；
- argparse CLI；
- Pydantic schema；
- manifest 和 artifact hash；
- Docker、doctor、pytest、ruff、mypy。

验收：CPU CI 全部通过。

### M1：数据与 Verifier，3–5 天，0–2 GPU 小时

- 下载已固定 revision 的 OpenR1-Math；
- 产生 smoke/train/validation split；
- 污染检查；
- Math-Verify wrapper；
- 数据卡。

验收：每条数据可追溯，verifier 三态正确。

### M2：Rollout，3–4 天，4–8 GPU 小时

- tiny model 本地 smoke；
- 8B vLLM rollout；
- 分片、恢复和错误记录；
- 先生成 800 条，通过截断率不高于 20% 的门禁后自动扩展到 7.5k。

验收：中断后不会重复生成成功 shard；`quality_gate.json` 为 `passed`。

### M3：Teacher Annotation，3–5 天，3–7 GPU 小时

- teacher-forcing；
- response mask；
- top-32 logits；
- entropy；
- sparse/full KL audit。

验收：无 token 偏移，sparse KL audit 可解释。

### M4：Baseline 与主方法，5–8 天，25–50 GPU 小时

- SFT；
- Vanilla OPD；
- Verifier-only；
- Confidence-only；
- Weighted OPD；
- early-stop、checkpoint resume。

验收：所有方法使用同一训练和评测入口。

### M5：Round 1 与配对分析，4–7 天，15–30 GPU 小时

- 固定 seed 42；
- 选择 Round 0 最佳 checkpoint；
- 生成 Round 1 rollout；
- 第二轮训练；
- 记录边际收益/Teacher token。

验收：至少完成一条真正的多轮 on-policy 链路。

### M6：正式评测与报告，4–6 天，10–20 GPU 小时

- MATH-500；
- AIME；
- GPQA Diamond；
- IFEval；
- 统计检验；
- 失败案例；
- 成本 Pareto；
- README、数据卡和模型卡。

预计总周期：3–5 周。

## 16. 测试与质量门禁

### 16.1 必须具备的单元测试

- prompt/response mask；
- padding、EOS、截断；
- sparse KL 与 full KL 小样本对照；
- verifier pass/fail/unknown；
- confidence 权重范围与归一化；
- manifest hash 稳定性；
- shard resume；
- 固定 seed 可复现。

### 16.2 End-to-end 测试

用 16–32 条 fixture 和 tiny model 执行：

```text
prepare
→ rollout
→ verify
→ teacher annotate
→ build training view
→ train one optimizer step
→ evaluate
→ build report
```

### 16.3 Checkpoint 晋级

只有满足以下条件才标记为 `promotable`：

- schema 和 artifact 完整；
- validation 不低于对应 baseline 的回退阈值；
- 无 NaN、OOM 未恢复或 checkpoint 损坏；
- 输出长度、格式和重复率正常；
- 配置、代码 commit 和数据 manifest 可追溯。

## 17. 远端 GPU 执行方式

### 17.1 每次实例启动

```text
拉取固定 Git commit
→ 启动固定 Docker image
→ opd doctor
→ 下载输入 manifest/shards
→ 执行一个阶段 job
→ 上传 artifact 和日志
→ 远端校验 checksum
→ 关闭实例
```

### 17.2 Job 拆分

禁止使用一个脚本从数据下载一直运行到最终评测。必须拆成：

```text
prepare-data
generate-rollouts
verify
annotate-teacher
build-training-view
train
evaluate
build-report
```

这样租用服务器中断时，不会丢失已经完成的 Teacher annotation 或 rollout。

### 17.3 日志和成本

每个 job 保存：

- GPU 型号和数量；
- wall time；
- tokens/s；
- prompt/response/Teacher tokens；
- 峰值显存；
- shard 成功率；
- estimated GPU hours/cost；
- Git commit；
- config hash；
- output manifest。

## 18. 最终交付物

代码仓库最终必须包含：

1. 一条命令运行的 tiny-model smoke demo；
2. 可恢复的 8B rollout 和 14B annotation pipeline；
3. SFT、Vanilla OPD 和 Weighted OPD 统一训练入口；
4. 自动化 validation、early-stop 和 checkpoint promotion；
5. MATH-500/AIME/GPQA/IFEval 结果；
6. 单 seed 的逐题 paired bootstrap、McNemar test 和限制说明；
7. 数据污染报告；
8. 数据卡与模型卡；
9. 质量—成本 Pareto 图；
10. 失败案例和负结果分析；
11. 实际 GPU 小时和 Teacher token 账单；
12. 5 分钟演示脚本。

## 19. 简历表述模板

获得真实数字后再填入结果：

> 构建面向 8B LLM 的预算感知 On-Policy Distillation 系统，使用 vLLM、Transformers、QLoRA 与 Math-Verify 解耦 Student rollout、14B Teacher sparse-logit annotation、Verifier 和训练任务；实现分片恢复、artifact lineage、validation early-stop 与标准 benchmark 评测，并通过 verifier-aware filtering 和 token-level confidence weighting，在固定 Teacher token 预算下对比 SFT、Vanilla OPD 和改进方法的质量—成本表现。

不能在实验前填写“提升 X%”或“降低 Y%”。

## 20. 立即执行清单

在租用主实验 GPU 前，依次完成：

1. 初始化 Git；
2. 创建 `pyproject.toml` 和锁文件；
3. 创建目录结构和 CLI；
4. 定义 Pydantic schemas；
5. 实现 artifact manifest 和 hash；
6. 准备 128 条 OpenR1-Math fixture；
7. 实现 Math verifier；
8. 实现 response mask 和测试；
9. 实现 sparse KL 和 full-KL audit 测试；
10. 用 tiny model 跑通 end-to-end；
11. 构建 Docker image；
12. 运行 8B/14B 的 32 条 GPU smoke；
13. 根据真实 tokens/s 更新 GPU 预算；
14. 从 15k 清洗候选池选择固定前 7.5k，生成 Round 0 rollout；800 条时自动执行截断门禁。

前 13 项没有完成前，不启动大规模生成或训练。
