# OPD-Lab 推荐完整版工程方案

> 状态：备用。预计需要 300–500 H100 GPU 小时；当前执行方案见 [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。

## 1. 项目概述

### 1.1 项目名称

**Budget-Aware Verifier- and Confidence-Weighted On-Policy Distillation for 7B Reasoning LLMs**

中文名称：**面向 7B 推理模型的预算感知、验证器与置信度加权在线策略蒸馏系统**。

### 1.2 项目目标

本项目面向 LLM 训练、后训练和训练系统方向的实习申请，目标不是单纯“微调一个模型”，而是构建一套可复现、可恢复、可评测、可审计、成本可控的 OPD 工程系统，并回答一个明确问题：

> 在固定 Student、训练 token 和 Teacher 推理预算下，Verifier-aware filtering 与 confidence-weighted distillation 能否比 SFT、离线蒸馏和 Vanilla OPD 获得更好的质量—成本表现？

项目主要展示以下能力：

- LLM 数据清洗、去重、污染检查和版本管理；
- Student rollout 和 Teacher annotation 的高吞吐推理；
- 自定义蒸馏 loss、LoRA/QLoRA 和混合精度训练；
- Verifier、置信度加权和 on-policy 多轮数据刷新；
- 标准 benchmark、统计显著性和回归门禁；
- 远端租用 GPU 的任务拆分、断点续跑和成本核算。

### 1.3 OPD 的定义

这里的 OPD 指 **On-Policy Distillation**：

1. 当前 Student 根据现成题目生成回答；
2. Teacher 在 Student 实际生成的上下文上给出 token 分布或近似分布；
3. Verifier 判断回答是否正确、可解析或存在异常；
4. Student 使用 Teacher 信号和 Verifier 权重继续训练；
5. 更新后的 Student 再生成新一轮 rollout。

公开数据集可以提供题目、参考答案和现成 reasoning traces，但无法完全替代 OPD 数据，因为真正的 on-policy rollout 必须由“当前版本的 Student”生成。

## 2. 项目范围

### 2.1 主线范围

第一版只做 **数学推理**，原因是：

- 存在规模足够大的现成训练数据；
- 可以使用确定性的答案 verifier；
- 不需要自行编写题目或参考答案；
- 不需要维护不安全的代码执行沙箱；
- 评测结果容易解释，也适合进行统计分析。

主线稳定后，再将代码推理作为第二阶段扩展。项目不在第一版同时加入工具调用、长上下文、对话安全和多模态任务，避免范围失控。

### 2.2 非目标

- 不从零预训练基础模型；
- 不训练独立 Reward Model；
- 不声称提出全新的基础蒸馏算法；
- 不以 32B/70B 多卡训练作为项目成立条件；
- 不自行合成大规模题目；
- 不用最终 benchmark 反复选择 checkpoint；
- 不把单次最好结果作为最终结论。

## 3. 模型方案

### 3.1 开发阶段

| 角色 | 建议模型规模 | 用途 |
|---|---:|---|
| Student | 0.5B–1.5B | 本地或低价 GPU 验证数据、mask、loss 和 checkpoint |
| Teacher | 3B–7B | 验证 annotation、top-k logits 和 confidence 计算 |

开发阶段只追求流程正确，不作为最终实验结果。

### 3.2 主实验阶段

| 角色 | 默认候选 | 运行方式 |
|---|---|---|
| Student | `Qwen3-8B` 同级 Instruct 模型 | LoRA/QLoRA 训练 |
| Teacher | 同系列 `14B` Instruct/Reasoning 模型 | 独立 annotation job，不参与反向传播 |
| Teacher 扩展 | 同系列 `32B` 模型 | 只做 teacher scale 消融，可选 |

最终模型名称和 revision 必须在第一次 GPU smoke test 后固定。Student 和 Teacher 优先使用同系列、相同 tokenizer/vocabulary 的模型，降低 token 对齐和 KL 计算风险。

### 3.3 显存建议

- 8B Student QLoRA：优先 24–48 GB GPU；
- 14B Teacher 推理：优先 48 GB GPU，长序列或大 batch 使用 80 GB；
- 32B Teacher：80 GB 单卡或多卡，仅作为可选扩展；
- Student rollout、Teacher annotation 和 Student training 分成不同任务，不要求模型同时驻留显存。

## 4. 数据方案

### 4.1 训练数据

主训练数据使用现成公开数据，不自行设计题目。

#### 主数据集：OpenR1-Math-220k

使用 `open-r1/OpenR1-Math-220k` 的 `default` 子集：

- 约 94k 个问题；
- 每题包含参考 solution、answer 和多个 reasoning generations；
- 包含 Math-Verify 等正确性字段；
- Apache-2.0；
- 可直接通过 Hugging Face Datasets 加载；
- 现成 reasoning traces 可用于 SFT 和 Offline KD baseline；
- `problem` 字段可直接作为 OPD rollout prompt。

主实验不一定使用全部数据。建议清洗、去重和污染检查后选择 30k–60k 个 prompt，以控制 Teacher 推理成本。

#### 扩展数据集：NuminaMath-1.5

`AI-MO/NuminaMath-1.5` 约有 896k 条训练数据，包含：

- problem；
- solution；
- answer；
- problem type；
- question type；
- source；
- validity；
- synthetic 标记。

它用于扩充题型、构建难度切片和研究领域采样，不建议第一轮直接使用全部数据。

#### 可选小型对照数据

- `bespokelabs/Bespoke-Stratos-17k`；
- `simplescaling/s1K`；
- `open-thoughts/OpenThoughts-114k`。

这些数据主要用于小数据、高质量 reasoning 的 SFT 对照。使用前必须固定具体 revision 并重新核对许可证。

### 4.2 最终 Benchmark

| 评测集 | 作用 | 主要指标 |
|---|---|---|
| MATH-500 | 数学主评测 | Accuracy、按 level/subject 切片 |
| AIME 2024/2025 | 高难度推理 | Accuracy、pass@1、pass@8 |
| AMC 2023 | 中等难度推理 | Accuracy |
| GPQA Diamond | 分布外科学推理 | Accuracy |
| IFEval | 通用指令回归 | Strict/Loose Accuracy |

MATH-500、AIME、GPQA 和 IFEval 只用于最终比较，不用于 early stopping 或日常调参。

### 4.3 数据划分

```text
OpenR1/NuminaMath 原始数据
          │
          ├── train：训练和生成 rollout
          ├── validation：early-stop、checkpoint selection、超参数选择
          └── final benchmark：完全独立的公开评测集
```

建议规模：

- train prompts：30k–60k；
- validation prompts：1k–2k；
- smoke prompts：32–128；
- regression prompts：每个主要题型 100–300；
- final benchmark：使用完整固定版本。

### 4.4 数据污染检查

在进入训练前，对 train、reference solution、公开 reasoning traces 和最终 benchmark 执行：

1. 规范化文本 exact hash；
2. n-gram/MinHash 近重复检查；
3. 数学表达式和答案规范化比较；
4. 高相似样本人工抽样；
5. 按 source 统计疑似重合比例。

疑似污染样本进入 quarantine，不直接静默删除。最终报告公开阈值、数量和处理方式。

## 5. 数据模型与存储

### 5.1 目录分层

```text
data/
├── raw/                  # 原始数据快照，只读
├── curated/              # 清洗、去重、切分后的 prompt pool
├── rollouts/             # Student 生成回答
├── annotations/          # Teacher top-k logits、entropy、版本信息
├── verifications/        # Verifier 输出
├── training_views/       # SFT/KD/OPD 可训练视图
├── eval/                 # 锁定的 validation 和 benchmark 配置
└── manifests/            # 每个 artifact 的 hash 和 lineage
```

### 5.2 核心记录

每个 rollout 至少包含：

```json
{
  "sample_id": "math_000001",
  "rollout_id": "round_01_math_000001_r02",
  "parent_sample_id": "math_000001",
  "source_dataset": "open-r1/OpenR1-Math-220k",
  "source_revision": "pinned_revision",
  "prompt": "...",
  "reference_answer": "...",
  "response": "...",
  "student_revision": "...",
  "tokenizer_revision": "...",
  "sampling_config_hash": "...",
  "seed": 42,
  "prompt_tokens": 256,
  "response_tokens": 768,
  "latency_ms": 1234,
  "status": "success"
}
```

Teacher annotation 至少包含：

```json
{
  "rollout_id": "round_01_math_000001_r02",
  "teacher_revision": "...",
  "topk_token_ids_uri": "...",
  "topk_logprobs_uri": "...",
  "token_entropy_uri": "...",
  "teacher_tokens": 1024,
  "latency_ms": 850,
  "status": "success"
}
```

Verifier 输出至少包含：

```json
{
  "rollout_id": "round_01_math_000001_r02",
  "verifier_name": "math_verify",
  "verifier_version": "...",
  "status": "pass",
  "score": 1.0,
  "extracted_answer": "...",
  "error_type": null
}
```

### 5.3 文件格式

- 元数据和文本：Parquet；
- top-k token ids/logprobs：分片 Parquet、Arrow 或 safetensors；
- 配置与 manifest：YAML/JSON；
- checkpoint：safetensors；
- 图表：PNG/SVG；
- 汇总结果：JSON + Parquet + Markdown。

大数据和 checkpoint 不提交 Git。Git 只保存代码、配置、schema、小型测试样本和 manifest。远端 artifact 使用 S3/MinIO 等对象存储或租用平台的持久化磁盘。

## 6. 系统架构

```text
Dataset Registry
      │
      ▼
Curate + Split + Decontaminate
      │
      ▼
Prompt Manifest
      │
      ▼
Student Rollout Workers ──────► Rollout Shards
      │                              │
      │                              ├──► Math Verifier
      │                              │
      │                              └──► Teacher Annotator
      │                                      │
      ▼                                      ▼
Quality/Confidence Join ◄──────── Annotation Shards
      │
      ▼
Training View Builder
      │
      ├──► SFT
      ├──► Offline KD
      ├──► Vanilla OPD
      └──► Weighted OPD
              │
              ▼
        Student Trainer
              │
              ▼
       Validation + Early Stop
              │
              ├── stop
              └── next rollout round
```

每个框代表独立 CLI job。每个 job：

- 只读取上游 manifest；
- 以分片方式写入临时文件；
- 完成校验后原子提交；
- 支持重复执行和断点续跑；
- 已成功的 shard 不重复计算；
- 显式记录失败、超时和跳过原因。

## 7. 技术栈

### 7.1 核心框架

| 功能 | 框架 | 选择原因 |
|---|---|---|
| 深度学习 | PyTorch | 自定义 OPD loss 和训练循环 |
| 模型/tokenizer | Transformers | 加载 Student/Teacher、teacher-forcing forward |
| 高吞吐 rollout | vLLM | Continuous batching 和高效生成 |
| 参数高效训练 | PEFT | LoRA/QLoRA |
| 量化 | bitsandbytes | 4-bit/8-bit Student 加载 |
| 单卡/多卡封装 | Accelerate | 混合精度、梯度累积和分布式启动 |
| 数据 | Datasets + PyArrow | Hugging Face 数据和 Parquet |
| 配置 | Hydra/OmegaConf | 可组合实验配置和命令行覆盖 |
| Schema | Pydantic | 输入输出数据校验 |
| CLI | Typer | 清晰的阶段式命令 |
| 数学验证 | Math-Verify | 答案抽取和数学等价验证 |
| 标准评测 | LightEval | MATH、AIME、GPQA、IFEval 等评测 |
| 实验追踪 | W&B 或 MLflow | 曲线、配置、artifact 和系统资源 |
| 测试与质量 | pytest、ruff、mypy | 正确性、格式和类型检查 |
| 环境 | Docker | 固定 CUDA、Python 和依赖版本 |

第一版不引入 Kubernetes、Kafka、Airflow 或微服务。任务调度使用 CLI + shell/Slurm 脚本；确实需要多个远端 worker 后再考虑 Ray。

### 7.2 建议 Python 版本

- Python 3.11；
- CUDA、PyTorch、vLLM 和 bitsandbytes 使用互相兼容的固定版本；
- `pyproject.toml` 和 lock file 固定依赖；
- Docker image 使用 digest 或明确 tag。

## 8. 仓库结构

```text
OPDProj/
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Makefile
├── configs/
│   ├── data/
│   ├── model/
│   ├── rollout/
│   ├── teacher/
│   ├── training/
│   ├── evaluation/
│   ├── sft.yaml
│   ├── offline_kd.yaml
│   ├── vanilla_opd.yaml
│   └── weighted_opd.yaml
├── src/opd/
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── artifacts.py
│   ├── data/
│   │   ├── registry.py
│   │   ├── prepare.py
│   │   ├── deduplicate.py
│   │   ├── contamination.py
│   │   └── manifest.py
│   ├── rollout/
│   │   ├── generator.py
│   │   ├── shard.py
│   │   └── store.py
│   ├── teacher/
│   │   ├── annotate.py
│   │   ├── sparse_logits.py
│   │   └── confidence.py
│   ├── verifier/
│   │   ├── base.py
│   │   ├── math.py
│   │   └── registry.py
│   ├── training/
│   │   ├── collator.py
│   │   ├── losses.py
│   │   ├── weighting.py
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   └── report.py
│   └── monitoring/
│       ├── logging.py
│       ├── resources.py
│       └── cost.py
├── scripts/
│   ├── remote_bootstrap.sh
│   ├── prepare_data.sh
│   ├── generate_rollouts.sh
│   ├── annotate_teacher.sh
│   ├── verify.sh
│   ├── build_training_view.sh
│   ├── train.sh
│   └── evaluate.sh
├── tests/
│   ├── fixtures/
│   ├── test_schemas.py
│   ├── test_response_mask.py
│   ├── test_sparse_kl.py
│   ├── test_weighting.py
│   ├── test_math_verifier.py
│   ├── test_manifest.py
│   └── test_resume.py
├── eval/
│   ├── registry.yaml
│   ├── prompts/
│   └── reports/
├── experiments/
│   ├── registry.yaml
│   └── templates/
└── reports/
    ├── figures/
    ├── data_card.md
    ├── model_card.md
    └── final_report.md
```

## 9. 训练方法

### 9.1 Baseline A：Base

不进行训练，直接评测原始 7B Student。它提供所有后续提升和退化的参照。

### 9.2 Baseline B：SFT

使用 OpenR1-Math 中经过验证的正确 reasoning trace：

```text
prompt → verified teacher/reference solution
```

只在 response token 上计算 causal language modeling loss。

### 9.3 Baseline C：Offline KD

使用固定 Teacher 生成的标准解答或数据集现成 Teacher trace。它与 SFT 的区别在于 Teacher 数据来源和蒸馏配置，但数据不是当前 Student 生成，因此不是 on-policy。

### 9.4 Baseline D：Vanilla OPD

当前 Student 先生成 rollout，Teacher 在相同 tokenizer 下对 `prompt + student response prefix` 做 teacher-forcing forward。训练时所有有效 token 使用统一权重，匹配 Teacher token distribution。

### 9.5 主方法：Verifier- and Confidence-Weighted OPD

在 Vanilla OPD 基础上加入两类权重。

#### 样本级 Verifier 权重

第一版保守配置：

| 状态 | 建议初始权重 | 含义 |
|---|---:|---|
| pass | 1.0 | 最终答案可验证正确 |
| unknown | 0.3 | 无法可靠解析，弱训练并保留分析 |
| fail | 0.0 | 主实验过滤；在消融中尝试低权重 |

最终答案 verifier 无法定位具体推理错误，因此第一版不对 fail rollout 强蒸馏。后续若实现 step-level verifier，可以只保留首个错误前的正确 prefix。

#### Token 级 Teacher confidence 权重

Teacher annotation 时计算每个 response token 位置的 entropy，并转成置信度：

```text
confidence_t = clip(1 - entropy_t / log(vocab_size), min_weight, max_weight)
```

为了避免不同 batch 的总权重差异改变有效学习率，token 权重在 batch 内归一化到均值约为 1。

#### 总损失

```text
L_OPD = sum(mask_t * verifier_weight * confidence_t * KL_sparse_t)
        / sum(mask_t * verifier_weight * confidence_t)
```

其中 `mask_t` 保证只训练 response、非 padding、非截断异常 token。

### 9.6 Teacher 分布存储

完整 vocabulary logits 存储量过大。主实验使用：

- Teacher top-k token ids；
- 对应 logprobs；
- tail probability mass 或归一化信息；
- 每个位置的完整 entropy 标量；
- response mask。

训练时计算 sparse/approximate KL。必须在文档中明确它是 top-k 近似，不宣称为完整 vocabulary KL。

在 32–128 条样本上增加 full-logits audit，比较 sparse KL 与 full KL 的 loss、梯度方向和短训练结果，验证近似误差。

### 9.7 On-policy 轮次

```text
Round 0：Base Student rollout → OPD training
Round 1：Updated Student rollout → OPD training
Round 2：可选，根据收益和预算决定
```

第一版最多运行 2–3 轮。每轮数据都保存 Student revision，禁止混淆不同轮次。

## 10. Rollout 与 Annotation

### 10.1 Rollout 配置

默认生成两套结果：

- 训练 rollout：temperature 0.7、top-p 0.9、每题 2 个回答；
- deterministic validation：temperature 0、每题 1 个回答。

最大 response 长度根据开发阶段长度分布确定，第一版建议从 2048 tokens 开始，只有证据表明大量正确答案被截断时才提高到 4096。

### 10.2 分片与恢复

- 每个 shard 100–500 prompts；
- shard 名称包含数据 manifest、模型 revision 和 sampling hash；
- 先写 `.partial`，通过条数和 schema 校验后提交正式文件；
- job 重启时扫描 manifest，只补失败或缺失 shard；
- 单条失败不终止整个 shard；
- 保存 OOM、timeout、empty、truncated 等明确状态。

### 10.3 Teacher annotation

Teacher 不重新自由生成答案，而是在 Student rollout 上执行 teacher-forcing：

```text
input = prompt + student_response
output = teacher distribution at each response position
```

这使 Teacher 信号对应 Student 当前访问到的状态。Teacher annotation job 按长度分桶并动态 batching，记录：

- prompt/response tokens；
- batch size；
- tokens/s；
- GPU 型号；
- 峰值显存；
- wall time；
- 失败原因；
- 估算费用。

## 11. 评测方案

### 11.1 三档评测

#### Smoke

- 32–64 条固定样本；
- 检查生成、答案抽取、schema 和报告；
- 每次代码变更均可运行；
- 不用于项目结论。

#### Regression

- 500–1500 条 validation 样本；
- 每 100–250 optimizer steps 运行；
- 用于 early stopping 和 checkpoint promotion；
- 输出题型、难度、长度和错误类型切片。

#### Full

- MATH-500；
- AIME 2024/2025；
- AMC 2023；
- GPQA Diamond；
- IFEval；
- 只对候选最终 checkpoint 运行。

### 11.2 固定评测条件

所有模型使用相同：

- prompt template；
- tokenizer/chat template；
- max tokens；
- stop tokens；
- temperature/top-p；
- benchmark revision；
- answer extractor/verifier revision；
- GPU dtype 和推理框架版本。

主表使用 greedy/pass@1。AIME 等高难度任务额外报告 temperature 0.7 下的 pass@8，但不能用 pass@8 代替主结果。

### 11.3 统计方法

- 每个主方法至少运行 3 个 seed；
- 报告 mean、standard deviation 和 95% confidence interval；
- 对同一批题使用 paired bootstrap；
- 对逐题正确/错误结果使用 McNemar test 或 paired permutation test；
- 差值置信区间包含 0 时，结论写为“未观察到显著差异”。

### 11.4 最终比较表

| 方法 | MATH-500 ↑ | AIME ↑ | GPQA ↑ | IFEval ↑ | Verifier pass ↑ | Teacher tokens ↓ | GPU h ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base |  |  |  |  |  | 0 | 0 |
| SFT |  |  |  |  |  |  |  |
| Offline KD |  |  |  |  |  |  |  |
| Vanilla OPD |  |  |  |  |  |  |  |
| Weighted OPD |  |  |  |  |  |  |  |

最终必须同时展示质量和成本 Pareto 图：

```text
x：Teacher tokens 或 GPU hours
y：MATH-500 / 综合 validation score
```

## 12. Early Stopping

### 12.1 单轮训练停止

不使用 train loss 单独决定停止。使用固定 validation composite score：

```text
score = 0.75 * math_accuracy
      + 0.15 * format_pass_rate
      + 0.10 * instruction_regression_score
```

初始建议：

```yaml
early_stopping:
  eval_steps: 200
  patience: 3
  min_delta: 0.005
  mode: max
  max_epochs_per_round: 1
```

连续 3 次评测提升不足 0.5 个百分点时，结束当前训练，并恢复该轮最佳 checkpoint。

### 12.2 异常硬停止

满足以下任一条件立即暂停：

- loss 为 NaN/Inf；
- gradient norm 连续超过阈值；
- validation 主指标显著下降；
- IFEval/通用回归下降超过预设阈值；
- 平均输出长度变化超过 30%；
- 重复率、空回答率或拒答率异常；
- entropy 快速坍缩；
- GPU OOM 连续出现且自动降 batch 后仍失败。

### 12.3 OPD 轮次停止

连续两轮出现以下任一情况时，不再生成下一轮 rollout：

- validation accuracy 提升小于 0.5 个百分点；
- 单位一百万 Teacher tokens 的收益低于预设阈值；
- verifier pass rate 基本不再提升；
- Student/Teacher 分歧显著降低且能力不再增长；
- 主任务提升伴随持续的通用能力退化；
- 已达到固定 Teacher token 或 GPU 小时预算。

## 13. 实验矩阵

### 13.1 必做实验

| ID | 方法 | 目的 |
|---|---|---|
| E0 | Base | 原始能力 |
| E1 | SFT | 标准监督微调基线 |
| E2 | Offline KD | 静态 Teacher 数据基线 |
| E3 | Vanilla OPD | 等权 on-policy 蒸馏基线 |
| E4 | Verifier-filtered OPD | 验证过滤是否有效 |
| E5 | Confidence-weighted OPD | 验证 token confidence 是否有效 |
| E6 | Verifier + Confidence OPD | 项目主方法 |

### 13.2 必做消融

1. `pass only` 与 `pass + unknown`；
2. uniform token weight 与 confidence weight；
3. top-k 为 16、32、64；
4. rollout 每题 1 个与 2 个；
5. Round 0 与 Round 1；
6. 25%、50%、100% Teacher token 预算。

### 13.3 可选扩展

- Student uncertainty/disagreement 驱动的 active annotation；
- replay buffer 的新旧 rollout 比例；
- 14B 与 32B Teacher 对比；
- 数学 + 代码多领域采样；
- step-level verifier 和错误前缀蒸馏。

## 14. 远端 GPU 工作流

### 14.1 本地完成的工作

- 代码开发；
- schema 和 manifest 测试；
- 32 条数据的 CPU/tiny-model smoke test；
- Docker build；
- 配置审查；
- Git 提交和实验注册。

### 14.2 GPU 实例启动流程

```text
启动实例
  → 拉取固定 commit
  → 启动固定 Docker image
  → 运行 doctor 自检
  → 拉取当前阶段输入 manifest
  → 执行单个 job
  → 上传 shard/checkpoint/日志
  → 校验远端 artifact
  → 关闭实例
```

`doctor` 应检查：

- GPU 型号和数量；
- CUDA/PyTorch 可用性；
- bf16 支持；
- 磁盘空间；
- 对象存储读写；
- Hugging Face 模型可读取；
- W&B/MLflow 是否可用；
- 一个最小 forward/generation 是否成功。

### 14.3 成本记录

每个 job 记录：

```json
{
  "job_type": "teacher_annotation",
  "gpu_type": "...",
  "gpu_count": 1,
  "wall_time_hours": 3.2,
  "prompt_tokens": 12000000,
  "response_tokens": 28000000,
  "tokens_per_second": 3500,
  "estimated_cost": 0.0,
  "currency": "CNY"
}
```

预算建议按比例划分：

- 10%：环境和 smoke；
- 35%：Student rollout；
- 30%：Teacher annotation；
- 15%：Student training；
- 10%：最终复现和完整评测。

任何阶段超预算时先缩小 prompt 数或 rollout 数，不直接降低数据审计和评测质量。

## 15. 可靠性设计

### 15.1 幂等性

相同输入 manifest、代码 commit 和配置 hash 必须产生同一个 artifact identity。已完成的 shard 不重复生成。

### 15.2 原子写入

所有大文件先写临时路径，通过 checksum、schema 和记录数校验后再重命名或提交 manifest。

### 15.3 Checkpoint

同时保存：

- latest：故障恢复；
- best：validation score 最优；
- end-of-round：分析不同 OPD round；
- final-promoted：通过完整回归门禁。

### 15.4 可观测性

训练日志至少包括：

- total/distillation loss；
- 有效 response tokens；
- verifier 权重分布；
- Teacher entropy 分布；
- gradient norm；
- learning rate；
- tokens/s；
- GPU utilization 和 peak memory；
- validation score；
-累计 Teacher tokens 和 GPU hours。

## 16. 测试与 CI

### 16.1 单元测试

- prompt/response mask 边界；
- padding、EOS 和截断；
- sparse KL 与 full KL 小词表对照；
- confidence 权重范围和归一化；
- pass/fail/unknown verifier；
- manifest hash 稳定性；
- shard resume 和重复执行；
- 多 seed 数据顺序可复现。

### 16.2 集成测试

使用 tiny model 和 16–32 条 fixture 完整运行：

```text
prepare → rollout → verify → annotate → build view → train → evaluate
```

### 16.3 CI 门禁

每个 pull request 执行：

- `ruff check`；
- `ruff format --check`；
- `mypy`；
- `pytest`；
- 配置解析；
- 纯 CPU pipeline smoke test。

GPU 测试不放在每个 PR 中，可在远端手动或定期运行。

## 17. 实施里程碑

### M0：工程骨架（2–3 天）

- 初始化 Git 和 Python 项目；
- 创建配置、CLI、schema、日志和测试；
- Docker 与 doctor 命令；
- CI 通过。

验收：`pytest` 和 CPU smoke 成功。

### M1：数据产品（3–5 天）

- 加载 OpenR1-Math；
- 清洗、切分、去重；
- 污染审计；
- Parquet 和 manifest；
- 数据卡。

验收：任意样本可以追溯来源和处理版本。

### M2：Rollout + Verifier（4–6 天）

- vLLM rollout；
- 分片、断点续跑；
- Math-Verify 三态输出；
- 100–1000 prompts GPU smoke。

验收：中断后可恢复，统计与记录数一致。

### M3：Teacher Annotation（4–6 天）

- teacher-forcing；
- response mask；
- top-k logits 与 entropy；
- annotation artifact；
- full/sparse KL audit。

验收：小词表/小样本 loss 对照通过，无 tokenizer 偏移。

### M4：训练 Baseline（5–7 天）

- SFT；
- Offline KD；
- Vanilla OPD；
- checkpoint resume；
- validation early-stop。

验收：所有 baseline 使用同一配置体系和评测入口。

### M5：主方法（5–7 天）

- verifier filtering；
- confidence weighting；
- loss normalization；
- Round 1 rollout 刷新；
- 主要消融。

验收：能解释每条样本的最终权重和训练去向。

### M6：完整评测与报告（5–10 天）

- 三个 seed；
- 完整 benchmark；
- paired bootstrap/McNemar；
- 成本 Pareto；
- 失败案例分类；
- 模型卡、数据卡和最终报告。

验收：干净环境可以复现一个缩小版结果，主表可由 artifact 自动生成。

预计总周期为 4–7 周，可根据 GPU 预算缩小数据量，但不要删除 baseline、validation 和统计比较。

## 18. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| Tokenizer 不一致 | Teacher/Student token 无法对齐 | 使用同系列模型并固定 tokenizer revision |
| Sparse KL 误差 | top-k 近似与 full KL 差异大 | 增大 k，保存 tail mass，执行 full-logits audit |
| Verifier 误判 | 正确答案被过滤 | pass/fail/unknown 三态和人工抽样 |
| Benchmark 污染 | 指标异常虚高 | exact/MinHash/source 审计和 quarantine |
| 模式坍缩 | 输出变短、重复或 entropy 下降 | 硬停止、输出监控、混合少量 SFT anchor data |
| 灾难性遗忘 | 数学提升但 IFEval/GPQA 下降 | 通用回归门禁和 SFT anchor mix |
| Teacher 成本过高 | annotation 占据大部分预算 | 缩短长度、减少 rollout 数、长度分桶、top-k 存储 |
| 租用实例中断 | 数据和 checkpoint 丢失 | shard、原子提交、对象存储、latest checkpoint |
| 提升不显著 | OPD 不优于 SFT | 如实报告负结果，分析分歧、数据新鲜度和成本边界 |

## 19. 项目成功标准

项目不以必须刷新公开 SOTA 为成功条件。满足以下条件即可构成成熟的简历项目：

1. 可以从固定配置重建数据、rollout、annotation 和训练视图；
2. 7B Student 的 SFT、Offline KD、Vanilla OPD 和 Weighted OPD 公平可比；
3. 至少完成两轮 on-policy 数据刷新；
4. 至少完成 3 个 seed 或对关键实验完成重复验证；
5. 报告置信区间、失败案例、污染审计和成本；
6. 明确指出方法有效和无效的场景；
7. 小规模 end-to-end demo 能在新环境一条命令运行；
8. 面试官可在 10 分钟内理解研究问题、系统架构和主要结论。

## 20. 简历与面试输出

在获得真实实验结果后，简历可以写成：

> 设计并实现面向 7B LLM 的预算感知 On-Policy Distillation 系统，解耦 Student rollout、Teacher sparse-logit annotation、数学 verifier 和 QLoRA training；通过 verifier-aware filtering 与 token-level confidence weighting，在固定 Teacher token 预算下比较 SFT、Offline KD、Vanilla OPD 与改进方法，并建立包含断点续跑、数据 lineage、污染审计、统计显著性和质量—成本 Pareto 分析的完整训练评测流水线。

面试重点准备：

- 为什么静态 Teacher trace 不属于 on-policy；
- Teacher 为什么在 Student rollout 上做 teacher-forcing；
- response mask 如何实现和测试；
- top-k KL 与 full KL 的差别；
- Verifier 为什么使用三态；
- 如何避免权重改变有效学习率；
- 为什么不能用 MATH-500 做 early-stop；
- 如何证明提升不是随机波动；
- 为什么拆分 rollout、annotation 和 training job；
- 相同质量下如何比较 Teacher tokens 和 GPU hours。

## 21. 第一批实际任务

按以下顺序开始实施：

1. 初始化 Git、`pyproject.toml`、目录结构和 CI；
2. 实现 Pydantic schema、artifact manifest 和 hash；
3. 加载 128 条 OpenR1-Math fixture，完成清洗和 split；
4. 实现 Math-Verify wrapper 和三态 verifier；
5. 用 tiny Student 生成 rollout 并支持 shard resume；
6. 实现 Teacher top-k annotation 和 response mask；
7. 在小词表/小样本上验证 sparse KL；
8. 跑通 SFT 和 Vanilla OPD smoke；
9. 实现 validation runner 和 early-stop；
10. 通过 Dev 阶段验收后，再租主实验 GPU。

完成以上十项之前，不启动 7B/14B 大规模实验。
