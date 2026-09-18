# OPD-Lab：24 GPU 小时 SuRe K2 Reverse-KL 方案

> 当前实现与执行基准。目标是在单张 RTX PRO 6000 Blackwell 96GB、单 seed `42`、约 24 个活跃
> GPU 小时内，先得到一条完整、可恢复、可审计的 Base→OPD 结果。Vanilla K2 和其他消融保留到
> 主结果完成之后，不占用首轮预算。

## 1. 主线实验

| ID | 模型 | 训练 | 作用 |
|---|---|---:|---|
| E0 | `Qwen/Qwen3-1.7B-Base` | 否 | 原始基线 |
| E1 | 同一 Student + SuRe K2 | 55 步 | 当前主方法 |

Teacher 固定为 `Qwen/Qwen3-8B`。训练数据固定为 `zwhe99/DeepMath-103K` 中
`difficulty >= 6` 的 hard split。正式训练消费 `55 × 512 = 28,160` 个不重复 prompt，每题只从
当前 Student 采样一个 response。

该预算是论文完整训练的缩短版：它保留论文的模型、数据筛选、loss、global batch、学习率、warmup、
生成上限和采样参数，只减少 optimizer step。它适合简历项目的第一版效果验证，不应描述为完整复现。

## 2. 固定版本

| 角色 | Hugging Face ID | revision |
|---|---|---|
| Student | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| Teacher | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Train | `zwhe99/DeepMath-103K` | `5cf055d1fe3d7a2eb19719ac020211469736ae44` |
| MATH-500 | `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| AMC23 | `zwhe99/amc23` | `f9810c0439cd3c670ec885d328a2f06a87f3694a` |

Student 使用 FP32 master 参数和优化器状态、BF16 前后向计算的全参数更新；不得在 OOM 时静默切换
LoRA/QLoRA。Student 和 Teacher 的 tokenizer
vocabulary fingerprint 必须相同，否则 sampled token ID 无法对齐，训练直接失败。

## 3. 严格 On-Policy 循环

每个 optimizer step 都执行：

```text
当前 Student θ_k
  → 为 512 个新 prompt 在线采样 response
  → Teacher 计算每个 sampled token 的精确 log-prob
  → Student 在完全相同 token/context 上重算可微 log-prob
  → 以全局 response-token mean 累积 SuRe K2 loss
  → 一次全参数 AdamW update，得到 θ_(k+1)
```

禁止预先生成完整 epoch 后连续消费旧 rollout。每步 Parquet 都写入生成时的 policy hash；resume 时重新
计算 SHA-256 hash chain，出现多余、缺失或被修改的 step artifact 就 fail closed。

Prompt 固定为：

```text
{problem}
Please reason step by step, and put your final answer within \boxed{}.
```

随后应用 Qwen tokenizer 的 chat template，并显式设置 `enable_thinking=false`。模型仍被要求逐步推理；
这里关闭的是 Qwen3 模板额外的 thinking 开关，以匹配论文设置。

## 4. Loss

对 Student 采样 response 中的 token `y_t`：

```text
delta_t  = log π_T(y_t | c_t) - log π_S(y_t | c_t)
k2_t     = 0.5 × delta_t²
weight_t = 1 + α × (1 - stop_gradient(π_S(y_t | c_t)))
loss     = sum(weight_t × k2_t) / 有效 response token 总数
```

主实验固定 `α=1.0`，合法 SuRe weight 位于 `[1, 2]`。`α=0` 与 vanilla K2 的数值和梯度等价，
已由单元测试覆盖。prompt、padding 不进入 loss，EOS 进入 loss。

Teacher 与 Student 都只保存实际 sampled token 的 log-prob，不保存 full-vocabulary logits。前向计算先
取得 hidden state，再把 LM head 按 token chunk 计算，避免在 8K response 上常驻巨大的
`tokens × vocabulary` logits tensor。

## 5. 训练参数

参数定义在 [`../configs/sure_k2_24h.yaml`](../configs/sure_k2_24h.yaml)：

| 参数 | 值 |
|---|---:|
| seed | `42` |
| optimizer steps | `55` |
| global prompt batch | `512` |
| unique prompts | `28,160` |
| rollouts / prompt | `1` |
| optimizer / learning rate | AdamW / `1e-6` |
| warmup | `10` steps |
| betas / weight decay | `[0.9, 0.999]` / `0.01` |
| gradient clip | `1.0` |
| parameter / compute dtype | FP32 master / BF16 compute |
| max prompt / response | `2048` / `8192` |
| max model length | `12288` |
| temperature / top-p | `1.0` / `1.0` |
| checkpoint | 每步 rolling；step `34`、`55` model-only |

论文直接给出的科学参数来自 *A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy
Distillation*。论文未明确的 AdamW 细节使用调研时固定的 VeRL 默认值。microbatch 和 token chunk
只属于单卡吞吐/显存适配，不改变 loss 或数据分布。

不使用 validation early stop。第 55 步是 24h 工程预算截点，不是论文公开 benchmark checkpoint。

## 6. 质量与恢复门禁

训练前会对全部 28,160 个 prompt 预分词；任何 prompt 超过 2048 tokens 都会在加载模型前失败。
每一步记录 loss、gradient norm、LR、delta 分布、SuRe weight 分布、verifier score、response 长度、
截断率、分阶段 wall time、response/token IDs、两侧 sampled-token log-prob 和 policy hash。

截断率超过 5% 告警；至少 512 个样本后超过 20% 立即停止。NaN/Inf、空 response、tokenizer 不一致、
SuRe weight 越界、非连续 step 或 hash chain 不一致都会立即失败。

rolling checkpoint 使用目录级 staging/backup 原子替换，包含 Student、optimizer、scheduler、data cursor、
global step、Python/NumPy/PyTorch/CUDA RNG 和历史 telemetry。中断后从 `rolling` 恢复，不重跑已提交 step。

## 7. 三道运行门

1. **CPU 工程门**：`make check`；
2. **8-sample GPU smoke**：真实 Student/Teacher，512-token response cap，验证反传和保存；
3. **512-prompt pilot**：正式 8192 cap，运行 step 1，退出并从 rolling 恢复 step 2。

用 pilot 实测 step time 判断 55 步是否落在租卡预算内。pilot 权重与正式目录隔离，不进入结果。

## 8. 首轮评测

| Benchmark | 采样 | 指标 |
|---|---:|---|
| MATH-500 | `k=1` | avg@1、pass@1 |
| AMC23 | `k=4` | avg@4、pass@4 |

评测固定 `temperature=0.7`、`top_p=0.9`、`max_new_tokens=8192`。Base 和 SuRe 使用相同 prompt、
seed 派生规则、verifier 和生成配置。每次采样保存 `candidate_index`、seed、response、token 数、提取答案、
正确性和截断状态。

```text
avg@k  = 所有题全部采样的平均正确率
pass@k = 至少一个采样正确的题目比例
```

比较报告按题聚合，输出 avg@k、pass@k、paired bootstrap 和 McNemar。单 seed 不能证明跨 seed 稳定性。

## 9. 预算决策

pilot 后按下式估算：

```text
训练预计小时 = 稳态 step_seconds × 55 / 3600
总预计小时   = 训练 + Base 评测 + SuRe 评测 + 20% 重试余量
```

若明显超过 24h，先改用 34 步/17,408 prompts 备用截点并标注预算适配；不要中途改变 loss、模型、
学习率、global batch 或长度上限。

## 10. 后续实验

主结果完成后依次补：同 55 步 Vanilla K2、step 34 成本—质量点、RA-OPD K1、LoRA K2、SuRe alpha
sweep。简历第一版可陈述严格在线 reverse-KL 系统、SuRe weighting、原子 resume、policy lineage、
质量门禁和统一成本—质量评测；效果数字只能在真实 benchmark 完成后填写。

## 11. 参考资料

1. *A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation*：https://arxiv.org/abs/2608.25643
2. *When Teacher Guidance Misleads: Reward-Aligned On-Policy Distillation*：https://arxiv.org/abs/2608.27960
3. *MiniLLM: Knowledge Distillation of Large Language Models*：https://arxiv.org/abs/2306.08543
4. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*：https://arxiv.org/abs/2306.13649
