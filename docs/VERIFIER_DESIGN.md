# 数学 Verifier 设计与审计

> 检索日期：2026-09-18。本文区分规则型 outcome verifier、学习型 outcome verifier 和
> process reward model，避免把“最终答案正确”误写成“整条推理过程正确”。

## 1. 文献结论

### 1.1 最终答案 verifier 是合理的低成本起点

- Cobbe et al., **Training Verifiers to Solve Math Word Problems** 证明了对候选解答做 outcome
  verification 可以显著改善数学问题求解；但论文使用的是学习型 verifier，而不是本项目的
  符号等价规则。<https://arxiv.org/abs/2110.14168>
- Agarwal et al., **On-Policy Distillation of Language Models** 让 Teacher 在 Student 自己生成的
  trajectory/state 上提供 token-level supervision。Student rollout 最终是否答对不是 OPD 数据是否
  有价值的充分条件，因此 verifier 更适合做选择、分层和可靠性信号，而不是替代 Teacher。
  <https://arxiv.org/abs/2306.13649>
- Akhondzadeh et al., **Reward-Gated On-Policy Distillation** 进一步指出，Teacher 可能认可错误轨迹，
  也可能不认可一条正确但路径不同的轨迹；合理的 gate 应联合 verifier reward 与 Teacher–Student
  likelihood 关系，而不是仅用固定 `pass/fail` 权重。<https://arxiv.org/abs/2607.04037>

结论：本项目使用符号 outcome verifier 作为低算力、可复现的第一层信号是合理的，但不能宣称它
验证了每一步推理，也不能据此认为所有 `fail` trajectory 都没有蒸馏价值。

### 1.2 提取器与等价判断必须分开

Hugging Face **Math-Verify** 的官方文档指出，只接受 `Final answer is X` 等固定格式会严重低估模型
能力，极端情况下可相差 40 个点；prediction 侧建议联合 `LatexExtractionConfig` 与
`ExprExtractionConfig`，并在可控 prompt 中要求模型用 `\boxed{}` 输出答案。
<https://github.com/huggingface/Math-Verify>

因此 verifier 实际包含两个不同问题：

1. **Extraction**：模型最终声称的答案是什么；
2. **Equivalence**：该答案与 gold 在数学上是否等价。

旧实现先用有限正则做 extraction，只有提取成功后才调用 Math-Verify 做 equivalence。这使
Math-Verify 的鲁棒 extraction 完全无法处理未带固定 marker 的答案，也是 1,457 个 `unknown` 中
大量未截断样本的主要风险来源。

### 1.3 Outcome verifier 不能判断过程质量

Lightman et al., **Let's Verify Step by Step** 区分：

- outcome supervision：只判断最终答案；
- process supervision：判断每一步推理。

论文在 MATH 上报告 process supervision 优于 outcome supervision，但 PRM800K 依赖昂贵的步骤级
标注。<https://arxiv.org/abs/2305.20050>

本项目预算有限，因此第一版不引入 PRM。即使 `status=pass`，也只表示最终答案等价；中间过程仍
可能包含无效推理或碰巧猜中。Teacher token-level distribution 是蒸馏信号，不能自动把 outcome
verifier 变成 process verifier。

## 2. Verifier v2

`src/opd/verifier/math.py` 使用三层、从高到低的提取策略：

1. `strict_boxed` / `strict_marker`：优先解析嵌套 `\boxed{}`、`Final answer`、`Answer is`、
   `####` 和中文答案标记；
2. `math_verify_anchored`：调用 Math-Verify 的 anchor-aware parser；
3. `math_verify_tail_fallback`：只对最后一个非空行启用无 anchor 的 LaTeX/表达式解析。

第三层不直接扫描完整 reasoning，目的是降低把中间公式误当最终答案的风险。每条记录保存：

- `extraction_mode`；
- `extraction_confidence`：`high`、`medium` 或 `none`；
- `parser_status`：`matched`、`no_match` 或 `error`；
- `reason` 和 `error_type`。

只有三层均无法提取时才输出 `unknown/answer_not_found`。成功提取但与 gold 不等价时输出 `fail`；
等价时输出 `pass`。Verifier 版本升级为 `2`，manifest 同时记录 status 与 extraction-mode 分布。

## 3. 当前实验如何处理

当前正在训练的 MVP 使用 v1 verification、selection 和 Teacher annotation，**不要中途替换这些
artifact，也不要覆盖原路径**。它应作为 `Verifier-v1 VFS-Weighted` 完整跑完并保留。

训练结束后可在同一批 rollout 上执行只消耗 CPU 的离线审计：

```bash
cd /root/autodl-tmp/OPDProj
git pull
source scripts/autodl_env.sh

uv run --no-sync opd verify math \
  --round 0 \
  --config configs/verifier_v2_audit.yaml
```

结果写入：

```text
artifacts/resume_mvp/audits/verifier_v2/round_0/math.parquet
artifacts/resume_mvp/audits/verifier_v2/round_0/manifest.json
```

它不会覆盖：

```text
artifacts/resume_mvp/data/verifications/round_0/math.parquet
```

## 4. 审计判据

至少报告以下混淆转移和人工抽检：

1. v1 `unknown` 中有多少变为 v2 `pass/fail`；
2. v2 的 `strict_*`、`math_verify_anchored`、`math_verify_tail_fallback` 各有多少条；
3. 从每种 mode × status 固定 seed 抽取至少 50 条人工检查；
4. 分别估计答案提取 precision、`pass` precision 和 `fail` precision；
5. 单独报告 truncated 与 non-truncated 样本，不能把截断问题归因于 parser；
6. 比较 v1/v2 重新分组后 boundary、uncertain、solved、failed 的变化。

Verifier v2 只有在人工抽检确认没有明显增加“中间值误判为最终答案”后，才用于新的 selection 和
训练。不能因为 `unknown` 数下降就直接认定 verifier 更好。

## 5. 后续升级路径

按投入从低到高排序：

1. rollout prompt 明确要求 `Put the final answer in \boxed{}`，并重新生成 rollout；
2. 使用 v2 重做 selection，比较同一 Teacher-token 预算下的 v1/v2 差异；
3. 审计并校准 extraction mode；只有证据充分时才将 reliability 纳入选择优先级；
4. 实现 RG-OPD 风格的 reward/likelihood gate，让错误轨迹在 Teacher 确实提供修正方向时仍能训练；
5. 预算允许时加入 PRM，对步骤级错误做过程监督。

第 4 项比简单设置 `fail=0.1` 更有文献依据。固定小权重无法判断 Teacher 是在纠错还是强化错误，
不应作为主方法直接采用。

## 6. Outcome status 是否应直接作为 KL 权重

### 6.1 当前权重不是标准 OPD，也不是 RG-OPD

当前主配置使用：

```yaml
weighting:
  verifier:
    pass: 1.0
    unknown: 0.3
    fail: 0.0
```

它把每条 trajectory 的 Teacher–Student KL 乘以一个终局状态权重。这个设计可以作为“只强信任
答案正确样本”的保守消融，但没有文献证明 `1.0/0.3/0.0` 是一般最优值：

- Agarwal et al. 的标准 GKD/OPD 在 Student 自己访问的状态上使用 Teacher token distribution，核心
  动机正是从 self-generated mistakes 学习；标准目标不会因最终答案错误而把整条 trajectory 清零。
  <https://arxiv.org/abs/2306.13649>
- STaR 确实过滤错误答案，但它是在正确 rationale 文本上做 self-training/SFT，而不是在错误前缀上
  匹配外部 Teacher 的 token distribution。不能直接把 STaR 的 correct-only filtering 移植成 OPD
  的 `fail=0`。<https://arxiv.org/abs/2203.14465>
- RG-OPD 说明无条件蒸馏也可能有害，但它不是只保留 `pass`。它联合 verifier reward 与
  Teacher–Student sampled-action likelihood gap：Teacher 确实反对错误 action 时，失败 trajectory
  仍会被保留；Teacher 反对一个正确 trajectory 时，该成功样本反而会被过滤。
  <https://arxiv.org/abs/2607.04037>
- Lightman et al. 指出错误终局标签只说明“至少有一步错了”，不能定位首个错误；`fail` 是弱的
  trajectory-level 信号，而不是“该 trajectory 上每个状态都无价值”的证明。
  <https://arxiv.org/abs/2305.20050>

此外，本项目当前实现的是稀疏 **forward KL**（Teacher || Student），RG-OPD 论文实验使用 top-k
**reverse KL**（Student || Teacher）。因此在没有重新推导和消融前，不能把固定 status weight 称为
RG-OPD 的实现。

### 6.2 当前数据上为什么风险很大

当前已选择并完成 Teacher 标注的 3,064 条记录中：

| status | 记录数 | 当前权重 | record-equivalent contribution |
| --- | ---: | ---: | ---: |
| `pass` | 324 | 1.0 | 324.0 |
| `unknown` | 784 | 0.3 | 235.2 |
| `fail` | 1,956 | 0.0 | 0.0 |

按记录数粗略估计，当前 loss 只保留了 `559.2 / 3064 = 18.25%` 的有效权重；63.84% 已经付费生成
Teacher logits 的记录被完全清零。严格比例应按 response token 数重算，但不改变“选择策略与训练
权重严重不一致”的结论。`unknown=0.3` 同样缺少校准依据：它混合了格式、提取失败、截断和真实
不可判定样本，不是 30% 正确率或 30% Teacher 可靠度。

### 6.3 实验决策

1. 正在运行的 `configs/vfs_weighted_mvp.yaml` 不改，完整保留为 `status-weighted` 基线。
2. 推荐主对照改用 `configs/vfs_ungated_mvp.yaml`，令 `pass/unknown/fail` 均为 `1.0`；它仍保留原有
   token entropy weighting，只移除 trajectory status gate。
3. 该对照复用完全相同的 rollout、selection 和 Teacher annotation，不重新消耗 Teacher 推理时间；
   只需重建 training view、训练并评测。
4. 两组保持 seed、训练步数、学习率、样本顺序策略和 benchmark 一致。先比较 MATH-500，随后检查
   AIME、IFEval、格式通过率和训练稳定性，不能只看 training loss。
5. 若要实现真正的 reward-aware gate，后续需额外保存 rollout-time Student sampled-action logprob，
   从 Teacher annotation 取同 action logprob，并针对本项目 forward-KL 目标重新验证 gate。

低成本对照命令：

```bash
uv run --no-sync opd data build-view \
  --round 0 \
  --method weighted-opd \
  --config configs/vfs_ungated_mvp.yaml

scripts/train.sh configs/vfs_ungated_mvp.yaml

scripts/merge_checkpoint.sh \
  configs/vfs_ungated_mvp.yaml \
  artifacts/resume_mvp/checkpoints/vfs_ungated_b50_seed42/best \
  artifacts/resume_mvp/merged/vfs_ungated_b50

scripts/evaluate_lighteval.sh \
  artifacts/resume_mvp/merged/vfs_ungated_b50 \
  artifacts/resume_mvp/lighteval/vfs_ungated_b50 \
  configs/vfs_ungated_mvp.yaml
```

这个对照不证明全状态权重必然更好，但它是当前成本最低、归因最清楚的实验；在没有对照结果前，
不应把 `1.0/0.3/0.0` 写成已验证能提升效果的创新点。

## 7. 是否把 extraction confidence 加入训练权重

### 7.1 当前结论：暂不加入

当前 `high` 和 `medium` 是由提取路径人工命名的类别：

- `high` 表示答案来自 boxed、明确 marker 或 anchor-aware parser；
- `medium` 表示答案来自最后一个非空行的无 anchor parser。

它们不是由标注集估计的正确概率，也没有经过 calibration。因此不能直接假设：

```text
high = 1.0
medium = 0.7
```

如果直接相乘，`0.7` 只是超参数猜测，不是 verifier confidence。它会同时改变 verifier、有效数据
分布和 loss，且当前单 seed、优先跑通主 pipeline 的实验无法可靠归因。

### 7.2 文献为何不支持直接使用规则标签

- Math-Verify 推荐 prediction 侧联合 LaTeX/表达式提取，并建议在 reward verification 中使用更严格
  的 normalization；它没有把不同 extraction mode 定义为概率，也没有提供 `boxed=1.0`、
  `fallback=0.7` 一类权重。<https://github.com/huggingface/Math-Verify>
- **Self-Consistency Improves Chain of Thought Reasoning** 使用多条独立 reasoning path 的答案一致性
  作为聚合依据，而不是用输出格式估计正确性。项目当前每题只有 `K=2`，可以做 boundary grouping，
  但不足以形成稳定的连续置信估计。<https://arxiv.org/abs/2203.11171>
- Selective prediction 文献要求用 risk–coverage、AUROC/AURC 或校准误差验证 confidence 是否真的能
  排序错误风险；“看起来更严格的规则”不能替代校准。近期工作还报告单独使用 entropy 可能不足以
  支持安全选择。<https://arxiv.org/abs/2603.21172>
- **Reward-Gated OPD** 使用 verifier reward 与 Teacher–Student sampled-action likelihood gap 的方向
  一致性决定是否蒸馏。这里的 confidence margin 作用于 likelihood gap，不是答案 parser 的模式。
  <https://arxiv.org/abs/2607.04037>
- 传统 confidence-weighted KD 的 confidence 通常来自 Teacher predictive probability，例如
  **Born Again Neural Networks** 的 teacher-max confidence；它也不能直接证明 parser mode 可以作为
  KD 权重。<https://arxiv.org/abs/1805.04770>

### 7.3 什么时候才可以加入

先对 v2 结果进行固定 seed 的分层人工标注。对每个 `extraction_mode × status` 至少抽取 50 条，人工
判断“提取结果是否确实是模型意图表达的最终答案”，然后计算：

```text
extraction precision(mode)
pass precision(mode)
fail precision(mode)
95% Wilson confidence interval
```

只有同时满足以下条件，才值得增加 mode weight：

1. `tail_fallback` 样本量足够大，对最终 loss 有可见影响；
2. 其 precision 明显低于 strict mode，且置信区间差异不是抽样噪声；
3. precision 在 subject、difficulty 和 truncated/non-truncated 切片中相对稳定；
4. 权重由审计集预先确定，不根据最终 MATH-500 分数反复调参；
5. 至少补一个“有/无 confidence weight”的消融。

如果证据成立，可把人工校准后的 reliability `c(mode)` 仅用于已提取样本：

```text
effective_weight = status_weight × c(extraction_mode)
```

不要让 `extraction_confidence=none` 把 `unknown=0.3` 自动乘成零；unknown 的权重属于另一项研究
假设。第一版更推荐把校准后的 confidence 用于 Teacher-budget selection 的 tie-break，而不是立即
修改训练 loss，因为前者更符合本项目 verifier-first 的研究问题，也更容易做等预算比较。
