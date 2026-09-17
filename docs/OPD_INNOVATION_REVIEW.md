# OPD-Lab 文献调研与创新方案

> 检索日期：2026-09-17。本文区分“已有论文贡献”和“本项目研究假设”，不把工程组合包装成
> 已被证明的全新算法。

## 1. 原始 OPD 解决了什么问题

传统离线蒸馏在 Teacher 生成或固定的数据上训练 Student，训练时看到的 token prefix 与 Student
部署时自己生成的 prefix 不一致，形成 exposure/distribution mismatch。

原始 OPD/GKD 的核心改动是：

1. 让当前 Student 自己生成 rollout；
2. Teacher 在 Student 实际访问到的 prefix 上提供 token-level 分布；
3. Student 在这些 on-policy states 上最小化 Teacher–Student 分布差异；
4. 更新后的 Student 再生成新一轮状态。

代表论文：

- Agarwal et al., **On-Policy Distillation of Language Models: Learning from Self-Generated
  Mistakes / Generalized Knowledge Distillation**, ICLR 2024：
  <https://arxiv.org/abs/2306.13649>
- Gu et al., **MiniLLM: On-Policy Distillation of Large Language Models**, ICLR 2024：
  <https://arxiv.org/abs/2306.08543>

## 2. 已有工作已经覆盖哪些“看起来像创新”的方向

### 2.1 只把 forward KL 改成 reverse/skew KL

- MiniLLM 使用 reverse KL 和 on-policy 优化；
- DistiLLM 使用 skew KL 与 adaptive off-policy 数据复用，并报告最高约 4.3× 加速。

来源：Ko et al., **DistiLLM**, ICML 2024：<https://arxiv.org/abs/2402.03898>

结论：仅替换 KL 方向不适合作为本项目主创新，但可以作为 loss 消融。

### 2.2 只做 Teacher entropy/confidence 加权

Entropy-Aware OPD 已直接研究高 Teacher entropy 下 reverse KL 的不稳定性，并在高 entropy token
混入 forward KL 以保持 diversity。

来源：Jin et al., **Entropy-Aware On-Policy Distillation of Language Models**, ICML 2026：
<https://arxiv.org/abs/2603.07079>

结论：本项目现有 confidence weighting 应降级为 baseline/ablation，不能再声称是核心算法创新。

### 2.3 只用 verifier 给 trajectory 加权或 gating

- RG-OPD 使用 reward/verifier 判断何时信任 Teacher logits；
- OPDVR 将 verifiable reward 与 sampled-token OPD 结合；
- 这些工作都说明“正确性信号 + dense Teacher signal”有效，但也直接覆盖简单的 verifier gating。

来源：

- Akhondzadeh et al., **Reward-Gated On-Policy Distillation**：
  <https://arxiv.org/abs/2607.04037>
- Lin et al., **On-policy Distillation with Verifiable Reward**：
  <https://arxiv.org/abs/2608.24696>

结论：简单的 `pass=1/unknown=0.3/fail=0` 仍是合理 baseline，但不足以作为论文味创新。

### 2.4 只按 token 位置或 Teacher–Student gap 加权

- IW-OPD 指出 rollout 后部位置的 Teacher supervision 质量更差，并按累计分布差异进行位置加权；
- TrOPD、REOPOLD 和 vOPD 分别从 trust region、reward clipping/动态采样、control variate 角度
  稳定 OPD。

来源：

- Xie et al., **On the Position Bias of On-Policy Distillation**：
  <https://arxiv.org/abs/2606.22600>
- Xing et al., **Trust Region On-Policy Distillation**：
  <https://arxiv.org/abs/2606.01249>
- Ko et al., **Scaling Reasoning Efficiently via Relaxed On-Policy Distillation**：
  <https://arxiv.org/abs/2603.11137>
- Xing et al., **KL for a KL: On-Policy Distillation with Control Variate Baseline**：
  <https://arxiv.org/abs/2605.07865>

结论：位置衰减、KL clipping 或 control variate 可以作为未来扩展，但第一版不要重复实现多个
近期算法，否则实验矩阵和解释成本会失控。

### 2.5 只做 selective KD

Selective KD 已系统研究 sample、position 和 vocabulary class 三个选择轴，并使用 Student entropy
减少 wall time、显存和缓存存储。

来源：Tavor et al., **Rethinking Selective Knowledge Distillation**：
<https://arxiv.org/abs/2602.01395>

结论：本项目必须强调 on-policy state acquisition、verifier-first 和显式 Teacher token budget，
不能只写“选择部分样本/token”。

## 3. 选择的主创新：VFS-OPD

项目名称：

**Verifier-First State-Budgeted On-Policy Distillation（VFS-OPD）**

中文：**验证器优先、状态预算感知的在线策略蒸馏**。

### 3.1 原始 OPD 的工程浪费

原始 OPD 通常对所有 Student rollout 调用 Teacher。Teacher forward、top-k logit 缓存和后续
训练都发生之后，系统才知道某条 trajectory 是否正确、是否重复、是否被截断、是否值得训练。

当前项目甚至可能先为 fail rollout 计算完整 Teacher annotation，最后又因为 `fail=0` 在 loss
中将其全部丢弃。这不会改变模型，但已经消耗了 Teacher GPU 时间和磁盘。

### 3.2 VFS-OPD 的核心变化

VFS-OPD 把“选择”移动到 Teacher annotation 之前：

```text
prompt
  → cheap Student rollouts × K
  → Math verifier + answer/self-consistency grouping
  → state selection manifest under fixed Teacher-token budget
  → Teacher only annotates selected states
  → OPD training
```

它优化的不是“固定样本数下最高分”，而是：

```text
在固定 Teacher forward tokens / GPU hours 下，获得最大的 validation 和 benchmark 收益。
```

### 3.3 为什么是 state-budgeted，而不是普通 sample filtering

OPD 的有效数据单元是 Student 实际访问到的 prefix/state，不只是原始 prompt。VFS-OPD 对同一个
prompt 生成 `K=2` 个便宜 Student rollout，再按 rollout group 的状态选择要询问 Teacher 的状态。

这也直接回应一篇最新反例论文：Data-free OPD 报告 prompt 数量可能并不重要，少量 prompt 的
反复采样也能暴露新 Teacher correction。VFS-OPD 因此不假设“更多题目一定更好”，而是明确比较
prompt coverage 与 state coverage。

来源：**Data-free On-policy Distillation**：<https://arxiv.org/abs/2609.14193>

### 3.4 状态分组与确定性选择

每题生成两个 Student rollout，先运行 Math-Verify。按优先级分组：

1. `boundary`：同题同时出现 pass 与 non-pass，说明 Student 决策不稳定；
2. `uncertain`：没有 pass，但答案互不一致或至少一个 unknown；
3. `solved`：两个 rollout 都 pass；
4. `failed`：两个 rollout 都 fail 且答案一致。

在每组中优先选择：

1. 非截断 trajectory；
2. `pass`，其次 `unknown`，最后 `fail`；
3. estimated Teacher tokens 更少的 trajectory；
4. 固定 seed 派生的稳定 hash 作为最后 tie-break。

选择器按 subject/difficulty 做 round-robin，避免预算全部被代数短题占据。累计 estimated Teacher
tokens 达到预算后停止，并输出 selection manifest。所有规则在运行 benchmark 前固定。

### 3.5 Teacher token 预算

第一版使用三档预算：

| 档位 | Teacher token 预算 | 用途 |
|---|---:|---|
| B25 | dense annotation 估算量的 25% | 极低成本曲线 |
| B50 | 50% | 主实验预算 |
| B100 | 100% | dense OPD 成本上界 |

Teacher token 使用 `min(prompt_tokens + response_tokens, teacher.max_length)` 预估，最终以真实
annotation job metrics 为准。方法间比较必须使用同一真实 Teacher token 上限，而不是相同记录数。

## 4. 与已有方法的区别

| 方法 | 选择发生时间 | 是否节省 Teacher forward | 是否使用 verifier | 是否比较固定 Teacher tokens |
|---|---|---:|---:|---:|
| 原始 OPD/GKD | 不选择 | 否 | 否 | 通常否 |
| Entropy-Aware OPD | Teacher forward 后 | 否 | 否 | 否 |
| RG-OPD/OPDVR | Teacher forward 后参与 loss | 通常否 | 是 | 非核心 |
| Selective KD | offline/cache 阶段 | 可节省 | 通常否 | 可选 |
| **VFS-OPD** | **Teacher forward 前** | **是** | **是** | **是** |

本项目不声称“截至 2026 年全球首次提出该思想”。严谨表述是：在本次检索范围内，没有发现与
“数学 verifier 优先、同题多 rollout 状态分组、Teacher-forward 前选择、固定 Teacher token
预算和完整成本 lineage”完全同构的公开方案。正式写论文前仍需扩大检索。

## 5. 分阶段实验矩阵

| ID | 方法 | Teacher 预算 | 回答的问题 |
|---|---|---:|---|
| E0 | Base Student | 0 | 原始能力 |
| E1 | SFT | 0 Teacher forward | 传统监督基线 |
| E2 | Dense Vanilla OPD | B100 | 原始 OPD 上界 |
| E3 | Random-Budget OPD | B50 | 只减预算会损失多少 |
| E4 | Verifier-Filtered Budget OPD | B50 | 只按 pass/unknown 过滤是否有效 |
| E5 | **VFS-Weighted OPD** | **B50** | selection + 双粒度加权的整体收益 |
| E6 | VFS-OPD | B25 | 极低预算是否仍保留收益 |
| E7 | Plain VFS-OPD | B50 | 去掉双粒度加权后的组件消融 |

执行顺序不是一次跑完 E0–E7：Phase A 先完成 E0、E1、E5，得到完整简历结果；Phase B 补 E3
和 E2，建立等预算选择证据与 Dense 上界；Phase C 再补 E4、E6、E7、K 值和 Round 1 消融。
在 E3 完成前不得声称 VFS selection 优于随机选择，在 E7 完成前不得声称加权组件独立有效。

## 6. 评测与成功判据

### 6.1 主指标

- MATH-500 accuracy；
- AIME 2024 pass@1；
- validation composite score；
- Teacher tokens；
- Teacher annotation GPU hours；
- 每提升 1 个 MATH-500 百分点所需 Teacher tokens；
- 质量—成本 Pareto frontier。

GPQA Diamond 与 IFEval 只做能力回归，避免数学优化掩盖通用能力退化。

### 6.2 核心比较

VFS-OPD 的主要对手不是 B100 dense OPD，而是相同 B50 预算下的 Random-Budget OPD。积极结论
必须满足：

1. VFS-OPD 在真实 Teacher tokens 不高于 Random-Budget 的前提下取得更高 validation/MATH-500；
2. paired bootstrap 差值趋势为正；
3. 不是通过只选短题造成题型分布偏差；
4. 截断率、输出长度、IFEval 没有明显退化；
5. selection manifest 可重建且同配置重复运行得到相同选择。

### 6.3 可证伪结果

以下结果也必须如实报告：

- VFS 与 random selection 无差异：支持“OPD 对具体数据选择不敏感”的近期观察；
- VFS 在 B50 接近 dense B100：说明 Teacher annotation 有大量冗余；
- VFS 只在 validation 有效、MATH-500 无效：可能发生 selection overfitting；
- boundary states 无效但 solved states 有效：Student 可能更适合巩固正确模式，而非修复错误状态；
- 多 rollout 的 Student 成本抵消 Teacher 节省：方法没有端到端成本优势。

## 7. 工程实现要求

新增一个独立阶段：

```text
rollout → verify → select-annotations → teacher annotate → build-view → train
```

选择 artifact 至少保存：

- `rollout_id`、`sample_id`、candidate index；
- verifier status、extracted answer；
- group type：boundary/uncertain/solved/failed；
- estimated Teacher tokens；
- selection rank、selection reason；
- selected/rejected；
- budget before/after；
- source rollout 和 verification manifest ids；
- config hash、Git commit、seed。

必须提供：

- `opd data select-annotations --round N`；
- deterministic selection 单元测试；
- token budget 绝不超支测试；
- subject/difficulty round-robin 测试；
- interrupted run/resume 测试；
- random-budget baseline 使用相同预算的测试；
- selection report 和成本 Pareto 图。

## 8. 为什么适合简历项目

VFS-OPD 同时展示：

- 能读懂 OPD、KL、exposure bias 和 verifier reward 文献；
- 能发现系统中的真实成本浪费，而不是只改 loss 公式；
- 能设计固定预算的公平实验和可证伪假设；
- 能实现多 rollout 分组、deterministic selector、manifest lineage 和断点恢复；
- 能把模型质量、Teacher tokens、GPU hours、存储量放在同一 Pareto 分析中；
- 即使方法没有显著提升，也能给出可信的负结果和工程结论。

建议简历表述：

> 设计 Verifier-First State-Budgeted OPD：在 1.5B Student 的同题多 rollout 上先执行数学验证
> 与状态分组，再在固定 Teacher token 预算内选择性调用 7B Math Teacher；实现可恢复 annotation
> selection manifest、sparse-logit caching 与质量—成本 Pareto 评测，并在相同 Teacher tokens 下
> 对比 random-budget、verifier-filtered 和 dense OPD。

## 9. 不应做出的表述

- 不写“提出全球首个 verifier-aware OPD”；RG-OPD/OPDVR 已存在。
- 不写“首次提出 entropy-aware OPD”；已有同名工作。
- 不写“证明多 seed 稳定”；本项目只运行 seed 42。
- 不写“显著优于 SOTA”，除非完整 benchmark 和统计检验确实支持。
- 不把 paired bootstrap 解释为训练随机性的置信区间。
- 不在 Teacher revision 未固定、selection 规则未预注册时开始正式实验。
