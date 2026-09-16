# OPD-Lab 研究扩展版

> 状态：备用，不在当前阶段执行。

## 目标

在推荐完整版基础上，研究 Teacher 规模、主动样本选择、多领域冲突和多轮 replay 对 OPD 的影响，形成更接近研究论文规模的实验。

## 资源边界

- 预计总成本：600–1000 H100 GPU 小时；
- Student：8B，可增加 14B Student 对照；
- Teacher：14B 与 32B；
- 数据：60k–100k prompts；
- 每题 2–4 个 rollouts；
- OPD 2–3 轮；
- 所有主要方法 3 seeds；
- 消融实验使用 25%–50% 数据。

## 扩展研究问题

1. 14B 与 32B Teacher 在相同 Teacher token 预算下是否存在稳定差异？
2. Student uncertainty、Teacher disagreement 和领域难度能否用于主动选择 annotation 样本？
3. 新旧 rollout replay 比例如何影响稳定性和灾难性遗忘？
4. 数学与代码联合训练是否产生负迁移，动态领域采样能否缓解？
5. top-k sparse KL 在不同 k 下与 full-vocabulary KL 的差距如何变化？

## 数据与评测扩展

- 数学训练：OpenR1-Math、NuminaMath；
- 代码训练：NVIDIA OpenCodeReasoning；
- 数学评测：MATH-500、AIME、AMC、OlympiadBench；
- 科学评测：GPQA Diamond；
- 代码评测：HumanEval+、MBPP+、LiveCodeBench；
- 通用回归：IFEval。

## 新增系统能力

- Ray 或等价的多 worker rollout/annotation 调度；
- 多机 artifact queue；
- code execution sandbox；
- replay buffer 和样本优先级；
- 14B/32B Teacher 路由；
- 多领域动态采样；
- 更完整的成本预测和自动停止策略。

## 启动条件

推荐完整版已完成，且主方法在至少一个可信指标上显示稳定收益；否则继续扩大规模没有足够价值。
