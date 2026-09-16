# OPD-Lab 备用执行方案

当前实际执行方案位于仓库根目录的 [`PROJECT_PLAN.md`](../PROJECT_PLAN.md)。本目录保存未启用但可随时升级的方案。

| 方案 | 文件 | 预计 H100 GPU 小时 | 适用场景 |
|---|---|---:|---|
| 最小可信版 | [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md) | 60–110 | 当前执行；固定单 seed，优先形成完整、可复现的简历项目 |
| 推荐完整版 | [`PROJECT_PLAN_RECOMMENDED.md`](PROJECT_PLAN_RECOMMENDED.md) | 300–500 | 增加数据规模、多 seed、两轮 OPD 和完整消融 |
| 研究扩展版 | [`PROJECT_PLAN_RESEARCH_SCALE.md`](PROJECT_PLAN_RESEARCH_SCALE.md) | 600–1000 | 增加 32B Teacher、多领域和大规模实验 |

## 升级条件

只有当前方案同时满足以下条件，才升级到推荐完整版：

1. Student rollout、Teacher annotation、Verifier 和训练均可断点恢复；
2. Vanilla OPD 相比 Base/SFT 至少表现出可复现的非负收益；
3. Weighted OPD 在 validation 上有明确趋势，或产生值得进一步验证的失败结论；
4. 数据污染、response mask 和 sparse KL audit 均通过；
5. 剩余 GPU 预算足以覆盖至少两个额外 seed，而不是只扩大单次训练规模。

研究扩展版只有在推荐完整版完成后再考虑，不作为简历项目按时完成的必要条件。
