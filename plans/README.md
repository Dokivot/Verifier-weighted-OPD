# OPD-Lab 方案索引

根目录 [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) 是当前入口。本目录同时保存当前详细计划与历史备用
设计；不同方案的模型、loss 和 artifact 不得混用。

| 状态 | 方案 | 文件 | 说明 |
|---|---|---|---|
| **当前** | 24h SuRe reverse-KL | [`PROJECT_PLAN_REVERSE_KL.md`](PROJECT_PLAN_REVERSE_KL.md) | Qwen3-1.7B-Base、Qwen3-8B、DeepMath hard split；Base vs 55-step SuRe |
| 备用 | 旧推荐完整版 | [`PROJECT_PLAN_RECOMMENDED.md`](PROJECT_PLAN_RECOMMENDED.md) | 早期 Qwen3 大规模、多 seed 设计；未启用 |
| 备用 | 研究扩展版 | [`PROJECT_PLAN_RESEARCH_SCALE.md`](PROJECT_PLAN_RESEARCH_SCALE.md) | 32B Teacher、多领域和更大规模；未启用 |

## 历史 Qwen2.5 主线

此前根计划使用 Qwen2.5 Student/Teacher、整轮预生成 rollout、top-k sparse forward-KL、VFS 与固定
verifier 权重。该实验产生了有价值的负结果，但不再是当前推荐训练方案。相关代码、runbook 和 artifact
暂不删除，以便审计和复盘；任何文档若仍把它写成“当前方案”，均视为待迁移历史文档。

## 当前升级顺序

1. 执行 8-sample smoke 和可恢复的两步 512-prompt pilot；
2. 丢弃 pilot 权重，从固定 Base 完成 55-step SuRe；
3. 统一评测 Base 与 SuRe；
4. 结果完整后再补 Vanilla K2、K1、RA-OPD、LoRA 或 alpha sweep。

GPU 小时不再用旧硬件区间外推。先测完整 512-prompt update 的 steady-state wall time，再按 55 步、
两组模型评测和 20% 重试余量估算。
