# OPD-Lab 当前项目计划

当前主线是在单张 RTX PRO 6000 Blackwell 96GB、单 seed `42`、约 24 GPU 小时内完成：

> `Qwen3-1.7B-Base` Student + `Qwen3-8B` Teacher + DeepMath hard split + SuRe sampled-token
> K2 reverse-KL。

首轮只比较未经训练的 Base 与 55-step SuRe OPD，优先得到完整 pipeline 和可写入简历的真实结果。
Vanilla K2、RA-OPD、LoRA 和 alpha sweep 留到主结果完成后。

## 固定范围

- 训练数据：DeepMath `difficulty >= 6`；
- 正式规模：`55 × 512 = 28,160` 个唯一 prompt；
- 训练方式：严格在线、每 prompt 一个 Student rollout、FP32 master + BF16 compute 全参数更新；
- loss：sampled-token K2 reverse-KL + SuRe detached weighting，`alpha=1.0`；
- 长度：prompt 2048、response 8192、model context 12288；
- 评测：MATH-500 `k=1`、AMC23 `k=4`；
- 停止：固定 55 步，不使用 validation early-stop。

## 执行入口

1. 完整参数、原理、门禁和文献：[`plans/PROJECT_PLAN_REVERSE_KL.md`](plans/PROJECT_PLAN_REVERSE_KL.md)
2. AutoDL 小白运行指南：[`docs/SURE_K2_24H_RUNBOOK.md`](docs/SURE_K2_24H_RUNBOOK.md)
3. 正式配置：[`configs/sure_k2_24h.yaml`](configs/sure_k2_24h.yaml)
4. 正式脚本：[`scripts/run_sure_k2_24h.sh`](scripts/run_sure_k2_24h.sh)

旧 Qwen2.5 sparse forward-KL 实验保留为历史负结果和工程回归测试，不与新模型、数据或 artifact 混用。
