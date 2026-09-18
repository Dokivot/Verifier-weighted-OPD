# SuRe K2 OPD 训练审计结论

更新时间：2026-09-18

## 1. 总体结论

当前项目的 SuRe K2 核心算法和训练语义基本正确，但尚未达到可以直接执行正式实验并对外发布结果的标准。

当前状态应定义为：

- **算法实现：通过初步审计**
- **CPU 代码质量：通过**
- **真实 GPU 可运行性：未认证**
- **24 GPU 小时预算：未验证，存在较大风险**
- **正式评测协议：暂不通过**
- **SuRe 相对 Vanilla OPD 的创新结论：尚未成立**

本结论记录审计及修复状态。P0-1、P0-2、P0-4、P0-5 已完成代码/配置层修复；它们仍必须在 RTX PRO
6000 96GB 上通过 smoke 和 pilot 才能视为运行时验收通过。P0-3（运行中预算硬停止）按当前实验约定
保留为人工监控风险，不在本轮修复范围内。

## 2. 已确认合理的部分

### 2.1 训练目标

当前实现使用 sampled-token reverse-KL 的 K2 估计器，并使用 detached 的 SuRe 权重：

```text
0.5 * (log p_teacher - log p_student)^2
    * (1 + alpha * (1 - p_student))
```

实现满足以下条件：

- 每个 optimizer step 先由当前 Student 生成新轨迹；
- Teacher 和 Student 在相同 token、相同 prefix 上打分；
- 使用完整词表 log-softmax 计算 sampled-token log-prob；
- SuRe 权重不参与梯度传播；
- prompt token 不计入 loss，response token 按全局 token 数归一化；
- Teacher 冻结，Student 更新；
- 使用 BF16 计算、FP32 master parameters、梯度裁剪和非有限值检查。

### 2.2 工程质量

当前本地静态检查结果：

- 81 项单元测试通过；
- Ruff 检查通过；
- Ruff format 检查通过；
- mypy 检查通过；
- `uv lock --check` 通过；
- SuRe smoke、pilot、formal 脚本 Shell 语法通过；
- 配置文件可以正常解析。

这些结果只能证明 CPU/静态层面正确，不能替代真实 GPU 验证。

## 3. P0：正式训练前必须修复

### P0-1：补齐 Qwen3 终止 token 处理

**状态：已修复，待 GPU 验收。**

当前在线生成主要使用单一 `tokenizer.eos_token_id`。Qwen3 对话生成可能使用 `<|im_end|>`，而模型默认 EOS 可能是 `<|endoftext|>`。如果未同时处理，模型可能在语义回答结束后继续生成，造成无效长尾和高截断率。

要求：

- 显式支持 `<|im_end|>` 和 `<|endoftext|>` 两个终止 token；
- 保留 EOS/终止 token 作为 response supervision；
- 在 GPU smoke 和 pilot 中记录 finish reason、长度分位数和截断率；
- 如果大量样本以 `<|im_end|>` 结束但仍继续生成，禁止进入正式训练。

### P0-2：验证单卡实际吞吐和 24 小时预算

**状态：已修复，待 pilot 验收。** `run_sure_k2_pilot.sh` 会运行可恢复的两个 512-prompt step，并使用
step 2 的稳态耗时生成预算报告；其 1440 秒/step 的门槛为 55 steps 留出初始化、保存与评测缓冲。

正式配置为 55 steps、global batch 512。若训练本身要控制在 24 GPU 小时内，平均每步最多约：

```text
24 * 3600 / 55 = 1571 秒 ≈ 26.2 分钟
```

考虑评测、初始化和保存开销，实际应以低于约 22–24 分钟/step 为准。必须先完成 2-step、512-prompt pilot，并用第二步的稳态 `step_seconds` 重新估算。

### P0-3：预算硬上限必须能中途生效

当前预算检查主要发生在训练前后。长时间训练过程中若实际消耗超过 hard cap，不能及时停止。正式运行前应增加每 step 的预算检查，或使用严格的 invocation step limit，避免预算控制只停留在报告层面。

### P0-4：正式数据必须经过独立去污染

**状态：已修复，待数据审计验收。** 正式配置将 train input 固定为 `train_clean.parquet`；正式脚本将
fetch 两个 pinned benchmark 后强制执行 contamination stage。审计报告现在记录输入/eval checksum、近重复
开关、shingle size、阈值及完整匹配列表。

当前正式脚本没有把 MATH-500/AMC23 去污染审计作为必经 stage，且基础配置中的 contamination 路径可能指向旧实验目录。

规范流程应为：

```text
prepare DeepMath
→ fetch pinned benchmarks
→ exact/near-duplicate contamination audit
→ train from train_clean.parquet
```

必须保留被移除样本、匹配类型、阈值、数据 checksum 和 audit manifest。

### P0-5：检查磁盘空间和 checkpoint 策略

**状态：已修复，待目标磁盘验收。** 训练加载 Student 后、加载 Teacher 前，按实际参数量估算原子 rolling
替换时的双份完整 state、保留的 milestone model、final model 和安全余量（含尚未缓存的 8B Teacher）；不满足
配置的 80 GiB 或估算峰值时拒绝开始训练。rolling 仅保留最新一份完整可恢复 state，逐 step telemetry 长期
保留；step 55 不再重复保存 milestone model，因为 final 已覆盖该用途。

全参数训练会同时保存 model、optimizer 和 scheduler 状态。当前每一步保存完整 rolling state，原子替换期间还会出现旧/新状态并存，存在 200GB 数据盘空间不足风险。

正式运行前应：

- 增加可用空间预检；
- 明确 rolling checkpoint 的保留策略；
- 避免不必要地重复保存 step milestone、rolling 和 final；
- 保留完整 rolling state 的同时，单独保留轻量 telemetry。

### P0-6：完成真实 RTX PRO 6000 GPU smoke

**状态：已加入正式运行准入检查，待目标 GPU 执行。** 正式脚本现在要求 smoke 的 completed summary 和
pilot 的 passed 预算报告；因此不能在未完成真实 GPU smoke 时误启动 55-step 训练。

必须在目标服务器验证：

- Qwen3-1.7B Student 和 Qwen3-8B Teacher 能加载；
- BF16、CUDA、vLLM/Transformers 组合正常；
- full-parameter 反向传播不 OOM；
- 终止 token 行为正确；
- loss、gradient norm、SuRe 权重有限且范围正确；
- 单步 checkpoint 能保存、恢复并继续训练。

## 4. P1：发布简历/项目结果前必须修复

### P1-1：评测长度与论文协议对齐

当前评测最大生成长度为 8192，而 SuRe 论文使用最大生成长度 31744。当前设置可以作为低成本 MVP，但不应直接声称与论文结果可比。

正式报告至少应做到以下之一：

- 使用与论文接近的 context/generation length；或
- 明确声明采用 budget-adapted evaluation，并报告 truncation rate。

### P1-2：必须加入 matched Vanilla K2 对照

Base vs SuRe 只能说明 SuRe checkpoint 相对初始模型发生了变化，不能证明 SuRe weighting 优于 Vanilla OPD。

最小可信对照为：

- 相同 Student、Teacher、数据顺序、seed、步数和评测协议；
- Vanilla K2：`alpha=0`；
- SuRe K2：`alpha=1`；
- 唯一方法差异为 per-token weighting。

在 Vanilla 完成前，简历中只能写“实现并运行了 SuRe K2 OPD”，不能写“SuRe 优于 Vanilla OPD”。

### P1-3：比较器必须拒绝不匹配评测

当前比较逻辑取 baseline/candidate 的 sample ID 交集，可能静默丢弃缺失样本。应改为：

- 两侧 sample ID 集合必须完全一致；
- 每个 sample 的 candidate index 必须完整且一致；
- 每个 sample 的 reference answer/prompt hash 必须一致；
- 任一评测不完整时直接失败，不生成比较报告。

### P1-4：使用 finish reason 判断截断

仅使用 `response_tokens >= max_new_tokens` 会把恰好在上限生成 EOS 的样本误判为截断。评测产物应保存 backend 的 finish reason，并以 `finish_reason == "length"` 作为主要截断判据。

### P1-5：Base 评测应在训练前完成

规范顺序应为：

```text
prepare
→ fetch benchmark
→ contamination audit
→ Base evaluation
→ GPU smoke/pilot
→ train candidate
→ candidate evaluation
→ paired comparison
```

这样可以在消耗训练 GPU 时间前验证 prompt、grader、benchmark 和生成长度设置。

### P1-6：增加数学以外的回归评测

至少加入一个低成本 OOD/回归 suite，例如 IFEval 或固定 MMLU-Pro 子集，报告训练前后变化。数学能力提升不能以未检查的 instruction-following/general capability 回退为代价。

### P1-7：完善 verifier 校准

当前 verifier 单元测试较完整，但还需要对实际 rollout 随机抽样人工核验：

- pass 是否真的正确；
- fail 是否真的错误；
- unknown 是否主要来自截断/格式问题；
- 与 benchmark 官方或论文 grader 的一致性。

## 5. P2：建议增强项

- 记录 actor entropy、sampled-token student probability 分位数、参数更新范数和 CUDA peak memory；
- 记录 verifier 状态分布，而不只记录均值；
- 在 checkpoint state 中保存模型文件、optimizer 和 scheduler 的 SHA-256；
- 明确说明 orphan step artifact 的恢复处理方式；
- 让评测 run ID 包含 prompt/checkpoint manifest、prompt template、`max_samples` 和 resolved generation config；
- 保持单 seed 的同时，报告“没有估计 training-seed variance”的限制。

## 6. 修复后的准入标准

只有同时满足以下条件，才允许开始正式 55-step 训练：

1. P0 项完成或明确接受对应风险；
2. RTX PRO 6000 GPU smoke 通过；
3. 2-step、512-prompt pilot 通过并完成耗时估算；
4. 终止 token 正确，pilot truncation rate 不超过质量门禁；
5. 训练数据完成 contamination audit；
6. Base benchmark 已在训练前生成并可复现；
7. 磁盘空间足以容纳 rolling checkpoint 和临时原子替换；
8. 正式日志、telemetry、manifest 和 checkpoint 备份路径已确认。

## 7. 审计评级

| 领域 | 评级 | 结论 |
|---|---:|---|
| SuRe K2 数学实现 | A | 与目标 sampled-token reverse-KL 设计一致 |
| On-policy 训练语义 | A- | 语义正确，仍需 GPU 行为验证 |
| CPU 工程质量 | A- | 测试、lint、类型检查和锁文件通过 |
| GPU 可行性 | B- | 未在目标硬件完成真实认证，预算存在风险 |
| 数据治理 | C+ | 去污染尚未成为新 pipeline 的强制 stage |
| 评测协议 | C | 长度、配对完整性、OOD 和 baseline 顺序需修复 |
| 简历创新结论 | C | 未完成 Vanilla matched control 前不能归因于 SuRe |

**最终准入结论：暂不通过正式训练；先完成 P0 修复和 pilot。**
