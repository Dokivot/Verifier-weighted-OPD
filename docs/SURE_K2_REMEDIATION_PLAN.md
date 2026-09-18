# SuRe K2 / OPD 整改方案设计

> 本文是对 `docs/SURE_K2_AUDIT_CONCLUSIONS.md` 中问题的解决方案设计，不表示这些修改已经全部实现。
> 目标是在单卡 RTX PRO 6000、单 seed 和约 24 GPU 小时的约束下，形成一个可复现、可解释、可以用于简历展示的工程实验。

## 1. 证据边界

### 1.1 直接相关的 SuRe/K2 论文

当前项目最直接的依据是：

- Shao et al., *A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation*, arXiv:2608.25643v2。
- 论文使用 Qwen3-8B teacher、Qwen3-1.7B-Base/Qwen3-4B-Base student，DeepMath difficulty `>= 6` hard split，训练 batch 512、learning rate `1e-6`、2 epochs、最大 prompt 2048、最大 response 8192、最大 model length 12288。
- 论文的 SuRe 权重为 `w_t = 1 + alpha * (1 - stop_gradient(p_student(y_t | c_t)))`，`alpha=0` 是 Vanilla OPD；论文扫过 `{0.2, 0.5, 1.0, 2.0}`，主实验使用 `alpha=1.0`。
- 论文报告的训练采样为 temperature 1.0、top-p 1.0；数学评测使用 temperature 0.7、top-p 0.9、最大生成长度 31744；MATH-500 使用 avg@4/pass@4，AMC23/AIME 使用 avg@8/pass@8，OOD 使用 pass@1。
- 论文明确把目标称为 sampled-token reverse-KL/K2 estimator，并没有把它等同于完整 vocabulary reverse KL。当前项目应沿用这个准确表述。

### 1.2 相关方向给出的风险信号

以下工作更适合作为风险和后续扩展依据，而不是本项目当前的主方法：

- *Trust Region On-Policy Distillation*（arXiv:2606.01249）指出 teacher/student 分布差异过大时，OPD 的 token supervision 可能不可靠，提出 trust-region、outlier masking 和 forward-KL fallback。
- *On the Position Bias of On-Policy Distillation*（arXiv:2606.22600）指出后段 token 的 teacher/student mismatch 可能更大，支持记录按位置的 gap、loss 和截断率，而不是只看全局均值。
- *On-policy Distillation with Verifiable Reward*（arXiv:2608.24696）将 verifier correctness 与 OPD signal 结合，说明 verifier 可以作为辅助信号，但不能把启发式权重直接当成已验证的最优目标。
- *Multi-Rollout On-Policy Distillation via Peer Successes and Failures*（arXiv:2605.12652）说明多 rollout 的成功/失败关系可能提供额外信号；当前项目要求 `rollouts_per_prompt=1`，因此不能声称实现了该类 peer-conditioned 方法。

### 1.3 工具链依据

- LightEval v0.9.2 的 vLLM backend 从 `generation_parameters.max_new_tokens` 或 task 的 `generation_size` 获取生成长度，并将其传给 vLLM `SamplingParams.max_tokens`；如果 `context + max_new_tokens > max_model_length`，会先截断 context，若预算大于 model length 会直接报错。
- vLLM 的 `SamplingParams` 明确区分 `max_tokens`、`stop` 和 `stop_token_ids`；因此评测命令必须显式固定生成预算和停止策略，不能依赖 task 默认值。

## 2. 总体整改原则

1. **先保证可归因，再追求新算法**：先固定 prompt、数据 revision、模型 revision、采样协议、评测样本和 seed，再比较 Vanilla K2 与 SuRe K2。
2. **训练 validation 与最终 benchmark 分离**：validation 只用于 checkpoint 选择和训练诊断；MATH-500、AMC23、IFEval 只在最终对比阶段使用。
3. **所有失败都显式暴露**：缺失样本、manifest 不匹配、生成长度不一致、verifier unknown 和评测回退都应使 gate 失败，而不是静默丢弃。
4. **预算在 step 边界安全停止**：不做可能损坏 checkpoint 的强制 kill；每个 step 前后检查预算，先保存可恢复状态，再退出。
5. **简历表述与目标函数一致**：使用“sampled-token reverse-KL K2 estimator”和“detached surprise-aware reweighting”，不要写“完整 reverse KL”或“证明了泛化提升”。

## 3. 问题到解决方案

### P0-1/P1-1：固定 LightEval 生成协议

**问题**：当前 wrapper 没有显式传递 `generation_parameters.max_new_tokens`、temperature 和 top-p，可能再次触发 context size 错误，也可能让 Base 与 candidate 使用不同的 task 默认值。

**设计**：

- 在 `benchmark.generation` 中增加 `max_new_tokens`、`temperature`、`top_p`、`top_k`、`stop_token_ids` 配置。
- `src/opd/evaluation/lighteval.py` 把这些值序列化到 LightEval vLLM model config 的嵌套 `generation_parameters={...}`，并把解析后的值写入 `command.json` 和 manifest metadata。
- 运行前做静态校验：`max_prompt_tokens + max_new_tokens <= max_model_length`。评测配置采用 `max_model_length=32768`、`max_new_tokens=31744` 只作为论文协议复现档；24 小时 MVP 可以使用 `max_model_length=12288`、`max_new_tokens=8192`，但报告中标明是 budget-adapted protocol，不能和论文数字直接比较。
- Base、Vanilla、SuRe 必须共享完全相同的 `generation_parameters`，包括 seed 和 chat template。
- 仍保存 backend finish reason，并只将 `finish_reason == "length"` 计为 truncation。

**验收**：command 中能看到完整 generation config；故意把 `max_new_tokens` 设为大于 model length 时 readiness gate 失败；同一 benchmark 的 Base 与 candidate command hash 除 checkpoint 外完全一致。

### P0-3：把 hard cap 变成安全的 step-boundary stop

**问题**：当前 GPU-hour 只在作业前后统计，训练期间超预算不会自动退出。

**设计**：

- 在训练 loop 每个 optimizer step 前后读取 monotonic elapsed time、实际 GPU 数和已完成 job metrics，计算 `projected_gpu_hours`。
- 到达 `gpu_hour_hard_cap` 或 `invocation_step_limit` 时，先完成当前 step 的 telemetry 和原子 rolling checkpoint，再写入 `budget_stop.json`，随后正常退出；不使用 `kill -9`。
- 每个 step 额外保存 `step_seconds`、`elapsed_gpu_hours`、`projected_final_gpu_hours`、`stop_reason`。
- 外部 `watch` 只作为辅助告警，不作为唯一停止机制。人工停止仍可用，但应在报告中标记为 `manual_stop`，不能标成 hard cap enforced。
- 对正式 24 小时配置，先由 2-step pilot 的稳态 step time 估算可完成 steps；`hard_cap` 只控制最大开销，不保证一定完成 55 steps。

**验收**：用很小的 hard cap 做测试，训练能在 step 边界产生完整可恢复 checkpoint；重新运行能从该 checkpoint 继续，而不是从头开始。

### P0-4/P0-5/P0-6：数据、磁盘和硬件准入统一为 manifest gate

**问题**：数据去污染、磁盘容量和 GPU smoke 虽有脚本，但 readiness 校验还不够强。

**设计**：readiness gate 必须同时验证：

- train-clean、MATH-500、AMC23、IFEval 输入文件的 SHA-256、record count 和 pinned revision；
- contamination manifest 的输入/输出 checksum 与当前 config hash 一致；
- student/teacher model revision、tokenizer revision、dtype 和 device 与当前 config 一致；
- smoke summary 为 `completed`，pilot 恰好完成配置要求的 step 数，且每个 summary 的 `run_id`、config hash、hardware fingerprint 一致；
- 磁盘检查同时覆盖模型、optimizer、rolling checkpoint 原子替换峰值和保留日志，不只检查当前可用空间。

**验收**：修改任一 benchmark revision、训练路径、模型 revision 或 generation config 后，旧 manifest 不能被错误复用，gate 必须要求重新生成对应 stage。

### P1-2：增加 matched Vanilla K2 control

**问题**：Base→SuRe 只能说明模型变化，不能证明 surprise reweighting 带来增益。

**设计**：

- `vanilla_k2` 与 `sure_k2` 使用同一 student、teacher、DeepMath-hard split、数据顺序、prompt schedule、seed、batch、learning rate、step 数和 checkpoint 规则。
- 唯一算法差异是 `sure_alpha=0` 与 `sure_alpha=1.0`；代码层面应额外断言 Vanilla 的 `sure_alpha == 0`。
- 两次训练分别生成 checkpoint，但使用同一套 frozen evaluation inputs、generation parameters 和评测 seed。
- 主比较顺序为 `Base vs Vanilla`、`Base vs SuRe`、`Vanilla vs SuRe`。只有第三个比较才能支持“SuRe 相对 Vanilla 的贡献”。
- 如果预算不足以跑完整 Vanilla，至少先跑 matched 16-step pilot；该结果只能作为 pipeline sanity check，不能作为最终效果结论。

**验收**：comparison manifest 明确记录 `method_diff = [sure_alpha]`，并拒绝比较不同 data revision、step 数或 generation config 的运行。

### P1-3：比较器改为严格配对

**问题**：当前取 sample ID 交集，会静默丢弃缺失题目。

**设计**：

- 每条 prediction 具有 `sample_id`、`candidate_index`、`prompt_sha256`、`reference_sha256`、`generation_config_hash`。
- 比较前要求两侧 sample ID 集合完全相同；每个 sample 的 candidate index 集合完全相同；prompt/reference checksum 和 eval manifest 完全相同。
- 任意缺失、重复、顺序不一致或 checksum 不一致直接失败，并输出 mismatch report；不再用交集继续统计。
- 保留逐题 paired outcomes，报告 accuracy/avg@k/pass@k、绝对差值、相对差值、paired bootstrap CI 和 McNemar test。统计区间只代表题目采样不确定性，不代表 training-seed variance。

**验收**：单元测试覆盖缺样本、重复 candidate、reference 不同、prompt hash 不同四种失败情况。

### P1-4：verifier 校准并分离内部指标和官方指标

**问题**：自定义 `MathVerifier` 的 pass/fail/unknown 可以用于训练分析，但不能直接当作论文或 benchmark 主结果。

**设计**：

- 保留内部 verifier，用于在线质量门禁和 token weighting 的诊断；其结果目录命名为 `internal_verifier`。
- 每次正式运行从 pass/fail/unknown/截断各分层抽样，人工核验固定数量，保存人工标签、verifier 标签、错误类型和 example ID。
- 生成 confusion matrix、unknown 原因分布、precision/recall、校准结果；不要在没有人工标签的情况下凭空设定“verifier accuracy”。
- 最终数学结果使用官方 LightEval task 或官方/公开 grader，内部 verifier 结果作为 `auxiliary training telemetry` 单独报告。
- 官方 grader 与内部 verifier 的 disagreement 样本必须保留，作为面试时解释 verifier engineering 的证据。

**验收**：正式报告同时包含 `official_benchmark` 和 `internal_verifier` 两类指标；任何一类 unknown 都不能静默转成 fail 或 pass。

### P1-5/P1-6：训练前 Base、训练后 candidate，并自动生成回归报告

**问题**：虽然脚本已有 Base stage 和 IFEval stage，但还没有自动汇总 Base→candidate 的 OOD delta 与回退 gate。

**设计**：

- 训练前固定生成 Base 的 MATH-500、AMC23、IFEval；训练后对 Vanilla、SuRe 分别运行相同协议。
- 增加 `compare_ood`：从 LightEval details/result JSON 提取 IFEval `prompt_level_strict_acc`，按 task 和总体输出 Base/Vanilla/SuRe/delta。
- 将 OOD 结果写进 promotion report：数学主指标提升、IFEval 不超过预设回退阈值、truncation/unknown 不恶化。阈值必须作为配置记录，而不是代码常量。
- 如果 LightEval 结果格式变化，解析器应 fail closed 并提示人工检查，不应输出空 delta。

**验收**：缺 Base IFEval、缺 candidate IFEval、无法解析 strict accuracy 或 benchmark manifest 不一致时，发布报告失败。

### 工程一致性：去除配置漂移和硬编码

**问题**：`sure_k2_24h.yaml` 深度继承 legacy `base.yaml`，残留 qLoRA、LoRA、旧 batch 和 early-stopping 字段；脚本还硬编码 artifact 路径和 checkpoint。

**设计**：

- 新建只包含 online K2 字段的 `configs/online_k2_base.yaml`；`sure_k2_24h.yaml` 和 `vanilla_k2_24h.yaml` 只继承它。
- legacy `base.yaml` 继续服务旧 weighted OPD，不作为 online K2 的父配置。
- 训练和评测路径全部从解析后的 config 获取；脚本只保留 stage 名称，不再重复声明 final/base/eval 路径。
- 用命名 stage 替代数字恢复点，或至少删除没有执行体的 `START_STAGE=30/31`；恢复时首先检查目标 artifact 是否与 config hash 匹配。
- 在 manifest 中记录 resolved config，不仅记录用户传入的 YAML 路径。

**验收**：`rg` 不应在 SuRe 配置解析结果中发现 `qlora`、`lora`、legacy `early_stopping`；修改 `paths` 后脚本和 gate 自动使用新路径。

### P2：监控与可解释性增强

建议在不改变主目标函数的前提下增加 telemetry：

- token-level `|log p_teacher - log p_student|`、student sampled-token probability、SuRe weight 的 P50/P90/P99 和最大值；
- response position 分桶的 loss/gap/truncation，验证 position-bias 风险；
- actor entropy、参数更新范数、CUDA peak memory、非有限值计数；
- 每步 loss、gradient norm、valid token count、stop reason 和 checkpoint hash。

这些指标用于诊断，不应在看到某个漂亮曲线后事后选择 checkpoint。

## 4. 推荐的低成本正式实验矩阵

### Phase A：一次性准备

1. 固定配置、代码 commit、dataset/model/benchmark revision。
2. 生成 train-clean 和 contamination manifest。
3. 运行真实 RTX PRO 6000 smoke。
4. 运行 2-step、512-prompt pilot，确认吞吐、显存、停止 token 和 checkpoint resume。
5. 训练前运行 Base MATH-500、AMC23、IFEval，并保存命令、details、manifest。

### Phase B：主训练

1. 先跑 Vanilla K2，`alpha=0`，55 steps 或预算安全停止。
2. 再跑 SuRe K2，`alpha=1.0`，其余配置完全一致。
3. 两个运行都保留 step 0、预设 milestone、最终或预算停止 checkpoint。
4. 若 pilot 表明两次完整训练无法满足预算，缩短两者到相同的 34 steps；不得只缩短一个方法。

### Phase C：评测与结论

1. 对 Base、Vanilla、SuRe 使用同一评测命令和同一输入 manifest。
2. 数学：MATH-500、AMC23；低成本模式至少固定 `max_samples` 和 `num_samples`，正式报告明确标注 MVP。
3. 回归：IFEval prompt-level strict accuracy。
4. 自动生成严格 paired comparison 和 OOD delta。
5. 只有在 `SuRe - Vanilla` 的方向、置信区间和 verifier/截断诊断一致时，才把 SuRe 写成项目创新结果；否则诚实地写成“完成了可复现的 OPD/SuRe 工程实现并分析了失败模式”。

## 5. 不建议当前立即做的修改

- 不建议仅因为近期论文使用 forward-KL fallback 就直接把当前 K2 改成 forward KL；这会同时改变目标方向和实验归因。应先完成 Vanilla K2 对照，再把 trust-region/forward-KL 作为独立 P2 消融。
- 不建议把 `pass=1, unknown=0.3, fail=0` 直接解释为最优权重。它只是 verifier-aware heuristic；应通过 calibration 和 matched ablation 验证。
- 不建议把单 seed 的 paired bootstrap CI 写成训练稳定性证明。
- 不建议为了模拟论文把生成长度直接改成 31744，除非显存、磁盘和预算已经通过 pilot；低预算项目优先使用可复现的 budget-adapted protocol。

## 6. 结论与发布门槛

当前最可信的主线不是继续增加复杂 loss，而是补齐实验归因和评测闭环：

```text
严格 manifest / 固定 generation
→ Base evaluation
→ matched Vanilla K2
→ matched SuRe K2
→ 严格 paired comparison
→ 官方数学 grader + IFEval regression + verifier calibration
```

完成上述闭环后，项目可以作为后训练岗位简历项目。简历中应写：

> Implemented a reproducible on-policy distillation pipeline for Qwen3-1.7B using a frozen Qwen3-8B teacher and the sampled-token reverse-KL K2 estimator; added detached surprise-aware reweighting, strict artifact/contamination tracking, paired benchmark evaluation, and verifier calibration.

只有在 `Vanilla K2` 和 `SuRe K2` 的 matched comparison 实际显示增益后，才可以补充“SuRe improved …”。

## 7. 参考资料

- SuRe/K2 paper: https://arxiv.org/abs/2608.25643
- TrOPD: https://arxiv.org/abs/2606.01249
- Position bias / IW-OPD: https://arxiv.org/abs/2606.22600
- OPDVR: https://arxiv.org/abs/2608.24696
- MOPD: https://arxiv.org/abs/2605.12652
- LightEval v0.9.2 vLLM backend: https://github.com/huggingface/lighteval/blob/v0.9.2/src/lighteval/models/vllm/vllm_model.py
- vLLM v0.10.1.1 sampling parameters: https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/sampling_params.py
- MATH benchmark / verifier background: https://github.com/hendrycks/math
