# AutoDL Validation Audit

本文记录“代码已检查到什么程度”，避免把本地 mock 成功表述成真实 GPU 实验成功。

## 当前结论

项目处于**有条件可运行**状态。CPU 工程链路已经验证，模型与数据 revision 已做官方元数据和
源码级静态核验；由于本地没有 NVIDIA GPU，尚不能声称方案一的 Qwen2.5-1.5B/Math-7B 在 AutoDL
已完整跑通。

## 本地已验证

- Python 3.11 下的 unittest、Ruff、mypy strict 与 `uv lock --check`。
- 无模型下载的完整 mock 链路：prepare、污染审计、rollout、verifier、annotation selection、
  Teacher annotation、training view、训练、评测、promotion、失败分析与成本报告。
- 固定 seed、artifact checksum/lineage、分片恢复、训练 batch cursor、Early Stop、Sparse KL、
  sample/token 权重、paired bootstrap 与 McNemar。
- Shell 语法与配置继承；Dense/Random/Filtered/VFS 使用隔离的 selection 与 checkpoint 目录。

## 官方静态核验

- `open-r1/OpenR1-Math-220k` 的固定 revision 与 `default` schema
  `problem/solution/answer`。
- Student、Teacher、MATH-500、AIME 2024、AIME 2025 均已固定 commit revision；AutoDL 首次
  下载后仍需在 manifest 中核对实际 revision。
- LightEval 0.9.2 的 vLLM `model_name=` 参数、chat template、逐题 details 与 IFEval 扩展任务模块。
- AutoDL 推荐 PyTorch 2.8.0 + CUDA 12.8 基础镜像；项目 `.venv` 按锁文件使用 PyTorch 2.7.1
  和 CUDA 12.6 runtime wheels。

## 已修复的阻塞问题

- 源码 checkpoint 包被 `.gitignore` 误排除。
- Docker context 的裸目录规则可能误排除同名源码包。
- shell 在 `uv sync` 后未稳定使用项目虚拟环境。
- Accelerate 多进程竞争写 manifest，以及训练恢复重复已消费 batch。
- 训练循环变量覆盖导致恢复 cursor 写错。
- Round 1 两个方法覆盖同一份 rollout/annotation。
- vLLM 超长默认上下文带来的 KV cache OOM 风险与 rollout 参数缺失。
- Rollout 的 8,192 context 与 4,096 response 上限分离，Teacher/训练显式截到 4,096 总 tokens。
- LightEval 参数名、IFEval 注册、chat template、逐题 details 与 smoke 限样缺失。
- benchmark 依赖手工制作本地文件，现已改为固定 revision 自动下载。
- Teacher annotation 前缺少预算选择，现已增加 deterministic VFS selector、token 硬预算、
  selection manifest 和 Dense annotation 实验缓存复用。
- Teacher annotation 只要存在失败记录就会在保存 manifest 后终止，禁止静默缩小训练集。

## AutoDL 必须实测

按顺序执行，任一步失败都停止扩容：

```bash
source scripts/autodl_env.sh
scripts/remote_bootstrap.sh
make check
make smoke
make tiny-gpu-smoke
scripts/qwen_gpu_smoke.sh
```

必须人工确认 CUDA/BF16、模型下载、tokenizer fingerprint、4-bit QLoRA、7B Math Teacher BF16、
adapter merge、vLLM 重新加载、LightEval results/details、峰值显存和剩余磁盘。Dockerfile 也仅做了
静态审查，尚未在本地构建 CUDA 镜像。

## 正式扩容门禁

只有 `artifacts/qwen_gpu_smoke/` 中所有阶段 manifest/job metrics 完整、无批量失败、loss 与
gradient norm 有限、merged checkpoint 可评测，且 LightEval 两题 smoke 产生 results/details，
才允许启动 6,000 候选、3,000 prompts 的简历 MVP。正式配置使用 8,192 context 和 4,096
response token 上限，并在 800 条成功样本后自动检查截断率；超过 20% 会立即失败，不得继续
Teacher annotation。完整 MVP 推荐单张 H100/A800 80GB，并始终保留至少 15GiB 数据盘空闲。

## 结果解释边界

- 最终 benchmark 以 LightEval 原生输出为准，内部 regression 只用于训练门禁和快速比较。
- 单 seed 结果只能写成“在 seed 42 和当前预算下观察到”，不能声称跨 seed 稳定。
- paired bootstrap 只描述题目抽样不确定性，不描述训练随机性。
- GPQA 可能需要 Hugging Face 授权；未获得授权时应记录为未运行，不得用其他数据替代。
