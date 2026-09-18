# SuRe K2 一键运行与恢复指南

`scripts/run_sure_k2_oneclick.sh` 是正式实验的总控入口。它按阶段调用现有脚本，并为每个阶段保存成功标记。

## 一次启动

```bash
cd /root/autodl-tmp/OPDProj
git fetch origin
git checkout rtx-pro-6000-blackwell
git pull --ff-only origin rtx-pro-6000-blackwell
source scripts/autodl_env.sh
tmux new -s sure-k2
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail
scripts/run_sure_k2_oneclick.sh 2>&1 | tee logs/sure_k2_oneclick_terminal.log
```

默认配置为 `configs/sure_k2_24h.yaml`，smoke 和 pilot 配置为 `configs/sure_k2_smoke.yaml`、
`configs/sure_k2_pilot.yaml`。总控脚本依次完成依赖安装、CPU 检查、数据准备、评测集下载、去污染、真实模型
smoke、pilot、Base 评测、readiness gate、55-step 训练、Candidate 评测、官方 benchmark、回归报告和最终报告。

## 成功标记和阶段

成功阶段的标记在 `logs/sure_k2_oneclick/state/stage_XXX.ok`，包含配置 hash、代码 commit、日志路径和完成时间。
下一次启动时，配置 hash 相同且标记存在的阶段会自动跳过；失败阶段没有成功标记，会重新执行。

| 阶段 | 内容 |
| --- | --- |
| 0–1 | 依赖安装、CPU 检查 |
| 10–13 | 数据准备、MATH-500/AMC23 下载、去污染 |
| 18–19 | 真实模型 smoke、2-step pilot |
| 14–17 | Base 内部评测、Base IFEval、Base 官方 MATH-500 |
| 20 | SuRe K2 readiness gate |
| 30 | 55-step 全参数训练 |
| 40–43 | Candidate 内部评测、IFEval、官方 MATH-500 |
| 50–51 | 对比报告、OOD 回归、最终报告、预算汇总 |

先运行 smoke/pilot 再运行 Base 评测，是为了先发现显存、tokenizer 和恢复问题；不影响最终报告中的
Base-before-Candidate 逻辑。

## 失败信息和恢复

失败时脚本返回非零退出码，并生成阶段日志和失败包：

```text
logs/sure_k2_oneclick/<run_id>/<stage>_<name>.log
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/summary.env
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/log_tail.txt
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/nvidia_smi.txt
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/disk.txt
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/memory.txt
logs/sure_k2_oneclick/<run_id>/failure_<stage>_<name>/processes.txt
logs/sure_k2_oneclick/state/latest_failure/
```

失败包包含退出码、最近 240 行日志、代码版本、git 状态、GPU、磁盘、内存和相关进程信息。向别人求助时，
优先提供 `latest_failure` 和对应阶段日志，不要上传 Hugging Face token。

修复代码后重新拉取并直接重跑同一命令：

```bash
git fetch origin
git pull --ff-only origin rtx-pro-6000-blackwell
source scripts/autodl_env.sh
scripts/run_sure_k2_oneclick.sh 2>&1 | tee -a logs/sure_k2_oneclick_terminal.log
```

代码 commit 更新后，stage 0–1 会重新确认环境；其余阶段只要配置 hash 不变且有成功标记，就会继续复用。失败
阶段会重跑，成功阶段不会从头开始。若正式训练 stage 30 已写出 `rolling` checkpoint，总控脚本会自动把它
传给训练器，继续已完成的 step，而不是从 step 1 开始。

## 强制重跑和指定起点

若修复改变了某阶段的产物语义，可从该阶段强制重跑：

```bash
scripts/run_sure_k2_oneclick.sh --force-from 14 \
  2>&1 | tee -a logs/sure_k2_oneclick_terminal.log
```

它不会自动删除旧 artifact。若底层工具因半成品拒绝运行，先保存失败包，再只删除该失败阶段明确指出的临时
输出；不要删除 `rolling`、manifest 或整个实验目录。

如果 stage 10–13 已成功，希望从 Base 评测继续：

```bash
scripts/run_sure_k2_oneclick.sh --from-stage 14 \
  2>&1 | tee -a logs/sure_k2_oneclick_terminal.log
```

只有 stage 10–17 均已成功，并且 smoke、pilot 仍对应当前配置时，才可从 stage 20 继续。默认恢复机制通常更安全。

## 监控、暂停和求职材料

```bash
watch -n 5 nvidia-smi
watch -n 30 'find artifacts/sure_k2_24h/checkpoints/sure_k2_seed42/telemetry -name "step_*.json" | sort | tail -3'
```

确认最新 telemetry 和 rolling checkpoint 已写完后，才在训练终端按 `Ctrl+C`。重新启动总控脚本即可恢复；若
中断发生在 checkpoint 写入过程中，训练器会拒绝不完整的 rolling 状态并保留失败包，不会静默猜测。

完成后至少保留 `training_summary.json`、训练 telemetry、`benchmark/base`、`benchmark/sure_k2`、`evaluation`、
`reports/sure_k2_24h.json`、两个 Base/Candidate comparison、`ood_regression.json`、readiness report、
`logs/sure_k2_oneclick/`、配置和 `uv.lock`。这些文件支持面试中的方法说明、训练曲线、官方 benchmark、OOD
回归、成本和可复现性说明；内部 verifier 指标必须和官方 benchmark 分开表述。
