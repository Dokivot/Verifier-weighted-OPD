# SuRe K2 warm-start fast-throughput runbook

This runbook tests and then uses the `4/2/2` micro-batch configuration on top
of the completed 512-prompt Pilot checkpoint. It keeps the global batch,
training objective, data, model revisions, seed, and response-token cap from
the existing warm-start plan unchanged. The existing `2/1/1` warm-start
configuration and artifacts are not overwritten.

## What changes

The fast configuration is
`configs/sure_k2_warmstart_fast_24h.yaml`:

```yaml
global_prompt_batch_size: 128
rollout_micro_batch_size: 4
teacher_micro_batch_size: 2
student_micro_batch_size: 2
max_steps: 14
max_response_tokens: 8192
```

The file initially contains `training.invocation_step_limit: 1`. This is an
operational limit only: it is ignored by the canonical config hash and the
online K2 run ID. It is removed after the two-step throughput check.

## Preconditions

Run from the repository root on the GPU server:

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
test -f artifacts/sure_k2_24h/pilot/checkpoint/final/config.json
test -f artifacts/sure_k2_warmstart_24h/data/contamination/manifest.json
test -f artifacts/sure_k2_warmstart_24h/data/contamination/train_clean.parquet
```

The initial checkpoint is the finished 512-prompt Pilot. The training data is
the already prepared and decontaminated warm-start data. Do not delete either
artifact.

## Step 1: run one fast step

Use `tmux` so a disconnected SSH session does not stop training:

```bash
tmux new -s opd-fast
set -o pipefail
scripts/train.sh configs/sure_k2_warmstart_fast_24h.yaml \
  2>&1 | tee logs/sure_k2_warmstart_fast_step1.log
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t opd-fast
```

The command should finish after exactly one optimizer step and create:

```text
artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42/
├── rolling/model/
├── rolling/training_state.pt
├── telemetry/step_000001.json
└── telemetry/step_000001_quality_gate.json
```

Check for OOM, NaN/Inf, and the step timing:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42")
for step in (1,):
    metrics = json.loads((root / f"telemetry/step_{step:06d}.json").read_text())
    quality = json.loads((root / f"telemetry/step_{step:06d}_quality_gate.json").read_text())
    print(metrics)
    print(quality)
PY
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

## Step 2: run one resumed fast step

Keep `invocation_step_limit: 1` in the config and resume from the atomic
rolling checkpoint:

```bash
set -o pipefail
scripts/train.sh \
  configs/sure_k2_warmstart_fast_24h.yaml \
  artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42/rolling \
  2>&1 | tee logs/sure_k2_warmstart_fast_step2.log
```

This must produce `telemetry/step_000002.json` and leave the same rolling
checkpoint usable for continuation.

Compare the two steps:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42")
for path in sorted((root / "telemetry").glob("step_*.json")):
    if path.name.endswith("_quality_gate.json"):
        continue
    data = json.loads(path.read_text())
    print(path.name, {
        key: data.get(key)
        for key in (
            "step_seconds", "rollout_seconds", "teacher_seconds",
            "student_seconds", "loss", "gradient_norm", "truncation_rate",
            "sure_weight_min", "sure_weight_max",
        )
    })
PY
```

Adopt `4/2/2` for the rest of the run only if both steps have no OOM or
CUDA error, no NaN/Inf, truncation remains at or below the configured 20%
hard gate, and step time is materially better than the old `2/1/1` Pilot.
The quality gate is a correctness requirement; the speed improvement is the
reason for keeping this candidate configuration.

## Step 3: continue to 14 steps

After Step 2 succeeds, remove only the temporary invocation limit:

```bash
sed -i '/^  invocation_step_limit: 1$/d' \
  configs/sure_k2_warmstart_fast_24h.yaml
```

Confirm it is gone and the micro-batches remain unchanged:

```bash
rg -n 'invocation_step_limit|global_prompt_batch_size|micro_batch_size|max_steps' \
  configs/sure_k2_warmstart_fast_24h.yaml
```

Then continue from the rolling checkpoint:

```bash
set -o pipefail
scripts/train.sh \
  configs/sure_k2_warmstart_fast_24h.yaml \
  artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42/rolling \
  2>&1 | tee logs/sure_k2_warmstart_fast_steps3_14.log
```

The run should finish at global step 14 and write:

```text
artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42/final/
artifacts/sure_k2_warmstart_fast_24h/checkpoints/sure_k2_seed42/training_summary.json
```

Do not use `initial_student_checkpoint` again after Step 1: a resumed run
must use the rolling checkpoint so optimizer, scheduler, RNG state, data
cursor, and policy-hash chain continue consistently.

## If the candidate fails

If either test step OOMs or violates the quality gate, stop and keep the
existing `2/1/1` warm-start plan unchanged. Do not resume the failed fast
checkpoint with a different micro-batch configuration. Instead, remove the
fast artifact directory and run the original warm-start configuration, or
start a new warm-start artifact directory with a safer candidate such as
`3/1/1`.

Changing micro-batch sizes after a run has started is not an exact resume,
even though `invocation_step_limit` and `resume_from_checkpoint` do not change
the canonical run identity. The two-step check is intentionally completed
before accepting the fast configuration for the remaining steps.
