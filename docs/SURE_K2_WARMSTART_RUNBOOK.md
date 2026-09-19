# SuRe K2 warm-start runbook

This variant reuses the completed two-step 512-prompt pilot as the initial
student model, then runs a separate budgeted experiment with a 128-prompt
batch. It does not resume the pilot optimizer or scheduler.

## Preconditions

The server must contain the completed pilot model at:

```text
artifacts/sure_k2_24h/pilot/checkpoint/final
```

Verify it before starting:

```bash
test -f artifacts/sure_k2_24h/pilot/checkpoint/final/config.json
find artifacts/sure_k2_24h/pilot/checkpoint/final -name '*.safetensors' -print
```

The original `models.student.name` remains the Hugging Face Base model. The
warm-start source is recorded separately as
`training.initial_student_checkpoint`, so Base evaluation is not accidentally
run on the Pilot weights.

## Run

```bash
cd /root/autodl-tmp/OPDProj
source scripts/autodl_env.sh
set -o pipefail

tmux new -s opd-warmstart
scripts/run_sure_k2_warmstart.sh \
  --no-bootstrap \
  2>&1 | tee logs/sure_k2_warmstart_terminal.log
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t opd-warmstart
```

The existing Pilot, rollout, and teacher artifacts are not deleted. The new
run writes under `artifacts/sure_k2_warmstart_24h/` and uses its own stage
markers and reports.

## Important interpretation

The result is a warm-start experiment, not an exact continuation of the
512-prompt run. The optimizer, scheduler, step counter, and output directory
are new. The final report records the Pilot checkpoint used for initialization.

The formal run uses 14 new optimizer steps with a 128-prompt batch. The
generation cap remains 8192 tokens; it is not changed to 4096.
