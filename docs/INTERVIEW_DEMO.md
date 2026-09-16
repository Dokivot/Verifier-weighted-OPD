# Five-Minute Interview Demo

1. Explain the question: can verifier filtering and Teacher-confidence weighting improve an 8B Student under the same token budget?
2. Show `PROJECT_PLAN.md` and the fixed experiment matrix with seed 42.
3. Run `make smoke`, then open the rollout, verification, annotation and training-view manifests.
4. Explain causal alignment: logits at `response_start - 1` predict the first response token.
5. Explain sparse KL: exact top-k Teacher terms plus one collapsed tail bucket, then show the 32-case loss/gradient/update audit.
6. Show shard resume by rerunning rollout and reading `reused_shards` from the manifest.
7. Show paired comparison output and state that its interval is item-level, not training-seed variance.
8. Show checkpoint promotion and cumulative GPU/Teacher-token accounting, then close with the quality-cost table, negative cases and the 110 H100h hard cap.
