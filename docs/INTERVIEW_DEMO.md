# Five-Minute Interview Demo

1. Explain the question: can verifier-first state acquisition improve a 1.5B Student under a fixed 7B Teacher-token budget?
2. Show the three project contributions: state acquisition, dual-granularity weighting and the reproducible cost-aware system.
3. Show `configs/vfs_weighted_mvp.yaml`, the fixed revisions, 3,000 prompts and single seed 42.
4. Open the rollout, verification, selection, annotation and training-view manifests to trace one state end to end.
5. Explain causal alignment: logits at `response_start - 1` predict the first response token.
6. Explain sparse KL and the sample-level verifier plus token-level entropy weights; do not claim independent gains before ablation.
7. Show shard resume, exact Teacher tokens, GPU hours, benchmark details and the Base/SFT/VFS-Weighted result table.
8. State that item bootstrap is not training-seed variance, then describe Random-B50 and Dense-B100 as the next evidence phase.
