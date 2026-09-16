# Data Card

## Training Source

- Dataset: `open-r1/OpenR1-Math-220k`, `default` subset.
- Usage: question as rollout prompt; verified solution as SFT target; answer as verifier reference.
- Planned split: 15,000 train, 1,000 validation, 128 smoke, selected deterministically with seed 42 after deduplication and contamination quarantine.
- No manually synthesized questions are required. Student rollouts and Teacher token distributions are generated artifacts.

## Lineage

Every curated record stores source dataset, source revision, source label, normalized sample hash, split, subject and difficulty. Derived artifacts include config hashes, SHA-256 checksums and upstream artifact IDs.

MATH-500/AIME snapshots are fetched at pinned revisions with `opd data fetch-eval`; `opd data import-eval` remains an offline fallback. The contamination report verifies their manifests before filtering. Missing benchmark artifacts are a hard error rather than an implicit no-op.

## Filtering

- Remove rows without problem or reference answer.
- Exact deduplicate normalized problems.
- Compare train prompts against frozen evaluation prompts using normalized exact match and token-shingle Jaccard similarity.
- Quarantine suspected overlaps instead of silently deleting them.
- Keep verifier `unknown` separate from `fail`.

## Limitations

- Public reasoning datasets may contain benchmark-derived questions or generated solutions with latent errors.
- Near-duplicate detection is lexical and can miss paraphrases.
- Licenses and dataset revisions must be reviewed and pinned before publishing trained weights.
- Actual source counts, filter rates and overlap findings are generated in `data_card.json` and contamination reports; this template must be updated with real run values.
