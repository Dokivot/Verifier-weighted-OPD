# Model Card Template

## Model

- Base Student: Qwen2.5-1.5B-Instruct, exact revision recorded in the run manifest.
- Teacher: Qwen2.5-Math-7B-Instruct, exact revision recorded in the run manifest.
- Adaptation: 4-bit QLoRA.
- Training seed: 42 only (recommended scheme one).

## Methods

The first resume MVP reports Base, SFT and VFS-Weighted OPD B50. The main method combines
Verifier-First state acquisition, sample-level verifier weighting and token-level Teacher-entropy
weighting. Later phases add Random-Budget B50, Dense OPD B100 and component ablations. Report the
exact selected and actual Teacher tokens for every method.

## Evaluation

The resume MVP benchmarks MATH-500, AIME 2024 and IFEval. After Random B50 is complete, the primary
research comparison is VFS-Weighted B50 versus Random B50 under no greater actual Teacher tokens.
Store per-item outputs and compare matched items with paired bootstrap and exact McNemar tests.

## Required Disclosure

This project uses one training seed to limit GPU cost. Confidence intervals over benchmark items do not estimate training-seed variance. Fill this card only with measured results; do not infer unrun benchmark scores or claim cross-seed stability.

## Intended Use and Risks

The model is a research artifact for studying budget-aware distillation, not a production mathematical authority. It may produce plausible but incorrect reasoning, inherit training-data contamination and regress outside mathematics.
