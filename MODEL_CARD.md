# Model Card Template

## Model

- Base Student: Qwen3-8B-class instruct model, exact revision recorded in the run manifest.
- Teacher: same-tokenizer Qwen3-14B-class instruct/reasoning model.
- Adaptation: 4-bit QLoRA.
- Training seed: 42 only.

## Methods

Report Base, SFT, Vanilla OPD, Verifier-only OPD, Confidence-only OPD and Verifier+Confidence Weighted OPD. Only Vanilla and Weighted are eligible for Round 1.

## Evaluation

Primary benchmarks are MATH-500 and AIME; GPQA Diamond and IFEval measure out-of-domain and instruction-following regression. Store per-item outputs and compare matched items with paired bootstrap and exact McNemar tests.

## Required Disclosure

This project uses one training seed to limit GPU cost. Confidence intervals over benchmark items do not estimate training-seed variance. Fill this card only with measured results; do not infer unrun benchmark scores or claim cross-seed stability.

## Intended Use and Risks

The model is a research artifact for studying budget-aware distillation, not a production mathematical authority. It may produce plausible but incorrect reasoning, inherit training-data contamination and regress outside mathematics.
