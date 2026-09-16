from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opd.schemas import PromptRecord, TrainingRecord, VerificationStatus
from opd.tableio import read_records, write_json
from opd.training.losses import sparse_kl_numpy


def train_mock(config: dict[str, Any]) -> Path:
    training = config["training"]
    method = training["method"]
    input_path = Path(training["input_path"])
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed)

    if method == "sft":
        prompts = [PromptRecord.model_validate(row) for row in read_records(input_path)]
        token_count = sum(
            len((record.reference_solution or record.reference_answer).split())
            for record in prompts
        )
        initial_loss = 3.0
        final_loss = 1.0 / max(1.0, np.log1p(token_count))
        validation_score = min(1.0, 0.25 + np.log1p(len(prompts)) / 20)
        record_count = len(prompts)
    else:
        records = [TrainingRecord.model_validate(row) for row in read_records(input_path)]
        losses: list[float] = []
        pass_count = 0
        for record in records:
            token_count = len(record.response_token_ids)
            if token_count == 0:
                continue
            vocab_size = int(config["models"].get("vocab_size", 256))
            logits = rng.normal(size=(token_count, vocab_size))
            weights = np.asarray(record.confidence_weights)
            losses.append(
                sparse_kl_numpy(
                    logits,
                    np.asarray(record.topk_token_ids),
                    np.asarray(record.topk_logprobs),
                    np.asarray(record.tail_mass),
                    weights,
                )
                * record.verifier_weight
            )
            pass_count += record.verifier_status == VerificationStatus.PASS
        initial_loss = float(np.mean(losses)) if losses else 0.0
        final_loss = initial_loss * 0.7
        validation_score = pass_count / max(1, len(records))
        record_count = len(records)

    checkpoint = {
        "backend": "mock",
        "method": method,
        "seed": seed,
        "record_count": record_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "validation_score": validation_score,
        "note": "Smoke-test artifact only; not a model checkpoint.",
    }
    checkpoint_path = output_dir / "checkpoint.json"
    write_json(checkpoint_path, checkpoint)
    write_json(
        output_dir / "training_summary.json",
        {
            "method": method,
            "global_step": int(training.get("max_steps", 1)),
            "best_validation_score": validation_score,
            "best_step": int(training.get("max_steps", 1)),
            "history": [
                {
                    "step": int(training.get("max_steps", 1)),
                    "loss": final_loss,
                    "validation_accuracy": validation_score,
                    "format_pass_rate": 1.0,
                    "instruction_regression_score": 1.0,
                    "composite_score": 0.8 * validation_score + 0.2,
                }
            ],
            "seed": seed,
            "note": "Deterministic CPU smoke metrics; not a model-quality measurement.",
        },
    )
    return checkpoint_path
