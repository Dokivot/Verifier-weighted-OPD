from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opd.artifacts import build_manifest, save_manifest
from opd.tableio import write_json
from opd.training.losses import log_softmax_numpy, sparse_kl_numpy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _cosine(first: FloatArray, second: FloatArray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(first, second) / denominator)


def _full_kl(teacher: FloatArray, student_logits: FloatArray) -> float:
    student_logprobs = log_softmax_numpy(student_logits[None, :])[0]
    teacher_logprobs = np.log(np.clip(teacher, 1e-12, 1.0))
    return float(np.sum(teacher * (teacher_logprobs - student_logprobs)))


def _sparse_gradient(student: FloatArray, teacher: FloatArray, top_ids: IntArray) -> FloatArray:
    top_mask = np.zeros_like(student, dtype=bool)
    top_mask[top_ids] = True
    teacher_tail = float(teacher[~top_mask].sum())
    student_tail = max(float(student[~top_mask].sum()), 1e-12)
    gradient = np.empty_like(student)
    gradient[top_mask] = student[top_mask] - teacher[top_mask]
    gradient[~top_mask] = student[~top_mask] * (1.0 - teacher_tail / student_tail)
    return gradient


def audit_sparse_kl(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    audit = config.get("audit", {}).get("sparse_kl", {})
    samples = int(audit.get("samples", 32))
    vocab_size = int(audit.get("vocab_size", 64))
    top_k = int(audit.get("top_k", 8))
    learning_rate = float(audit.get("learning_rate", 0.05))
    teacher_scale = float(audit.get("teacher_logit_scale", 2.5))
    student_noise = float(audit.get("student_noise", 1.0))
    if samples < 1 or vocab_size < 2 or not 1 <= top_k < vocab_size:
        raise ValueError("Sparse KL audit requires samples >= 1 and 1 <= top_k < vocab_size")

    rng = np.random.default_rng(int(config["project"]["seed"]))
    rows: list[dict[str, float | bool]] = []
    for _ in range(samples):
        teacher_logits = rng.normal(scale=teacher_scale, size=vocab_size)
        teacher = np.exp(log_softmax_numpy(teacher_logits[None, :]))[0]
        student_logits = teacher_logits + rng.normal(scale=student_noise, size=vocab_size)
        student = np.exp(log_softmax_numpy(student_logits[None, :]))[0]
        top_ids = np.argsort(teacher)[-top_k:][::-1]
        top_logprobs = np.log(teacher[top_ids])[None, :]
        tail_mass = np.asarray([teacher.sum() - teacher[top_ids].sum()])
        full_loss = _full_kl(teacher, student_logits)
        sparse_loss = sparse_kl_numpy(
            student_logits[None, :],
            top_ids[None, :],
            top_logprobs,
            tail_mass,
        )
        full_gradient = student - teacher
        sparse_gradient = _sparse_gradient(student, teacher, top_ids)
        full_update = student_logits - learning_rate * full_gradient
        sparse_update = student_logits - learning_rate * sparse_gradient
        full_after_full_step = _full_kl(teacher, full_update)
        full_after_sparse_step = _full_kl(teacher, sparse_update)
        rows.append(
            {
                "full_loss": full_loss,
                "sparse_loss": sparse_loss,
                "relative_loss_error": abs(sparse_loss - full_loss) / max(full_loss, 1e-12),
                "topk_mass": float(teacher[top_ids].sum()),
                "gradient_cosine": _cosine(full_gradient, sparse_gradient),
                "update_cosine": _cosine(
                    full_update - student_logits, sparse_update - student_logits
                ),
                "full_after_full_step": full_after_full_step,
                "full_after_sparse_step": full_after_sparse_step,
                "sparse_step_improves_full_kl": full_after_sparse_step < full_loss,
            }
        )

    report: dict[str, Any] = {
        "samples": samples,
        "vocab_size": vocab_size,
        "top_k": top_k,
        "seed": int(config["project"]["seed"]),
        "mean_topk_mass": float(np.mean([float(row["topk_mass"]) for row in rows])),
        "mean_relative_loss_error": float(
            np.mean([float(row["relative_loss_error"]) for row in rows])
        ),
        "mean_gradient_cosine": float(np.mean([float(row["gradient_cosine"]) for row in rows])),
        "mean_update_cosine": float(np.mean([float(row["update_cosine"]) for row in rows])),
        "sparse_step_improvement_rate": float(
            np.mean([bool(row["sparse_step_improves_full_kl"]) for row in rows])
        ),
        "rows": rows,
        "interpretation": (
            "This deterministic synthetic audit compares grouped top-k sparse KL with full KL. "
            "It validates approximation behavior, not downstream model quality."
        ),
    }
    destination = Path(output_path)
    write_json(destination, report)
    manifest = build_manifest(
        artifact_type="sparse_kl_audit",
        stage="audit.sparse_kl",
        config=config,
        files=[destination],
        record_count=samples,
        success_count=samples,
        metadata={
            "top_k": top_k,
            "vocab_size": vocab_size,
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    save_manifest(destination.with_suffix(".manifest.json"), manifest)
    return {key: value for key, value in report.items() if key != "rows"} | {
        "output_path": str(destination)
    }
