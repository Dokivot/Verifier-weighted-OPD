from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def log_softmax_numpy(logits: FloatArray, axis: int = -1) -> FloatArray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return np.asarray(
        shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True)),
        dtype=np.float64,
    )


def sparse_kl_numpy(
    student_logits: FloatArray,
    teacher_topk_ids: IntArray,
    teacher_topk_logprobs: FloatArray,
    teacher_tail_mass: FloatArray,
    weights: FloatArray | None = None,
) -> float:
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [tokens, vocab]")
    token_count = student_logits.shape[0]
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError("teacher top-k ids and logprobs must have identical shapes")
    if teacher_topk_ids.shape[0] != token_count or teacher_tail_mass.shape[0] != token_count:
        raise ValueError("teacher fields must match student token count")
    student_logprobs = log_softmax_numpy(student_logits)
    rows = np.arange(token_count)[:, None]
    selected_student = student_logprobs[rows, teacher_topk_ids]
    teacher_probs = np.exp(teacher_topk_logprobs)
    top_kl = (teacher_probs * (teacher_topk_logprobs - selected_student)).sum(axis=-1)
    student_top_mass = np.exp(selected_student).sum(axis=-1)
    student_tail_mass = np.clip(1.0 - student_top_mass, 1e-12, 1.0)
    teacher_tail = np.clip(teacher_tail_mass, 1e-12, 1.0)
    tail_kl = teacher_tail * (np.log(teacher_tail) - np.log(student_tail_mass))
    per_token = top_kl + tail_kl
    if weights is None:
        return float(per_token.mean())
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (token_count,):
        raise ValueError("weights must have shape [tokens]")
    denominator = float(weights.sum())
    if denominator <= 0:
        return 0.0
    return float((per_token * weights).sum() / denominator)


def sparse_kl_torch(
    student_logits: Any,
    teacher_topk_ids: Any,
    teacher_topk_logprobs: Any,
    teacher_tail_mass: Any,
    weights: Any,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch loss requires the GPU optional dependencies") from exc
    student_logprobs = torch.log_softmax(student_logits.float(), dim=-1)
    selected_student = torch.gather(student_logprobs, dim=-1, index=teacher_topk_ids.long())
    teacher_probs = teacher_topk_logprobs.float().exp()
    top_kl = (teacher_probs * (teacher_topk_logprobs.float() - selected_student)).sum(dim=-1)
    student_top_mass = selected_student.exp().sum(dim=-1)
    student_tail_mass = (1.0 - student_top_mass).clamp(min=1e-12, max=1.0)
    teacher_tail = teacher_tail_mass.float().clamp(min=1e-12, max=1.0)
    tail_kl = teacher_tail * (teacher_tail.log() - student_tail_mass.log())
    per_token = top_kl + tail_kl
    effective_weights = weights.float()
    denominator = effective_weights.sum().clamp(min=1e-12)
    return (per_token * effective_weights).sum() / denominator
