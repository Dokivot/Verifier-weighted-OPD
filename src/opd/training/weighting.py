from __future__ import annotations

import math
from typing import cast

import numpy as np

from opd.schemas import VerificationStatus


def verifier_weight(status: VerificationStatus | str, weights: dict[str, float]) -> float:
    key = status.value if isinstance(status, VerificationStatus) else status
    if key not in weights:
        raise KeyError(f"Missing verifier weight for status: {key}")
    return float(weights[key])


def confidence_weights(
    entropy: list[float],
    *,
    vocab_size: int,
    minimum: float = 0.1,
    maximum: float = 1.0,
    normalize: bool = True,
) -> list[float]:
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if not entropy:
        return []
    values = np.asarray(entropy, dtype=np.float64)
    confidence = 1.0 - values / math.log(vocab_size)
    confidence = np.clip(confidence, minimum, maximum)
    if normalize:
        mean = float(confidence.mean())
        if mean > 0:
            confidence = confidence / mean
    return cast(list[float], confidence.tolist())
