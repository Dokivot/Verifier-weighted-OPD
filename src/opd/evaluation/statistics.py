from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    samples: int


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired inputs must have the same non-zero length")
    differences = [right - left for left, right in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, int(alpha * iterations))
    upper_index = min(iterations - 1, int((1.0 - alpha) * iterations) - 1)
    return BootstrapResult(
        mean_difference=sum(differences) / len(differences),
        ci_low=means[lower_index],
        ci_high=means[upper_index],
        samples=len(differences),
    )


def mcnemar_exact(baseline: list[bool], candidate: list[bool]) -> dict[str, float | int]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired inputs must have the same non-zero length")
    baseline_only = 0
    candidate_only = 0
    for left, right in zip(baseline, candidate, strict=True):
        if left and not right:
            baseline_only += 1
        elif right and not left:
            candidate_only += 1
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(baseline_only, candidate_only)
        probability = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2.0 * probability)
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "p_value": p_value,
    }
