from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from opd.schemas import VerificationRecord, VerificationStatus

_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_FINAL_PATTERNS = [
    re.compile(r"(?:final answer|answer is|答案是)\s*[:：]?\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
]


def extract_answer(response: str) -> str | None:
    boxed = _BOXED.findall(response)
    if boxed:
        return str(boxed[-1]).strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(response)
        if matches:
            return str(matches[-1]).strip().rstrip(".。")
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) == 1 and len(lines[0]) <= 80:
        return lines[0].rstrip(".。")
    return None


def _normalize_answer(value: str) -> str:
    result = value.strip()
    result = result.replace("$", "").replace("\\,", "")
    result = result.replace("\\left", "").replace("\\right", "")
    result = result.replace(" ", "").lower()
    if result.startswith("\\boxed{") and result.endswith("}"):
        result = result[7:-1]
    return result


def _as_number(value: str) -> float | None:
    normalized = _normalize_answer(value)
    normalized = normalized.replace("\\frac", "frac")
    fraction_match = re.fullmatch(r"frac\{?(-?\d+)\}?\{?(-?\d+)\}?", normalized)
    if fraction_match:
        fraction_denominator = int(fraction_match.group(2))
        if fraction_denominator == 0:
            return None
        return float(Fraction(int(fraction_match.group(1)), fraction_denominator))
    if re.fullmatch(r"-?\d+/\d+", normalized):
        numerator, denominator = normalized.split("/", maxsplit=1)
        if int(denominator) == 0:
            return None
        return float(Fraction(int(numerator), int(denominator)))
    try:
        return float(normalized)
    except ValueError:
        return None


def answers_equivalent(reference: str, candidate: str) -> bool:
    reference_number = _as_number(reference)
    candidate_number = _as_number(candidate)
    if reference_number is not None and candidate_number is not None:
        return math.isclose(reference_number, candidate_number, rel_tol=1e-9, abs_tol=1e-9)
    try:
        from math_verify import parse, verify

        return bool(verify(parse(reference), parse(candidate)))
    except Exception:
        return _normalize_answer(reference) == _normalize_answer(candidate)


@dataclass(frozen=True)
class MathVerifier:
    name: str = "math_verify_with_fallback"
    version: str = "1"

    def verify(
        self,
        *,
        rollout_id: str,
        sample_id: str,
        response: str,
        reference_answer: str,
    ) -> VerificationRecord:
        extracted = extract_answer(response)
        if extracted is None:
            return VerificationRecord(
                rollout_id=rollout_id,
                sample_id=sample_id,
                verifier_name=self.name,
                verifier_version=self.version,
                status=VerificationStatus.UNKNOWN,
                score=0.0,
                reference_answer=reference_answer,
                extracted_answer=None,
                error_type="answer_not_found",
                details={
                    "comparison": "not_run",
                    "reason": "answer_not_found",
                    "message": "",
                },
            )
        try:
            equivalent = answers_equivalent(reference_answer, extracted)
        except Exception as exc:
            return VerificationRecord(
                rollout_id=rollout_id,
                sample_id=sample_id,
                verifier_name=self.name,
                verifier_version=self.version,
                status=VerificationStatus.UNKNOWN,
                score=0.0,
                reference_answer=reference_answer,
                extracted_answer=extracted,
                error_type="verifier_error",
                details={
                    "comparison": "numeric_or_math_verify",
                    "reason": "verifier_error",
                    "message": str(exc),
                },
            )
        return VerificationRecord(
            rollout_id=rollout_id,
            sample_id=sample_id,
            verifier_name=self.name,
            verifier_version=self.version,
            status=VerificationStatus.PASS if equivalent else VerificationStatus.FAIL,
            score=1.0 if equivalent else 0.0,
            reference_answer=reference_answer,
            extracted_answer=extracted,
            error_type=None if equivalent else "wrong_answer",
            details={
                "comparison": "numeric_or_math_verify",
                "reason": "equivalent" if equivalent else "not_equivalent",
                "message": "",
            },
        )
