from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from opd.schemas import VerificationRecord, VerificationStatus

_BOXED_START = re.compile(r"\\boxed\s*\{")
_FINAL_PATTERNS = [
    re.compile(
        r"(?:final answer(?:\s+is)?|answer is|答案是)\s*[:：]?\s*([^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(r"####\s*([^\n]+)"),
]


@dataclass(frozen=True)
class _AnswerCandidate:
    text: str
    extraction_mode: str
    confidence: str


def _last_boxed_answer(response: str) -> str | None:
    answers: list[str] = []
    for match in _BOXED_START.finditer(response):
        depth = 1
        cursor = match.end()
        while cursor < len(response) and depth > 0:
            character = response[cursor]
            if character == "{" and (cursor == 0 or response[cursor - 1] != "\\"):
                depth += 1
            elif character == "}" and (cursor == 0 or response[cursor - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    answers.append(response[match.end() : cursor].strip())
                    break
            cursor += 1
    return answers[-1] if answers else None


def _explicit_answer(response: str) -> _AnswerCandidate | None:
    boxed = _last_boxed_answer(response)
    if boxed:
        return _AnswerCandidate(boxed, "strict_boxed", "high")
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(response)
        if matches:
            return _AnswerCandidate(
                str(matches[-1]).strip().rstrip(".。"),
                "strict_marker",
                "high",
            )
    return None


def extract_answer(response: str) -> str | None:
    explicit = _explicit_answer(response)
    if explicit is not None:
        return explicit.text
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

        return bool(verify(parse(reference), parse(candidate), strict=True))
    except Exception:
        return _normalize_answer(reference) == _normalize_answer(candidate)


def _math_verify_extract(text: str, *, allow_unanchored: bool) -> str | None:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    parsed: list[Any] = parse(
        text,
        extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=allow_unanchored),
            ExprExtractionConfig(try_extract_without_anchor=allow_unanchored),
        ],
        fallback_mode="first_match",
        extraction_mode="any_match",
        raise_on_error=False,
    )
    if not parsed:
        return None
    for value in reversed(parsed):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(parsed[0])


def _last_nonempty_line(response: str) -> str:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _extract_candidate(response: str) -> _AnswerCandidate | None:
    explicit = _explicit_answer(response)
    if explicit is not None:
        return explicit

    anchored = _math_verify_extract(response, allow_unanchored=False)
    if anchored is not None:
        return _AnswerCandidate(anchored, "math_verify_anchored", "high")

    answer_tail = _last_nonempty_line(response)
    if not answer_tail:
        return None
    fallback = _math_verify_extract(answer_tail, allow_unanchored=True)
    if fallback is None:
        return None
    return _AnswerCandidate(fallback, "math_verify_tail_fallback", "medium")


def _details(
    *,
    comparison: str,
    reason: str,
    extraction_mode: str,
    extraction_confidence: str,
    parser_status: str,
    message: str = "",
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "reason": reason,
        "message": message,
        "extraction_mode": extraction_mode,
        "extraction_confidence": extraction_confidence,
        "parser_status": parser_status,
    }


@dataclass(frozen=True)
class MathVerifier:
    name: str = "math_verify_with_fallback"
    version: str = "2"

    def verify(
        self,
        *,
        rollout_id: str,
        sample_id: str,
        response: str,
        reference_answer: str,
    ) -> VerificationRecord:
        try:
            candidate = _extract_candidate(response)
        except Exception as exc:
            return VerificationRecord(
                rollout_id=rollout_id,
                sample_id=sample_id,
                verifier_name=self.name,
                verifier_version=self.version,
                status=VerificationStatus.UNKNOWN,
                score=0.0,
                reference_answer=reference_answer,
                extracted_answer=None,
                error_type="verifier_error",
                details=_details(
                    comparison="not_run",
                    reason="parser_error",
                    extraction_mode="none",
                    extraction_confidence="none",
                    parser_status="error",
                    message=str(exc),
                ),
            )
        if candidate is None:
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
                details=_details(
                    comparison="not_run",
                    reason="answer_not_found",
                    extraction_mode="none",
                    extraction_confidence="none",
                    parser_status="no_match",
                ),
            )
        try:
            equivalent = answers_equivalent(reference_answer, candidate.text)
        except Exception as exc:
            return VerificationRecord(
                rollout_id=rollout_id,
                sample_id=sample_id,
                verifier_name=self.name,
                verifier_version=self.version,
                status=VerificationStatus.UNKNOWN,
                score=0.0,
                reference_answer=reference_answer,
                extracted_answer=candidate.text,
                error_type="verifier_error",
                details=_details(
                    comparison="numeric_or_math_verify",
                    reason="verifier_error",
                    extraction_mode=candidate.extraction_mode,
                    extraction_confidence=candidate.confidence,
                    parser_status="matched",
                    message=str(exc),
                ),
            )
        return VerificationRecord(
            rollout_id=rollout_id,
            sample_id=sample_id,
            verifier_name=self.name,
            verifier_version=self.version,
            status=VerificationStatus.PASS if equivalent else VerificationStatus.FAIL,
            score=1.0 if equivalent else 0.0,
            reference_answer=reference_answer,
            extracted_answer=candidate.text,
            error_type=None if equivalent else "wrong_answer",
            details=_details(
                comparison="numeric_or_math_verify",
                reason="equivalent" if equivalent else "not_equivalent",
                extraction_mode=candidate.extraction_mode,
                extraction_confidence=candidate.confidence,
                parser_status="matched",
            ),
        )
