from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_LATEX_SPACING = re.compile(r"\\(?:,|;|!|quad\b|qquad\b)")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    normalized = _LATEX_SPACING.sub(" ", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized


def token_shingles(value: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = normalize_text(value).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_similarity(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
