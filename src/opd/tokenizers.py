from __future__ import annotations

from typing import Any

from opd.hashing import stable_hash


def tokenizer_fingerprint(tokenizer: Any) -> str:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Tokenizer returned an empty or invalid vocabulary")
    normalized = sorted((str(token), int(token_id)) for token, token_id in vocabulary.items())
    return stable_hash(normalized, length=24)


def mock_tokenizer_fingerprint(*, vocab_size: int, revision: str) -> str:
    return stable_hash(
        {"type": "mock-whitespace-tokenizer", "vocab_size": vocab_size, "revision": revision},
        length=24,
    )
