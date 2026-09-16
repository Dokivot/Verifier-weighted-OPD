from __future__ import annotations

import math
from dataclasses import dataclass

from opd.tokenizers import mock_tokenizer_fingerprint


@dataclass(frozen=True)
class SimpleTokenizer:
    vocab_size: int = 256
    revision: str = "mock-tokenizer-v1"

    def encode(self, text: str) -> list[int]:
        return [2 + (sum(token.encode("utf-8")) % (self.vocab_size - 2)) for token in text.split()]


@dataclass
class MockTeacherAnnotator:
    model_name: str = "mock-teacher"
    model_revision: str = "mock-teacher-v1"
    top_k: int = 4

    def __post_init__(self) -> None:
        self.tokenizer = SimpleTokenizer()
        self.tokenizer_fingerprint = mock_tokenizer_fingerprint(
            vocab_size=self.tokenizer.vocab_size,
            revision=self.tokenizer.revision,
        )

    def annotate(self, prompt: str, response: str) -> dict[str, object]:
        prompt_ids = self.tokenizer.encode(prompt) or [1]
        response_ids = self.tokenizer.encode(response) or [1]
        topk_ids: list[list[int]] = []
        topk_logprobs: list[list[float]] = []
        tail_mass: list[float] = []
        entropy: list[float] = []
        for token_id in response_ids:
            ids = [token_id]
            for offset in range(1, self.top_k):
                ids.append((token_id + offset) % self.tokenizer.vocab_size)
            probabilities = [0.72] + [0.18 / max(1, self.top_k - 1)] * (self.top_k - 1)
            tail = 0.10
            topk_ids.append(ids)
            topk_logprobs.append([math.log(probability) for probability in probabilities])
            tail_mass.append(tail)
            tail_bucket_count = self.tokenizer.vocab_size - self.top_k
            token_entropy = -sum(
                probability * math.log(probability) for probability in probabilities
            )
            token_entropy -= tail * math.log(tail / tail_bucket_count)
            entropy.append(token_entropy)
        return {
            "input_ids": [*prompt_ids, *response_ids],
            "response_start": len(prompt_ids),
            "response_token_ids": response_ids,
            "topk_token_ids": topk_ids,
            "topk_logprobs": topk_logprobs,
            "tail_mass": tail_mass,
            "token_entropy": entropy,
            "tokenizer_revision": self.tokenizer.revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }
