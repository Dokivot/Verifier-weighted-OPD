from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    response_tokens: int


class RolloutBackend(Protocol):
    model_name: str
    model_revision: str
    tokenizer_revision: str

    def generate(self, prompts: list[str], *, seed: int) -> list[Generation]: ...
