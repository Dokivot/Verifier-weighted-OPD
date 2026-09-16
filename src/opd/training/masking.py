from __future__ import annotations


def build_response_labels(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    eos_token_id: int,
    max_length: int,
) -> tuple[list[int], list[int]]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    input_ids = [*prompt_ids, *response_ids, eos_token_id][:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return input_ids, labels


def response_logit_bounds(
    *, response_start: int, response_tokens: int, input_length: int
) -> tuple[int, int]:
    if response_start < 1:
        raise ValueError("response_start must be at least one")
    if response_tokens < 0:
        raise ValueError("response_tokens must be non-negative")
    start = response_start - 1
    end = start + response_tokens
    if end > input_length - 1:
        raise ValueError("Response targets extend beyond the causal input")
    return start, end
