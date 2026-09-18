from __future__ import annotations

from typing import Any


def generation_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the tokenizer's EOS token plus common Qwen chat terminators."""
    candidates: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        candidates.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        candidates.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unknown_token_id:
                candidates.append(token_id)

    unique = tuple(dict.fromkeys(candidates))
    if not unique:
        raise ValueError("Tokenizer exposes no usable generation stop token")
    return unique
