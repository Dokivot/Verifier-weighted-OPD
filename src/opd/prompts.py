from __future__ import annotations

from typing import Any


def render_user_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool | None = None,
    thinking_marker: str | None = None,
) -> str:
    if thinking_marker:
        prompt = f"{prompt.rstrip()}\n{thinking_marker}"
    if getattr(tokenizer, "chat_template", None):
        template_arguments: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_arguments["enable_thinking"] = enable_thinking
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **template_arguments,
            )
        )
    return prompt
