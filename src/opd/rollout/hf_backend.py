from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opd.exceptions import DependencyError
from opd.prompts import render_user_prompt
from opd.rollout.base import Generation
from opd.rollout.stop_tokens import generation_stop_token_ids
from opd.tokenizers import tokenizer_fingerprint


@dataclass
class HFRolloutBackend:
    model_name: str
    model_revision: str
    tokenizer_revision: str
    generation_config: dict[str, Any]
    dtype: str = "bfloat16"
    tokenizer_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyError("HF rollout requires `pip install -e .[gpu]`") from exc
        self._torch = torch
        tokenizer_arguments: dict[str, Any] = {"trust_remote_code": False}
        if not Path(self.model_name).exists():
            tokenizer_arguments["revision"] = self.tokenizer_revision
        tokenizer_loader: Any = AutoTokenizer
        self._tokenizer = tokenizer_loader.from_pretrained(
            self.model_name,
            **tokenizer_arguments,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._stop_token_ids = generation_stop_token_ids(self._tokenizer)
        self.tokenizer_fingerprint = tokenizer_fingerprint(self._tokenizer)
        model_dtype = getattr(torch, self.dtype)
        model_arguments: dict[str, Any] = {
            "torch_dtype": model_dtype,
            "device_map": "auto",
            "trust_remote_code": False,
        }
        if not Path(self.model_name).exists():
            model_arguments["revision"] = self.model_revision
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_arguments,
        )
        model.eval()
        self._model = model

    def generate(self, prompts: list[str], *, seed: int) -> list[Generation]:
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        thinking_value = self.generation_config.get("enable_thinking")
        enable_thinking = bool(thinking_value) if thinking_value is not None else None
        rendered = [
            render_user_prompt(
                self._tokenizer,
                prompt,
                enable_thinking=enable_thinking,
            )
            for prompt in prompts
        ]
        encoded = self._tokenizer(rendered, return_tensors="pt", padding=True)
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        temperature = float(self.generation_config.get("temperature", 0.0))
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 512)),
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": list(self._stop_token_ids),
        }
        if temperature > 0:
            generation_arguments["temperature"] = temperature
            generation_arguments["top_p"] = float(self.generation_config.get("top_p", 0.95))
        with torch.inference_mode():
            outputs = self._model.generate(**encoded, **generation_arguments)
        prompt_width = int(encoded["input_ids"].shape[1])
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        generations: list[Generation] = []
        stop_token_ids = set(self._stop_token_ids)
        max_new_tokens = int(self.generation_config.get("max_new_tokens", 512))
        for output, prompt_tokens in zip(outputs, prompt_lengths, strict=True):
            response_ids = output[prompt_width:]
            response_list = response_ids.tolist()
            terminal_index = next(
                (
                    index
                    for index, token_id in enumerate(response_list)
                    if token_id in stop_token_ids
                ),
                None,
            )
            if terminal_index is not None:
                response_ids = response_ids[: terminal_index + 1]
                finish_reason = "stop"
            else:
                response_ids = response_ids[:max_new_tokens]
                finish_reason = "length" if len(response_ids) >= max_new_tokens else "unknown"
            text = self._tokenizer.decode(
                response_ids,
                skip_special_tokens=bool(self.generation_config.get("skip_special_tokens", False)),
            )
            generations.append(
                Generation(
                    text=text,
                    prompt_tokens=int(prompt_tokens),
                    response_tokens=int(response_ids.numel()),
                    finish_reason=finish_reason,
                )
            )
        return generations
