from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opd.exceptions import DependencyError
from opd.prompts import render_user_prompt
from opd.rollout.base import Generation
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
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **tokenizer_arguments,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self.tokenizer_fingerprint = tokenizer_fingerprint(self._tokenizer)
        model_dtype = getattr(torch, self.dtype)
        model_arguments: dict[str, Any] = {
            "torch_dtype": model_dtype,
            "device_map": "auto",
            "trust_remote_code": False,
        }
        if not Path(self.model_name).exists():
            model_arguments["revision"] = self.model_revision
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_arguments,
        )
        self._model.eval()

    def generate(self, prompts: list[str], *, seed: int) -> list[Generation]:
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rendered = [render_user_prompt(self._tokenizer, prompt) for prompt in prompts]
        encoded = self._tokenizer(rendered, return_tensors="pt", padding=True)
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        temperature = float(self.generation_config.get("temperature", 0.0))
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 512)),
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_arguments["temperature"] = temperature
            generation_arguments["top_p"] = float(self.generation_config.get("top_p", 0.95))
        with torch.inference_mode():
            outputs = self._model.generate(**encoded, **generation_arguments)
        prompt_width = int(encoded["input_ids"].shape[1])
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        generations: list[Generation] = []
        for output, prompt_tokens in zip(outputs, prompt_lengths, strict=True):
            response_ids = output[prompt_width:]
            text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            generations.append(
                Generation(
                    text=text,
                    prompt_tokens=int(prompt_tokens),
                    response_tokens=int(response_ids.numel()),
                )
            )
        return generations
