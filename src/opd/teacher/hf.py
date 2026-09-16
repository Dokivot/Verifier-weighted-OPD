from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from opd.exceptions import DependencyError
from opd.prompts import render_user_prompt
from opd.tokenizers import tokenizer_fingerprint


@dataclass
class HFTeacherAnnotator:
    model_name: str
    model_revision: str
    tokenizer_revision: str
    top_k: int = 32
    dtype: str = "bfloat16"
    load_in_8bit: bool = False
    max_length: int = 2048
    tokenizer_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise DependencyError("HF annotation requires `pip install -e .[gpu]`") from exc
        self._torch = torch
        tokenizer_loader: Any = AutoTokenizer
        self._tokenizer = tokenizer_loader.from_pretrained(
            self.model_name,
            revision=self.tokenizer_revision,
            trust_remote_code=False,
        )
        dtype = getattr(torch, self.dtype)
        quantization_config_type: Any = BitsAndBytesConfig
        quantization_config = (
            quantization_config_type(load_in_8bit=True) if self.load_in_8bit else None
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=False,
        )
        model.eval()
        self._model = model
        self.tokenizer_fingerprint = tokenizer_fingerprint(self._tokenizer)

    def _encode(self, prompt: str, response: str) -> tuple[list[int], list[int]]:
        rendered_prompt = render_user_prompt(self._tokenizer, prompt)
        prompt_ids = self._tokenizer.encode(rendered_prompt, add_special_tokens=True)
        response_ids = self._tokenizer.encode(response, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("Prompt produced no tokens")
        if not response_ids:
            raise ValueError("Response produced no tokens")
        available_response_tokens = self.max_length - len(prompt_ids)
        if available_response_tokens <= 0:
            raise ValueError(
                f"Prompt length {len(prompt_ids)} leaves no room in teacher max_length "
                f"{self.max_length}"
            )
        response_ids = response_ids[:available_response_tokens]
        return prompt_ids, response_ids

    def annotate(self, prompt: str, response: str) -> dict[str, Any]:
        torch = self._torch
        prompt_ids, response_ids = self._encode(prompt, response)
        input_ids = [*prompt_ids, *response_ids]
        device = next(self._model.parameters()).device
        tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model(input_ids=tensor).logits[0].float()
            start = len(prompt_ids) - 1
            response_logits = logits[start : start + len(response_ids)]
            logprobs = torch.log_softmax(response_logits, dim=-1)
            probs = logprobs.exp()
            entropy = -(probs * logprobs).sum(dim=-1)
            values, indices = torch.topk(logprobs, k=min(self.top_k, logprobs.shape[-1]), dim=-1)
            top_mass = values.exp().sum(dim=-1)
            tail_mass = torch.clamp(1.0 - top_mass, min=0.0, max=1.0)
        return {
            "input_ids": input_ids,
            "response_start": len(prompt_ids),
            "response_token_ids": response_ids,
            "topk_token_ids": indices.cpu().tolist(),
            "topk_logprobs": values.cpu().tolist(),
            "tail_mass": tail_mass.cpu().tolist(),
            "token_entropy": entropy.cpu().tolist(),
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "vocab_size": int(logprobs.shape[-1]),
            "normalized_entropy": (entropy / math.log(logprobs.shape[-1])).cpu().tolist(),
        }
