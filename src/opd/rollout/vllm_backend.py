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
class VLLMRolloutBackend:
    model_name: str
    model_revision: str
    tokenizer_revision: str
    generation_config: dict[str, Any]
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_length: int = 4096
    gpu_memory_utilization: float = 0.9
    tokenizer_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise DependencyError("vLLM rollout requires `pip install -e .[gpu]`") from exc
        self._sampling_params_type = SamplingParams
        model_arguments: dict[str, Any] = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_length,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": False,
        }
        if not Path(self.model_name).exists():
            model_arguments["revision"] = self.model_revision
        self._llm = LLM(**model_arguments)
        self._tokenizer = self._llm.get_tokenizer()
        self._stop_token_ids = generation_stop_token_ids(self._tokenizer)
        self.tokenizer_fingerprint = tokenizer_fingerprint(self._tokenizer)

    def generate(self, prompts: list[str], *, seed: int) -> list[Generation]:
        params = self._sampling_params_type(
            temperature=float(self.generation_config.get("temperature", 0.7)),
            top_p=float(self.generation_config.get("top_p", 0.9)),
            top_k=int(self.generation_config.get("top_k", -1)),
            max_tokens=int(self.generation_config.get("max_new_tokens", 1024)),
            seed=seed,
            skip_special_tokens=bool(self.generation_config.get("skip_special_tokens", False)),
            stop_token_ids=list(self._stop_token_ids),
        )
        thinking_value = self.generation_config.get("enable_thinking")
        enable_thinking = bool(thinking_value) if thinking_value is not None else None
        rendered_prompts = [
            render_user_prompt(
                self._tokenizer,
                prompt,
                enable_thinking=enable_thinking,
            )
            for prompt in prompts
        ]
        outputs = self._llm.generate(rendered_prompts, params)
        generations: list[Generation] = []
        for output in outputs:
            candidate = output.outputs[0]
            if output.prompt_token_ids is None:
                raise RuntimeError("vLLM did not return prompt token IDs")
            generations.append(
                Generation(
                    text=candidate.text,
                    prompt_tokens=len(output.prompt_token_ids),
                    response_tokens=len(candidate.token_ids),
                    finish_reason=str(getattr(candidate, "finish_reason", None) or "unknown"),
                )
            )
        return generations
