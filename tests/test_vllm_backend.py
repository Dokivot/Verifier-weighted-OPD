from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from opd.rollout.vllm_backend import VLLMRolloutBackend


class _Tokenizer:
    eos_token_id = 151643
    unk_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {"<eos>": 0, "hello": 1}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(token, 0)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return messages[0]["content"]


class VLLMBackendTest(unittest.TestCase):
    def test_forwards_engine_and_sampling_configuration(self) -> None:
        captured: dict[str, object] = {}
        module = ModuleType("vllm")

        class FakeSamplingParams:
            def __init__(self, **kwargs: object) -> None:
                captured["sampling"] = kwargs

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                captured["engine"] = kwargs

            def get_tokenizer(self) -> _Tokenizer:
                return _Tokenizer()

            def generate(self, prompts: list[str], params: object) -> list[object]:
                captured["prompts"] = prompts
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        outputs=[
                            SimpleNamespace(text="answer", token_ids=[3], finish_reason="stop")
                        ],
                    )
                ]

        module.LLM = FakeLLM  # type: ignore[attr-defined]
        module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
        with (
            patch.dict(sys.modules, {"vllm": module}),
            patch.object(Path, "exists", return_value=False),
        ):
            backend = VLLMRolloutBackend(
                model_name="remote/model",
                model_revision="fixed-revision",
                tokenizer_revision="fixed-revision",
                generation_config={
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_new_tokens": 512,
                },
                tensor_parallel_size=2,
                dtype="bfloat16",
                max_model_length=4096,
                gpu_memory_utilization=0.85,
            )
            generations = backend.generate(["question"], seed=42)

        self.assertEqual(
            captured["engine"],
            {
                "model": "remote/model",
                "tensor_parallel_size": 2,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.85,
                "trust_remote_code": False,
                "revision": "fixed-revision",
            },
        )
        self.assertEqual(
            captured["sampling"],
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 512,
                "seed": 42,
                "skip_special_tokens": False,
                "stop_token_ids": [151643, 151645],
            },
        )
        self.assertEqual(generations[0].text, "answer")
        self.assertEqual(generations[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
