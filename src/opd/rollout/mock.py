from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from opd.rollout.base import Generation
from opd.tokenizers import mock_tokenizer_fingerprint

Number: TypeAlias = float | int
BinaryOperator: TypeAlias = Callable[[Number, Number], Number]
UnaryOperator: TypeAlias = Callable[[Number], Number]

_BINARY_OPS: dict[type[ast.operator], BinaryOperator] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], UnaryOperator] = {
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> Number:
    def evaluate(node: ast.AST) -> Number:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Unsupported expression")

    return evaluate(ast.parse(expression, mode="eval").body)


@dataclass
class MockRolloutBackend:
    model_name: str = "mock-student"
    model_revision: str = "mock-v1"
    tokenizer_revision: str = "mock-tokenizer-v1"

    @property
    def tokenizer_fingerprint(self) -> str:
        return mock_tokenizer_fingerprint(vocab_size=256, revision=self.tokenizer_revision)

    def generate(self, prompts: list[str], *, seed: int) -> list[Generation]:
        del seed
        generations: list[Generation] = []
        for prompt in prompts:
            match = re.search(r"(?:compute|calculate)\s+([^.?]+)", prompt, re.IGNORECASE)
            if match:
                try:
                    value = _safe_eval(match.group(1).strip())
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    text = (
                        f"I evaluate the expression step by step. Final answer: \\boxed{{{value}}}"
                    )
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                    text = "I cannot determine the final answer reliably."
            else:
                text = "I cannot determine the final answer reliably."
            generations.append(
                Generation(
                    text=text,
                    prompt_tokens=len(prompt.split()),
                    response_tokens=len(text.split()),
                )
            )
        return generations
