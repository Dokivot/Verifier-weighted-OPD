from __future__ import annotations

from typing import Any


def parameter_update_mode(training: dict[str, Any]) -> str:
    """Resolve the parameter update mode, preserving legacy qlora configs."""
    configured = training.get("parameter_update_mode")
    if configured is not None:
        mode = str(configured)
    else:
        mode = "qlora" if bool(training.get("qlora", True)) else "full_parameter"
    if mode not in {"full_parameter", "lora", "qlora"}:
        raise ValueError(
            "training.parameter_update_mode must be one of full_parameter, lora, qlora"
        )
    return mode


def uses_lora(training: dict[str, Any]) -> bool:
    return parameter_update_mode(training) in {"lora", "qlora"}


def uses_qlora(training: dict[str, Any]) -> bool:
    return parameter_update_mode(training) == "qlora"
