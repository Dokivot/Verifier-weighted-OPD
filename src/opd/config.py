from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from opd.exceptions import ConfigurationError


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Config root must be a mapping: {config_path}")

    defaults = loaded.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ConfigurationError("defaults must be a string or list")

    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = (config_path.parent / str(default)).resolve()
        merged = _deep_merge(merged, load_config(default_path))
    config = _deep_merge(merged, loaded)
    _validate_seed_policy(config)
    _validate_rollout_lengths(config)
    _validate_online_k2(config)
    return config


def _validate_seed_policy(config: dict[str, Any]) -> None:
    project = config.get("project")
    if not isinstance(project, dict) or "seed" not in project:
        return
    seed = project["seed"]
    if not isinstance(seed, int):
        raise ConfigurationError("project.seed must be an integer")
    if project.get("seed_policy") != "single":
        return
    registered = project.get("registered_seeds", [seed])
    if registered != [seed]:
        raise ConfigurationError(
            "Single-seed mode requires project.registered_seeds to contain only project.seed"
        )


def _validate_rollout_lengths(config: dict[str, Any]) -> None:
    rollout = config.get("rollout")
    if not isinstance(rollout, dict):
        return
    generation = rollout.get("generation")
    if not isinstance(generation, dict):
        return
    max_model_length = rollout.get("max_model_length")
    max_new_tokens = generation.get("max_new_tokens")
    if max_model_length is None or max_new_tokens is None:
        return
    if not isinstance(max_model_length, int) or max_model_length <= 0:
        raise ConfigurationError("rollout.max_model_length must be a positive integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ConfigurationError("rollout.generation.max_new_tokens must be a positive integer")
    if max_new_tokens >= max_model_length:
        raise ConfigurationError(
            "rollout.generation.max_new_tokens must be smaller than "
            "rollout.max_model_length so the prompt fits in the vLLM context"
        )


def _validate_online_k2(config: dict[str, Any]) -> None:
    training = config.get("training")
    if not isinstance(training, dict) or training.get("backend") != "online_k2":
        return
    method = training.get("method")
    if method not in {"vanilla_k2", "sure_k2"}:
        raise ConfigurationError("online_k2 training.method must be vanilla_k2 or sure_k2")
    if training.get("parameter_update_mode", "full_parameter") != "full_parameter":
        raise ConfigurationError("online_k2 currently requires full_parameter updates")
    if training.get("master_parameter_dtype", "float32") != "float32":
        raise ConfigurationError(
            "online_k2 requires float32 master parameters for stable low-learning-rate updates"
        )
    if training.get("dtype", "bfloat16") != "bfloat16":
        raise ConfigurationError("online_k2 currently requires bfloat16 compute")
    positive_integer_keys = (
        "max_steps",
        "global_prompt_batch_size",
        "rollout_micro_batch_size",
        "teacher_micro_batch_size",
        "student_micro_batch_size",
        "max_prompt_tokens",
        "max_response_tokens",
        "max_model_length",
    )
    for key in positive_integer_keys:
        value = training.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"training.{key} must be a positive integer")
    if (
        training["max_prompt_tokens"] + training["max_response_tokens"]
        > training["max_model_length"]
    ):
        raise ConfigurationError(
            "training.max_prompt_tokens + training.max_response_tokens cannot exceed "
            "training.max_model_length"
        )
    alpha = float(training.get("sure_alpha", 0.0))
    if method == "sure_k2" and alpha <= 0:
        raise ConfigurationError("sure_k2 requires training.sure_alpha > 0")
    if method == "vanilla_k2" and alpha != 0.0:
        raise ConfigurationError("vanilla_k2 requires training.sure_alpha = 0")
    if int(training.get("rollouts_per_prompt", 1)) != 1:
        raise ConfigurationError("online_k2 currently requires one rollout per prompt")
    quality_gate = training.get("quality_gate", {})
    if not isinstance(quality_gate, dict):
        raise ConfigurationError("training.quality_gate must be a mapping")
    warning_rate = float(quality_gate.get("warning_truncation_rate", 0.05))
    maximum_rate = float(quality_gate.get("maximum_truncation_rate", 0.2))
    if not 0.0 <= warning_rate <= maximum_rate <= 1.0:
        raise ConfigurationError(
            "training truncation thresholds must satisfy 0 <= warning <= maximum <= 1"
        )
    invocation_step_limit = training.get("invocation_step_limit")
    if invocation_step_limit is not None and (
        not isinstance(invocation_step_limit, int) or invocation_step_limit <= 0
    ):
        raise ConfigurationError("training.invocation_step_limit must be a positive integer")
    checkpointing = config.get("checkpointing", {})
    if not isinstance(checkpointing, dict):
        raise ConfigurationError("checkpointing must be a mapping")
    for key in ("minimum_free_disk_gib", "reserve_artifact_gib", "safety_factor"):
        value = checkpointing.get(key)
        if value is not None and float(value) <= 0:
            raise ConfigurationError(f"checkpointing.{key} must be positive")
    if checkpointing.get("rolling_retention", 1) != 1:
        raise ConfigurationError("online_k2 currently supports checkpointing.rolling_retention = 1")


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigurationError(f"Missing required config key: {dotted_key}")
        current = current[part]
    return current
