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


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigurationError(f"Missing required config key: {dotted_key}")
        current = current[part]
    return current
