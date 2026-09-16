from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def run_doctor(config: dict[str, Any] | None = None) -> dict[str, Any]:
    optional = [
        "numpy",
        "pydantic",
        "yaml",
        "pyarrow",
        "datasets",
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "vllm",
        "math_verify",
        "lighteval",
        "more_itertools",
        "wandb",
        "xxhash",
    ]
    modules = {name: importlib.util.find_spec(name) is not None for name in optional}
    gpu: dict[str, Any] = {"available": False, "count": 0, "names": []}
    if modules["torch"]:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        gpu["count"] = int(torch.cuda.device_count())
        gpu["names"] = [torch.cuda.get_device_name(index) for index in range(gpu["count"])]
    paths = {}
    if config:
        for key, value in config.get("paths", {}).items():
            path = Path(value)
            paths[key] = {
                "path": str(path),
                "exists": path.exists(),
                "writable_parent": (path if path.exists() else path.parent).exists(),
            }
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "git": shutil.which("git"),
        "docker": shutil.which("docker"),
        "modules": modules,
        "gpu": gpu,
        "paths": paths,
    }
    return result
