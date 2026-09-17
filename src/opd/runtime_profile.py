from __future__ import annotations

import re
from collections.abc import Sequence

BLACKWELL_TORCH_VERSION = "2.7.1+cu128"
BLACKWELL_CUDA_VERSION = "12.8"
BLACKWELL_MIN_DRIVER = (570, 26)
BLACKWELL_MIN_MEMORY_GIB = 90.0


def _version_pair(value: str, *, name: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"Unable to parse {name} version: {value}")
    return int(match.group(1)), int(match.group(2))


def validate_rtx_pro_6000_runtime(
    *,
    torch_version: str,
    torch_cuda: str | None,
    driver_version: str,
    device_name: str,
    compute_capability: tuple[int, int],
    memory_gib: float,
    compiled_architectures: Sequence[str],
) -> None:
    if torch_version != BLACKWELL_TORCH_VERSION:
        raise ValueError(
            f"RTX PRO 6000 profile requires torch=={BLACKWELL_TORCH_VERSION}; found {torch_version}"
        )
    if torch_cuda != BLACKWELL_CUDA_VERSION:
        raise ValueError(
            f"RTX PRO 6000 profile requires a CUDA {BLACKWELL_CUDA_VERSION} PyTorch wheel; "
            f"found {torch_cuda or 'none'}"
        )
    if _version_pair(driver_version, name="NVIDIA driver") < BLACKWELL_MIN_DRIVER:
        minimum = ".".join(str(part) for part in BLACKWELL_MIN_DRIVER)
        raise ValueError(
            f"RTX PRO 6000 profile requires NVIDIA driver >= {minimum}; found {driver_version}"
        )
    normalized_name = device_name.upper()
    if "RTX PRO 6000" not in normalized_name or "BLACKWELL" not in normalized_name:
        raise ValueError(
            f"This branch targets an NVIDIA RTX PRO 6000 Blackwell GPU; found {device_name}"
        )
    if compute_capability < (12, 0):
        raise ValueError(
            "RTX PRO 6000 Blackwell requires compute capability 12.0 or newer; "
            f"found {compute_capability[0]}.{compute_capability[1]}"
        )
    if memory_gib < BLACKWELL_MIN_MEMORY_GIB:
        raise ValueError(
            f"Expected at least {BLACKWELL_MIN_MEMORY_GIB:.0f} GiB for the 96GB profile; "
            f"found {memory_gib:.1f} GiB"
        )
    has_blackwell_architecture = any(
        architecture in {"sm_120", "compute_120"} for architecture in compiled_architectures
    )
    if not has_blackwell_architecture:
        raise ValueError(
            "The installed PyTorch wheel does not contain Blackwell SM 12.0 kernels: "
            f"{list(compiled_architectures)}"
        )
