from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opd.config import config_hash
from opd.exceptions import ArtifactError
from opd.hashing import file_sha256, stable_hash
from opd.schemas import ArtifactManifest
from opd.tableio import read_json, write_json


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip()


def build_manifest(
    *,
    artifact_type: str,
    stage: str,
    config: dict[str, Any],
    files: Sequence[str | Path],
    record_count: int,
    success_count: int,
    failure_count: int = 0,
    upstream_artifact_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    file_hashes = {str(Path(path)): file_sha256(path) for path in files}
    canonical_config_hash = config_hash(config)
    identity = {
        "artifact_type": artifact_type,
        "stage": stage,
        "config_hash": canonical_config_hash,
        "upstream": upstream_artifact_ids or [],
        "files": file_hashes,
    }
    return ArtifactManifest(
        artifact_id=stable_hash(identity, length=20),
        artifact_type=artifact_type,
        stage=stage,
        git_commit=current_git_commit(),
        config_hash=canonical_config_hash,
        upstream_artifact_ids=upstream_artifact_ids or [],
        files=file_hashes,
        record_count=record_count,
        success_count=success_count,
        failure_count=failure_count,
        metadata=metadata or {},
    )


def save_manifest(path: str | Path, manifest: ArtifactManifest) -> Path:
    return write_json(path, manifest.model_dump(mode="json"))


def load_manifest(path: str | Path) -> ArtifactManifest:
    return ArtifactManifest.model_validate(read_json(path))


def verified_manifest_id(path: str | Path) -> str:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ArtifactError(f"Required upstream manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    for file_path, expected_hash in manifest.files.items():
        artifact_path = Path(file_path)
        if not artifact_path.exists():
            raise ArtifactError(f"Manifest file is missing: {artifact_path}")
        actual_hash = file_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise ArtifactError(
                f"Manifest checksum mismatch for {artifact_path}: {actual_hash} != {expected_hash}"
            )
    return manifest.artifact_id


def find_manifest_for_artifact(path: str | Path) -> Path | None:
    artifact_path = Path(path)
    candidates: list[Path] = []
    if artifact_path.is_file() or artifact_path.suffix:
        candidates.extend(
            [
                artifact_path.with_suffix(".manifest.json"),
                artifact_path.parent / "manifest.json",
                artifact_path.parent.parent / "manifest.json",
            ]
        )
    else:
        candidates.extend(
            [
                artifact_path / "manifest.json",
                artifact_path.parent / "manifest.json",
                artifact_path.parent.parent / "manifest.json",
            ]
        )
    if artifact_path.parent.exists():
        candidates.extend(sorted(artifact_path.parent.glob("*.manifest.json")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        manifest = load_manifest(candidate)
        artifact_resolved = artifact_path.resolve()
        for file_path in manifest.files:
            manifested = Path(file_path).resolve()
            if manifested == artifact_resolved:
                return candidate
            if artifact_path.is_dir() and artifact_resolved in manifested.parents:
                return candidate
    return None


def verified_artifact_manifest_id(path: str | Path) -> str:
    manifest_path = find_manifest_for_artifact(path)
    if manifest_path is None:
        raise ArtifactError(f"No manifest contains artifact: {Path(path)}")
    return verified_manifest_id(manifest_path)
