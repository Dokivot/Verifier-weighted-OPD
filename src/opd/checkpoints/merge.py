from __future__ import annotations

from pathlib import Path
from typing import Any

from opd.artifacts import build_manifest, save_manifest, verified_artifact_manifest_id
from opd.exceptions import DependencyError
from opd.tableio import write_json


def merge_adapter(
    config: dict[str, Any], *, adapter_path: str | Path, output_path: str | Path
) -> Path:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise DependencyError("Checkpoint merge requires `pip install -e .[gpu]`") from exc

    adapter = Path(adapter_path)
    output = Path(output_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter}")
    upstream_ids = [verified_artifact_manifest_id(adapter)]
    student = config["models"]["student"]
    use_qlora = bool(config["training"].get("qlora", True))
    base_model_path = Path(student["name"])
    if use_qlora and base_model_path.exists():
        base_manifest_id = verified_artifact_manifest_id(base_model_path)
        if base_manifest_id not in upstream_ids:
            upstream_ids.append(base_manifest_id)
    dtype_name = config["training"].get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    model_arguments: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": False,
    }
    if use_qlora and not base_model_path.exists():
        model_arguments["revision"] = student.get("revision")
    if use_qlora:
        base_model = AutoModelForCausalLM.from_pretrained(
            student["name"],
            **model_arguments,
        )
        model = PeftModel.from_pretrained(base_model, adapter)
        merged = model.merge_and_unload()
    else:
        full_model_arguments = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": False,
        }
        merged = AutoModelForCausalLM.from_pretrained(adapter, **full_model_arguments)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer_arguments: dict[str, Any] = {"trust_remote_code": False}
    tokenizer_source = adapter if not use_qlora else student["name"]
    if use_qlora and not base_model_path.exists():
        tokenizer_arguments["revision"] = student.get("tokenizer_revision", student.get("revision"))
    tokenizer_loader: Any = AutoTokenizer
    tokenizer = tokenizer_loader.from_pretrained(tokenizer_source, **tokenizer_arguments)
    tokenizer.save_pretrained(output)
    metadata_path = output / "merge_metadata.json"
    write_json(
        metadata_path,
        {
            "base_model": student["name"],
            "base_revision": student.get("revision"),
            "adapter_path": str(adapter),
            "parameter_update_mode": "qlora" if use_qlora else "full_parameter",
            "output_path": str(output),
            "seed": int(config["project"]["seed"]),
        },
    )
    output_files = [
        path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    ]
    manifest = build_manifest(
        artifact_type="merged_checkpoint",
        stage="checkpoint.merge",
        config=config,
        files=output_files,
        record_count=1,
        success_count=1,
        upstream_artifact_ids=upstream_ids,
        metadata={
            "base_model": student["name"],
            "base_revision": student.get("revision"),
            "adapter_path": str(adapter),
            "output_path": str(output),
            "parameter_update_mode": "qlora" if use_qlora else "full_parameter",
            "experiment_seed": int(config["project"]["seed"]),
        },
    )
    save_manifest(output / "manifest.json", manifest)
    return output
