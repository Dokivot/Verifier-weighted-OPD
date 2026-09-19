from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from opd.artifacts import verified_artifact_manifest_id
from opd.config import config_hash, normalized_online_training_config
from opd.hashing import file_sha256, stable_hash
from opd.prompts import render_user_prompt
from opd.schemas import PromptRecord
from opd.tableio import read_records, write_json, write_records
from opd.tokenizers import tokenizer_fingerprint
from opd.training.losses import sampled_token_k2_torch
from opd.verifier.math import MathVerifier


@dataclass
class OnlineTrajectory:
    prompt: PromptRecord
    rendered_prompt: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response: str
    seed: int
    truncated: bool
    finish_reason: str
    terminal_token_id: int | None


def online_run_id(
    *,
    student_config: dict[str, Any],
    teacher_config: dict[str, Any],
    input_checksum: str,
    training: dict[str, Any],
    seed: int,
) -> str:
    return stable_hash(
        {
            "student": student_config,
            "teacher": teacher_config,
            "input_checksum": input_checksum,
            "training": normalized_online_training_config(training),
            "seed": seed,
        },
        length=20,
    )


def _require_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_constant_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError("Online K2 training requires the GPU optional dependencies") from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "get_constant_schedule_with_warmup": get_constant_schedule_with_warmup,
    }


def _model_source(model_config: dict[str, Any], *, override: Path | None = None) -> str:
    return str(override if override is not None else model_config["name"])


def _load_tokenizer(
    dependencies: dict[str, Any],
    model_config: dict[str, Any],
    *,
    override: Path | None = None,
) -> Any:
    source = _model_source(model_config, override=override)
    arguments: dict[str, Any] = {"trust_remote_code": False}
    if override is None and not Path(source).exists():
        arguments["revision"] = model_config.get("tokenizer_revision", model_config["revision"])
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(source, **arguments)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    dependencies: dict[str, Any],
    model_config: dict[str, Any],
    *,
    dtype: Any,
    device: Any,
    override: Path | None = None,
) -> Any:
    source = _model_source(model_config, override=override)
    arguments: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if override is None and not Path(source).exists():
        arguments["revision"] = model_config["revision"]
    model = dependencies["AutoModelForCausalLM"].from_pretrained(source, **arguments)
    return model.to(device)


def _format_problem(problem: str, prompt_template: str) -> str:
    if "{problem}" not in prompt_template:
        raise ValueError("training.prompt_template must contain {problem}")
    return prompt_template.replace("{problem}", problem)


def _encode_prompts(
    prompts: list[PromptRecord],
    tokenizer: Any,
    training: dict[str, Any],
) -> list[tuple[PromptRecord, str, list[int]]]:
    encoded: list[tuple[PromptRecord, str, list[int]]] = []
    prompt_template = str(
        training.get(
            "prompt_template",
            "{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.",
        )
    )
    max_prompt_tokens = int(training["max_prompt_tokens"])
    enable_thinking = bool(training.get("enable_thinking", False))
    thinking_marker = training.get("thinking_marker")
    if thinking_marker is not None and not isinstance(thinking_marker, str):
        raise ValueError("training.thinking_marker must be a string when configured")
    for record in prompts:
        problem = _format_problem(record.problem, prompt_template)
        rendered = render_user_prompt(
            tokenizer,
            problem,
            enable_thinking=enable_thinking,
            thinking_marker=thinking_marker,
        )
        token_ids = list(tokenizer.encode(rendered, add_special_tokens=True))
        if not token_ids:
            raise ValueError(f"Prompt {record.sample_id} produced no tokens")
        if len(token_ids) > max_prompt_tokens:
            raise ValueError(
                f"Prompt {record.sample_id} has {len(token_ids)} tokens, exceeding "
                f"training.max_prompt_tokens={max_prompt_tokens}"
            )
        encoded.append((record, rendered, token_ids))
    return encoded


def _padded_batch(
    sequences: list[list[int]],
    *,
    pad_token_id: int,
    torch: Any,
    device: Any,
    left_pad: bool,
) -> tuple[Any, Any]:
    max_length = max(len(sequence) for sequence in sequences)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence in sequences:
        padding = max_length - len(sequence)
        if left_pad:
            input_rows.append([pad_token_id] * padding + sequence)
            mask_rows.append([0] * padding + [1] * len(sequence))
        else:
            input_rows.append(sequence + [pad_token_id] * padding)
            mask_rows.append([1] * len(sequence) + [0] * padding)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
    )


def _trim_generated_tokens(
    token_ids: list[int],
    *,
    eos_token_id: int | None = None,
    pad_token_id: int,
) -> list[int]:
    response_ids, _, _ = _classify_generated_tokens(
        token_ids,
        stop_token_ids=() if eos_token_id is None else (eos_token_id,),
        pad_token_id=pad_token_id,
    )
    return response_ids


def _classify_generated_tokens(
    token_ids: list[int],
    *,
    stop_token_ids: tuple[int, ...],
    pad_token_id: int,
) -> tuple[list[int], str, int | None]:
    stop_tokens = set(stop_token_ids)
    trimmed: list[int] = []
    for token_id in token_ids:
        if token_id in stop_tokens:
            trimmed.append(token_id)
            return trimmed, "stop", token_id
        if token_id == pad_token_id:
            return trimmed, "padding", None
        trimmed.append(token_id)
    return trimmed, "length", None


def _stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return all Qwen-compatible generation terminators present in a tokenizer."""
    candidates: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        candidates.append(eos_token_id)
    elif isinstance(eos_token_id, list):
        candidates.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unknown_token_id:
            candidates.append(token_id)

    unique = tuple(dict.fromkeys(candidates))
    if not unique:
        raise ValueError("Tokenizer exposes no usable generation stop token")
    return unique


def _response_length_percentiles(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"p50": 0, "p95": 0, "p99": 0, "maximum": 0}
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[int((len(ordered) - 1) * fraction)]

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": ordered[-1],
    }


def _generate_trajectories(
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    training: dict[str, Any],
    *,
    step: int,
    base_seed: int,
    torch: Any,
    device: Any,
    compute_dtype: Any,
) -> tuple[list[OnlineTrajectory], float]:
    started = perf_counter()
    encoded = _encode_prompts(prompts, tokenizer, training)
    micro_batch_size = int(training["rollout_micro_batch_size"])
    max_response_tokens = int(training["max_response_tokens"])
    trajectories: list[OnlineTrajectory] = []
    stop_token_ids = _stop_token_ids(tokenizer)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for start in range(0, len(encoded), micro_batch_size):
        batch = encoded[start : start + micro_batch_size]
        input_ids, attention_mask = _padded_batch(
            [item[2] for item in batch],
            pad_token_id=int(tokenizer.pad_token_id),
            torch=torch,
            device=device,
            left_pad=True,
        )
        batch_seed = base_seed + step * 1_000_000 + start
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=device.type == "cuda",
            ),
        ):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(training.get("temperature", 1.0)),
                top_p=float(training.get("top_p", 1.0)),
                top_k=int(training.get("top_k", 0)),
                max_new_tokens=max_response_tokens,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=list(stop_token_ids),
                use_cache=True,
            )
        prompt_width = int(input_ids.shape[1])
        for offset, (record, rendered, prompt_ids) in enumerate(batch):
            response_ids, finish_reason, terminal_token_id = _classify_generated_tokens(
                generated[offset, prompt_width:].detach().cpu().tolist(),
                stop_token_ids=stop_token_ids,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            if not response_ids:
                raise RuntimeError(f"Student generated no response tokens for {record.sample_id}")
            trajectories.append(
                OnlineTrajectory(
                    prompt=record,
                    rendered_prompt=rendered,
                    prompt_token_ids=prompt_ids,
                    response_token_ids=response_ids,
                    response=str(tokenizer.decode(response_ids, skip_special_tokens=True)),
                    seed=batch_seed,
                    truncated=finish_reason == "length",
                    finish_reason=finish_reason,
                    terminal_token_id=terminal_token_id,
                )
            )
    model.config.use_cache = False
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    model.train()
    return trajectories, perf_counter() - started


def _sampled_logprobs(
    model: Any,
    trajectories: list[OnlineTrajectory],
    *,
    micro_batch_size: int,
    torch: Any,
    device: Any,
    pad_token_id: int,
    require_grad: bool,
    compute_dtype: Any,
) -> list[Any]:
    results: list[Any] = []
    context = torch.enable_grad if require_grad else torch.inference_mode
    for start in range(0, len(trajectories), micro_batch_size):
        batch = trajectories[start : start + micro_batch_size]
        sequences = [
            [*trajectory.prompt_token_ids, *trajectory.response_token_ids] for trajectory in batch
        ]
        input_ids, attention_mask = _padded_batch(
            sequences,
            pad_token_id=pad_token_id,
            torch=torch,
            device=device,
            left_pad=False,
        )
        with (
            context(),
            torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=device.type == "cuda",
            ),
        ):
            backbone = getattr(model, "model", None)
            lm_head = getattr(model, "lm_head", None)
            if backbone is None or lm_head is None:
                raise TypeError(
                    "Online K2 requires a causal LM exposing model.model and model.lm_head"
                )
            hidden_states = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            for row_index, trajectory in enumerate(batch):
                response_length = len(trajectory.response_token_ids)
                response_start = len(trajectory.prompt_token_ids) - 1
                response_hidden_states = hidden_states[
                    row_index,
                    response_start : response_start + response_length,
                ]
                targets = torch.tensor(
                    trajectory.response_token_ids,
                    dtype=torch.long,
                    device=device,
                )
                token_chunk_size = 256
                chunks: list[Any] = []
                for token_start in range(0, response_length, token_chunk_size):
                    token_end = token_start + token_chunk_size
                    chunk_logits = lm_head(response_hidden_states[token_start:token_end]).float()
                    chunk_targets = targets[token_start:token_end]
                    selected = chunk_logits.gather(
                        -1,
                        chunk_targets.unsqueeze(-1),
                    ).squeeze(-1)
                    chunks.append(selected - torch.logsumexp(chunk_logits, dim=-1))
                logprobs = torch.cat(chunks)
                results.append(logprobs if require_grad else logprobs.detach().cpu())
    return results


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_state(
    destination: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    state: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.partial")
    backup = destination.with_name(f"{destination.name}.previous")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model_dir = staging / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    state_path = staging / "training_state.pt"
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(torch),
        },
        state_path,
    )
    write_json(
        staging / "checkpoint_state.json",
        {key: value for key, value in state.items() if key not in {"history"}}
        | {"history_length": len(state["history"])},
    )
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _save_model_checkpoint(model: Any, tokenizer: Any, destination: Path) -> None:
    staging = destination.with_name(f"{destination.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)


def _checkpoint_storage_plan(
    *,
    parameter_count: int,
    max_steps: int,
    milestone_steps: set[int],
    checkpointing: dict[str, Any],
    free_bytes: int,
) -> dict[str, Any]:
    """Estimate the peak disk footprint while atomically replacing rolling state."""
    if parameter_count <= 0:
        raise ValueError("Checkpoint storage estimate requires a positive parameter count")
    gib = 1024**3
    model_bytes = parameter_count * 4
    optimizer_bytes = parameter_count * 8
    rolling_checkpoint_bytes = model_bytes + optimizer_bytes
    retained_milestones = sum(step < max_steps for step in milestone_steps)
    model_only_checkpoints = retained_milestones + 1
    reserve_bytes = int(float(checkpointing.get("reserve_artifact_gib", 12)) * gib)
    safety_factor = float(checkpointing.get("safety_factor", 1.1))
    estimated_peak_bytes = int(
        (2 * rolling_checkpoint_bytes + model_only_checkpoints * model_bytes + reserve_bytes)
        * safety_factor
    )
    minimum_free_bytes = int(float(checkpointing.get("minimum_free_disk_gib", 80)) * gib)
    required_free_bytes = max(estimated_peak_bytes, minimum_free_bytes)
    return {
        "parameter_count": parameter_count,
        "model_gib": model_bytes / gib,
        "optimizer_gib": optimizer_bytes / gib,
        "rolling_checkpoint_gib": rolling_checkpoint_bytes / gib,
        "atomic_rolling_copies": 2,
        "retained_milestone_model_checkpoints": retained_milestones,
        "final_model_checkpoints": 1,
        "reserve_artifact_gib": reserve_bytes / gib,
        "safety_factor": safety_factor,
        "estimated_peak_gib": estimated_peak_bytes / gib,
        "minimum_free_disk_gib": minimum_free_bytes / gib,
        "required_free_disk_gib": required_free_bytes / gib,
        "available_free_disk_gib": free_bytes / gib,
        "passed": free_bytes >= required_free_bytes,
    }


def _assert_checkpoint_storage(
    *,
    output_dir: Path,
    parameter_count: int,
    max_steps: int,
    milestone_steps: set[int],
    checkpointing: dict[str, Any],
) -> None:
    free_bytes = shutil.disk_usage(output_dir).free
    plan = _checkpoint_storage_plan(
        parameter_count=parameter_count,
        max_steps=max_steps,
        milestone_steps=milestone_steps,
        checkpointing=checkpointing,
        free_bytes=free_bytes,
    )
    write_json(output_dir / "checkpoint_storage_plan.json", plan)
    if not plan["passed"]:
        raise RuntimeError(
            "Insufficient free disk space for online K2 checkpoints: "
            f"available={plan['available_free_disk_gib']:.1f} GiB, "
            f"required={plan['required_free_disk_gib']:.1f} GiB. "
            "Free space or lower checkpoint retention before training."
        )


def _resolve_resume_dir(resume_dir: Path) -> Path:
    if resume_dir.exists():
        return resume_dir
    backup = resume_dir.with_name(f"{resume_dir.name}.previous")
    if backup.exists():
        return backup
    return resume_dir


def _load_resume_state(torch: Any, resume_dir: Path) -> dict[str, Any]:
    resume_dir = _resolve_resume_dir(resume_dir)
    state_path = resume_dir / "training_state.pt"
    model_dir = resume_dir / "model"
    if not state_path.exists() or not model_dir.exists():
        raise FileNotFoundError(f"Incomplete online K2 checkpoint: {resume_dir}")
    return dict(torch.load(state_path, map_location="cpu", weights_only=False))


def _validate_step_artifacts(
    step_dir: Path,
    *,
    global_step: int,
    history: list[dict[str, Any]],
    policy_hash: str,
) -> None:
    existing = sorted(step_dir.glob("step_*.parquet")) if step_dir.exists() else []
    expected_names = [f"step_{step:06d}.parquet" for step in range(1, global_step + 1)]
    if [path.name for path in existing] != expected_names:
        raise RuntimeError(
            "Committed online-K2 step artifacts do not match the resume checkpoint; "
            "refusing to consume stale or orphaned trajectories"
        )
    if len(history) != global_step:
        raise RuntimeError("Resume history length does not match global_step")
    for step, (path, metrics) in enumerate(zip(existing, history, strict=True), start=1):
        if int(metrics["step"]) != step:
            raise RuntimeError("Resume history contains a non-contiguous optimizer step")
        calculated = stable_hash(
            {
                "parent": str(metrics["policy_hash"]),
                "step": step,
                "step_artifact": file_sha256(path),
            },
            length=24,
        )
        if calculated != str(metrics["next_policy_hash"]):
            raise RuntimeError(f"Policy hash chain mismatch at optimizer step {step}")
    if history and str(history[-1]["next_policy_hash"]) != policy_hash:
        raise RuntimeError("Resume policy hash does not match the committed step chain")


def _audit_prompt_lengths(
    records: list[PromptRecord],
    tokenizer: Any,
    training: dict[str, Any],
) -> dict[str, Any]:
    lengths: list[int] = []
    for record in records:
        encoded = _encode_prompts([record], tokenizer, training)
        lengths.append(len(encoded[0][2]))
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[int((len(ordered) - 1) * fraction)]

    return {
        "records": len(lengths),
        "minimum_prompt_tokens": min(lengths),
        "mean_prompt_tokens": sum(lengths) / len(lengths),
        "p50_prompt_tokens": percentile(0.50),
        "p95_prompt_tokens": percentile(0.95),
        "maximum_prompt_tokens": max(lengths),
        "configured_max_prompt_tokens": int(training["max_prompt_tokens"]),
    }


def _step_records(
    trajectories: list[OnlineTrajectory],
    teacher_logprobs: list[Any],
    student_logprobs: list[Any],
    *,
    verifier: MathVerifier,
    step: int,
    policy_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory, teacher_values, student_values in zip(
        trajectories,
        teacher_logprobs,
        student_logprobs,
        strict=True,
    ):
        verification = verifier.verify(
            rollout_id=f"step{step}_{trajectory.prompt.sample_id}",
            sample_id=trajectory.prompt.sample_id,
            response=trajectory.response,
            reference_answer=trajectory.prompt.reference_answer,
        )
        rows.append(
            {
                "step": step,
                "sample_id": trajectory.prompt.sample_id,
                "policy_hash": policy_hash,
                "seed": trajectory.seed,
                "prompt": trajectory.prompt.problem,
                "response": trajectory.response,
                "prompt_token_ids": trajectory.prompt_token_ids,
                "response_token_ids": trajectory.response_token_ids,
                "teacher_sampled_logprobs": teacher_values.tolist(),
                "student_sampled_logprobs": student_values.tolist(),
                "response_tokens": len(trajectory.response_token_ids),
                "truncated": trajectory.truncated,
                "finish_reason": trajectory.finish_reason,
                "terminal_token_id": trajectory.terminal_token_id,
                "verifier_status": verification.status.value,
                "verifier_score": verification.score,
                "extracted_answer": verification.extracted_answer,
            }
        )
    return rows


def train_online_k2(config: dict[str, Any]) -> Path:
    dependencies = _require_dependencies()
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("Online K2 training requires a CUDA GPU")
    training = config["training"]
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda", int(training.get("device_index", 0)))
    compute_dtype = getattr(torch, str(training.get("dtype", "bfloat16")))
    master_parameter_dtype = getattr(
        torch,
        str(training.get("master_parameter_dtype", "float32")),
    )
    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "gpu_count": int(torch.cuda.device_count()),
        "device_index": int(device.index or 0),
        "device_name": str(torch.cuda.get_device_name(device)),
        "total_memory_gib": float(properties.total_memory) / (1024**3),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "compiled_architectures": list(torch.cuda.get_arch_list()),
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda or "unknown"),
        "device": str(device),
        "dtype": str(training.get("dtype", "bfloat16")),
        "master_parameter_dtype": str(training.get("master_parameter_dtype", "float32")),
    }
    hardware_fingerprint = stable_hash(hardware, length=24)
    input_path = Path(training["input_path"])
    if bool(training.get("require_decontaminated_input", False)):
        expected_clean_path = (
            Path(config["paths"]["data_dir"])
            / "contamination"
            / ("train_clean" + input_path.suffix)
        )
        if input_path != expected_clean_path:
            raise ValueError(
                "Online K2 formal/pilot training must use the contamination clean split: "
                f"expected {expected_clean_path}, got {input_path}"
            )
        verified_artifact_manifest_id(input_path)
    input_checksum = file_sha256(input_path)
    records = [PromptRecord.model_validate(row) for row in read_records(input_path)]
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("Online K2 input contains duplicate sample_id values")
    random.Random(seed).shuffle(records)
    max_records = training.get("max_records")
    if max_records is not None:
        records = records[: int(max_records)]
    global_batch_size = int(training["global_prompt_batch_size"])
    max_steps = int(training["max_steps"])
    required_records = global_batch_size * max_steps
    if len(records) < required_records:
        raise ValueError(
            f"Online K2 needs {required_records} unique prompts for {max_steps} steps, "
            f"but only {len(records)} are available"
        )

    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_value = training.get("resume_from_checkpoint")
    resume_dir = _resolve_resume_dir(Path(resume_value)) if resume_value else None
    resume_state = _load_resume_state(torch, resume_dir) if resume_dir else None
    student_config = config["models"]["student"]
    teacher_config = config["models"]["teacher"]
    initial_student_value = training.get("initial_student_checkpoint")
    if resume_dir is not None and initial_student_value:
        raise ValueError(
            "training.initial_student_checkpoint cannot be combined with "
            "training.resume_from_checkpoint"
        )
    student_override = resume_dir / "model" if resume_dir else None
    if student_override is None and initial_student_value:
        student_override = Path(str(initial_student_value))
        if not student_override.exists():
            raise FileNotFoundError(
                "Configured initial student checkpoint does not exist: "
                f"{student_override}"
            )
    student_tokenizer = _load_tokenizer(
        dependencies,
        student_config,
        override=student_override,
    )
    teacher_tokenizer = _load_tokenizer(dependencies, teacher_config)
    student_fingerprint = tokenizer_fingerprint(student_tokenizer)
    teacher_fingerprint = tokenizer_fingerprint(teacher_tokenizer)
    if student_fingerprint != teacher_fingerprint:
        raise ValueError(
            "Student and Teacher tokenizer vocabularies differ; sampled-token IDs cannot align"
        )
    prompt_audit = _audit_prompt_lengths(
        records[:required_records],
        student_tokenizer,
        training,
    )
    write_json(output_dir / "prompt_length_audit.json", prompt_audit)
    student = _load_model(
        dependencies,
        student_config,
        dtype=master_parameter_dtype,
        device=device,
        override=student_override,
    )
    student.config.use_cache = False
    if bool(training.get("gradient_checkpointing", True)):
        student.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.999])),
        weight_decay=float(training.get("weight_decay", 0.01)),
        foreach=False,
    )
    scheduler = dependencies["get_constant_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=int(training.get("warmup_steps", 10)),
    )
    teacher = _load_model(
        dependencies,
        teacher_config,
        dtype=compute_dtype,
        device=device,
    )
    teacher.eval()
    teacher.requires_grad_(False)

    run_id = online_run_id(
        student_config=student_config,
        teacher_config=teacher_config,
        input_checksum=input_checksum,
        training=training,
        seed=seed,
    )
    global_step = 0
    data_cursor = 0
    history: list[dict[str, Any]] = []
    policy_hash = stable_hash(
        {"student": student_config, "seed": seed, "run_id": run_id},
        length=24,
    )
    if resume_state is not None:
        if resume_state["run_id"] != run_id:
            raise ValueError("Resume checkpoint belongs to a different online K2 run")
        if resume_state["input_checksum"] != input_checksum:
            raise ValueError("Resume checkpoint input checksum does not match current dataset")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        data_cursor = int(resume_state["data_cursor"])
        policy_hash = str(resume_state["policy_hash"])
        history = [dict(item) for item in resume_state["history"]]
        _restore_rng_state(torch, resume_state["rng"])

    verifier = MathVerifier()
    alpha = float(training.get("sure_alpha", 0.0))
    milestone_steps = {int(value) for value in training.get("milestone_steps", [])}
    state_save_steps = int(training.get("state_save_steps", 1))
    if state_save_steps <= 0:
        raise ValueError("training.state_save_steps must be positive")
    telemetry_dir = output_dir / "telemetry"
    step_dir = output_dir / "steps"
    rolling_dir = output_dir / "rolling"
    checkpoints_dir = output_dir / "checkpoints"
    _assert_checkpoint_storage(
        output_dir=output_dir,
        parameter_count=sum(parameter.numel() for parameter in student.parameters()),
        max_steps=max_steps,
        milestone_steps=milestone_steps,
        checkpointing=dict(config.get("checkpointing", {})),
    )
    _validate_step_artifacts(
        step_dir,
        global_step=global_step,
        history=history,
        policy_hash=policy_hash,
    )
    invocation_start_step = global_step
    invocation_step_limit_value = training.get("invocation_step_limit")
    invocation_step_limit = (
        int(invocation_step_limit_value) if invocation_step_limit_value is not None else max_steps
    )
    quality_gate = training.get("quality_gate", {})
    quality_gate_enabled = bool(quality_gate.get("enabled", True))
    quality_gate_minimum = int(quality_gate.get("minimum_samples", global_batch_size))
    warning_truncation_rate = float(quality_gate.get("warning_truncation_rate", 0.05))
    maximum_truncation_rate = float(quality_gate.get("maximum_truncation_rate", 0.2))

    while global_step < max_steps and global_step - invocation_start_step < invocation_step_limit:
        step = global_step + 1
        batch = records[data_cursor : data_cursor + global_batch_size]
        if len(batch) != global_batch_size:
            raise RuntimeError("Online K2 data cursor reached an incomplete global batch")
        step_started = perf_counter()
        trajectories, rollout_seconds = _generate_trajectories(
            student,
            student_tokenizer,
            batch,
            training,
            step=step,
            base_seed=seed,
            torch=torch,
            device=device,
            compute_dtype=compute_dtype,
        )
        total_response_tokens = sum(
            len(trajectory.response_token_ids) for trajectory in trajectories
        )
        if total_response_tokens <= 0:
            raise RuntimeError(f"Online K2 step {step} produced no response tokens")
        rollout_truncated = sum(trajectory.truncated for trajectory in trajectories)
        rollout_truncation_rate = rollout_truncated / len(trajectories)
        finish_reason_counts = {
            reason: sum(trajectory.finish_reason == reason for trajectory in trajectories)
            for reason in sorted({trajectory.finish_reason for trajectory in trajectories})
        }
        write_json(
            telemetry_dir / f"step_{step:06d}_quality_gate.json",
            {
                "step": step,
                "status": (
                    "failed"
                    if quality_gate_enabled
                    and len(trajectories) >= quality_gate_minimum
                    and rollout_truncation_rate > maximum_truncation_rate
                    else "warning"
                    if rollout_truncation_rate > warning_truncation_rate
                    else "passed"
                ),
                "successful_records": len(trajectories),
                "truncated_records": rollout_truncated,
                "truncation_rate": rollout_truncation_rate,
                "finish_reason_counts": finish_reason_counts,
                "response_length_tokens": _response_length_percentiles(
                    [len(trajectory.response_token_ids) for trajectory in trajectories]
                ),
                "stop_token_ids": list(_stop_token_ids(student_tokenizer)),
                "stop_tokens": [
                    str(student_tokenizer.convert_ids_to_tokens(token_id))
                    for token_id in _stop_token_ids(student_tokenizer)
                ],
                "warning_truncation_rate": warning_truncation_rate,
                "maximum_truncation_rate": maximum_truncation_rate,
                "minimum_samples": quality_gate_minimum,
                "enabled": quality_gate_enabled,
            },
        )
        if rollout_truncation_rate > warning_truncation_rate:
            print(
                f"WARNING: online K2 step {step} truncation rate is {rollout_truncation_rate:.2%}"
            )
        if (
            quality_gate_enabled
            and len(trajectories) >= quality_gate_minimum
            and rollout_truncation_rate > maximum_truncation_rate
        ):
            raise RuntimeError(
                f"Online K2 truncation gate failed at step {step}: "
                f"{rollout_truncation_rate:.2%} > {maximum_truncation_rate:.2%}"
            )

        teacher_started = perf_counter()
        teacher_logprobs = _sampled_logprobs(
            teacher,
            trajectories,
            micro_batch_size=int(training["teacher_micro_batch_size"]),
            torch=torch,
            device=device,
            pad_token_id=int(teacher_tokenizer.pad_token_id),
            require_grad=False,
            compute_dtype=compute_dtype,
        )
        teacher_seconds = perf_counter() - teacher_started

        student_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        student_logprobs: list[Any] = []
        loss_total = 0.0
        weight_sum = 0.0
        weight_count = 0
        weight_min = math.inf
        weight_max = -math.inf
        delta_sum = 0.0
        delta_square_sum = 0.0
        delta_count = 0
        micro_batch_size = int(training["student_micro_batch_size"])
        for start in range(0, len(trajectories), micro_batch_size):
            micro_trajectories = trajectories[start : start + micro_batch_size]
            current_student_logprobs = _sampled_logprobs(
                student,
                micro_trajectories,
                micro_batch_size=len(micro_trajectories),
                torch=torch,
                device=device,
                pad_token_id=int(student_tokenizer.pad_token_id),
                require_grad=True,
                compute_dtype=compute_dtype,
            )
            current_teacher_logprobs = teacher_logprobs[start : start + len(micro_trajectories)]
            flat_student = torch.cat(current_student_logprobs)
            flat_teacher = torch.cat(
                [values.to(device=device) for values in current_teacher_logprobs]
            )
            mask = torch.ones_like(flat_student, dtype=torch.float32)
            loss, weights = sampled_token_k2_torch(
                flat_student,
                flat_teacher,
                alpha=alpha,
                mask=mask,
                denominator=torch.tensor(
                    total_response_tokens,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError(f"Non-finite K2 loss at step {step}")
            loss.backward()
            loss_total += float(loss.detach().item())
            detached_weights = weights.detach().float()
            if not bool(torch.isfinite(detached_weights).all().item()):
                raise FloatingPointError(f"Non-finite SuRe weights at step {step}")
            weight_sum += float(detached_weights.sum().item())
            weight_count += int(detached_weights.numel())
            weight_min = min(weight_min, float(detached_weights.min().item()))
            weight_max = max(weight_max, float(detached_weights.max().item()))
            delta = (flat_teacher - flat_student.detach()).float()
            delta_sum += float(delta.sum().item())
            delta_square_sum += float(delta.square().sum().item())
            delta_count += int(delta.numel())
            student_logprobs.extend(value.detach().cpu() for value in current_student_logprobs)

        expected_weight_min = 1.0
        expected_weight_max = 1.0 + alpha
        if weight_min < expected_weight_min - 1e-5 or weight_max > expected_weight_max + 1e-5:
            raise FloatingPointError(
                f"SuRe weights left [{expected_weight_min}, {expected_weight_max}] at step {step}"
            )

        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            float(training.get("max_grad_norm", 1.0)),
        )
        gradient_norm = float(gradient_norm_tensor.detach().item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        student_seconds = perf_counter() - student_started

        rows = _step_records(
            trajectories,
            teacher_logprobs,
            student_logprobs,
            verifier=verifier,
            step=step,
            policy_hash=policy_hash,
        )
        step_path = step_dir / f"step_{step:06d}.parquet"
        if step_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing committed step artifact: {step_path}"
            )
        write_records(step_path, rows)
        truncated = sum(bool(row["truncated"]) for row in rows)
        verifier_score = sum(float(row["verifier_score"]) for row in rows) / len(rows)
        delta_mean = delta_sum / max(1, delta_count)
        delta_variance = max(0.0, delta_square_sum / max(1, delta_count) - delta_mean**2)
        next_policy_hash = stable_hash(
            {
                "parent": policy_hash,
                "step": step,
                "step_artifact": file_sha256(step_path),
            },
            length=24,
        )
        step_metrics = {
            "step": step,
            "run_id": run_id,
            "hardware_fingerprint": hardware_fingerprint,
            "data_cursor_start": data_cursor,
            "records": len(rows),
            "response_tokens": total_response_tokens,
            "mean_response_tokens": total_response_tokens / len(rows),
            "truncated_records": truncated,
            "truncation_rate": truncated / len(rows),
            "verifier_score": verifier_score,
            "loss": loss_total,
            "gradient_norm": gradient_norm,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "delta_mean": delta_mean,
            "delta_std": math.sqrt(delta_variance),
            "sure_weight_mean": weight_sum / max(1, weight_count),
            "sure_weight_min": weight_min,
            "sure_weight_max": weight_max,
            "rollout_seconds": rollout_seconds,
            "teacher_seconds": teacher_seconds,
            "student_seconds": student_seconds,
            "step_seconds": perf_counter() - step_started,
            "policy_hash": policy_hash,
            "next_policy_hash": next_policy_hash,
            "step_artifact": str(step_path),
        }
        write_json(telemetry_dir / f"step_{step:06d}.json", step_metrics)
        history.append(step_metrics)
        global_step = step
        data_cursor += global_batch_size
        policy_hash = next_policy_hash

        state = {
            "run_id": run_id,
            "input_checksum": input_checksum,
            "global_step": global_step,
            "data_cursor": data_cursor,
            "policy_hash": policy_hash,
            "history": history,
        }
        if global_step % state_save_steps == 0 or global_step == max_steps:
            _save_training_state(
                rolling_dir,
                model=student,
                tokenizer=student_tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                state=state,
            )
        if global_step in milestone_steps and global_step < max_steps:
            _save_model_checkpoint(
                student,
                student_tokenizer,
                checkpoints_dir / f"step_{global_step:06d}",
            )

    completed = global_step == max_steps
    if not completed and global_step % state_save_steps != 0:
        _save_training_state(
            rolling_dir,
            model=student,
            tokenizer=student_tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            state={
                "run_id": run_id,
                "input_checksum": input_checksum,
                "global_step": global_step,
                "data_cursor": data_cursor,
                "policy_hash": policy_hash,
                "history": history,
            },
        )
    checkpoint_dir = output_dir / "final" if completed else rolling_dir / "model"
    if completed:
        _save_model_checkpoint(student, student_tokenizer, checkpoint_dir)
    invocation_history = history[invocation_start_step:]
    write_json(
        output_dir / "training_summary.json",
        {
            "status": "completed" if completed else "paused",
            "config_hash": config_hash(config),
            "method": training["method"],
            "parameter_update_mode": "full_parameter",
            "master_parameter_dtype": str(training.get("master_parameter_dtype", "float32")),
            "compute_dtype": str(training.get("dtype", "bfloat16")),
            "global_step": global_step,
            "records": data_cursor,
            "response_tokens": sum(int(item["response_tokens"]) for item in history),
            "last_invocation_records": sum(int(item["records"]) for item in invocation_history),
            "last_invocation_response_tokens": sum(
                int(item["response_tokens"]) for item in invocation_history
            ),
            "final_policy_hash": policy_hash,
            "student_tokenizer_fingerprint": student_fingerprint,
            "teacher_tokenizer_fingerprint": teacher_fingerprint,
            "hardware": hardware,
            "hardware_fingerprint": hardware_fingerprint,
            "model_identity": {
                "student_name": student_config["name"],
                "student_revision": student_config["revision"],
                "student_tokenizer_revision": student_config.get(
                    "tokenizer_revision", student_config["revision"]
                ),
                "teacher_name": teacher_config["name"],
                "teacher_revision": teacher_config["revision"],
                "teacher_tokenizer_revision": teacher_config.get(
                    "tokenizer_revision", teacher_config["revision"]
                ),
            },
            "initial_student_checkpoint": (
                str(initial_student_value) if initial_student_value else None
            ),
            "input_path": str(input_path),
            "input_checksum": input_checksum,
            "history": history,
            "seed": seed,
            "run_id": run_id,
            "generation_protocol": {
                "temperature": float(training.get("temperature", 1.0)),
                "top_p": float(training.get("top_p", 1.0)),
                "top_k": int(training.get("top_k", 0)),
                "max_new_tokens": int(training["max_response_tokens"]),
                "max_model_length": int(training["max_model_length"]),
                "enable_thinking": bool(training.get("enable_thinking", False)),
                "thinking_marker": training.get("thinking_marker"),
                "chat_template": "tokenizer_default_if_available",
            },
        },
    )
    return checkpoint_dir
