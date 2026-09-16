from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opd.exceptions import DependencyError
from opd.prompts import render_user_prompt
from opd.schemas import PromptRecord, TrainingRecord
from opd.tableio import read_json, read_records, write_json
from opd.tokenizers import tokenizer_fingerprint
from opd.training.losses import sparse_kl_torch
from opd.training.masking import build_response_labels, response_logit_bounds
from opd.verifier.math import MathVerifier


def _require_gpu_dependencies() -> dict[str, Any]:
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise DependencyError("HF training requires `pip install -e .[gpu]`") from exc
    return {
        "torch": torch,
        "Accelerator": Accelerator,
        "set_seed": set_seed,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup,
    }


class _ListDataset:
    def __init__(self, records: list[Any]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        return self.records[index]


@dataclass
class EarlyStopState:
    best_score: float = -math.inf
    stale_evaluations: int = 0
    best_step: int = 0

    def update(self, score: float, *, step: int, min_delta: float) -> bool:
        if score >= self.best_score + min_delta:
            self.best_score = score
            self.best_step = step
            self.stale_evaluations = 0
            return True
        self.stale_evaluations += 1
        return False


def _load_model(
    config: dict[str, Any], dependencies: dict[str, Any], accelerator: Any
) -> tuple[Any, Any]:
    torch = dependencies["torch"]
    training = config["training"]
    model_config = config["models"]["student"]
    model_path = Path(model_config["name"])
    dtype = getattr(torch, training.get("dtype", "bfloat16"))
    quantization = None
    if training.get("qlora", True):
        quantization = dependencies["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer_arguments: dict[str, Any] = {"trust_remote_code": False}
    if not model_path.exists():
        tokenizer_arguments["revision"] = model_config.get(
            "tokenizer_revision", model_config["revision"]
        )
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        model_config["name"], **tokenizer_arguments
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_arguments: dict[str, Any] = {
        "torch_dtype": dtype,
        "quantization_config": quantization,
        "device_map": {"": accelerator.local_process_index} if quantization else None,
        "trust_remote_code": False,
    }
    if not model_path.exists():
        model_arguments["revision"] = model_config["revision"]
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        model_config["name"], **model_arguments
    )
    if training.get("qlora", True):
        model = dependencies["prepare_model_for_kbit_training"](model)
    lora = training["lora"]
    peft_config = dependencies["LoraConfig"](
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = dependencies["get_peft_model"](model, peft_config)
    if training.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def _sft_collator(
    tokenizer: Any, max_length: int
) -> Callable[[list[PromptRecord]], dict[str, Any]]:
    torch = __import__("torch")

    def collate(records: list[PromptRecord]) -> dict[str, Any]:
        encoded: list[tuple[list[int], list[int]]] = []
        for record in records:
            prompt_text = render_user_prompt(tokenizer, record.problem)
            target = record.reference_solution or record.reference_answer
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            response_ids = tokenizer.encode(target, add_special_tokens=False)
            input_ids, labels = build_response_labels(
                prompt_ids,
                response_ids,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_length,
            )
            encoded.append((input_ids, labels))
        max_size = max(len(item[0]) for item in encoded)
        input_batch = []
        label_batch = []
        attention_batch = []
        for input_ids, labels in encoded:
            padding = max_size - len(input_ids)
            input_batch.append(input_ids + [tokenizer.pad_token_id] * padding)
            label_batch.append(labels + [-100] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
        }

    return collate


def _opd_collator(tokenizer: Any) -> Callable[[list[TrainingRecord]], dict[str, Any]]:
    torch = __import__("torch")

    def collate(records: list[TrainingRecord]) -> dict[str, Any]:
        max_size = max(len(record.input_ids) for record in records)
        input_batch = []
        attention_batch = []
        for record in records:
            padding = max_size - len(record.input_ids)
            input_batch.append(record.input_ids + [tokenizer.pad_token_id] * padding)
            attention_batch.append([1] * len(record.input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "records": records,
        }

    return collate


def _generate_validation_response(
    model: Any, tokenizer: Any, record: PromptRecord, *, device: Any, max_new_tokens: int
) -> str:
    prompt = render_user_prompt(tokenizer, record.problem)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=True))


def _evaluate_math_validation(
    model: Any,
    tokenizer: Any,
    validation: list[PromptRecord],
    *,
    device: Any,
    max_samples: int,
    max_new_tokens: int,
) -> tuple[float, float]:
    torch = __import__("torch")
    verifier = MathVerifier()
    model.eval()
    correct = 0.0
    formatted = 0
    total = min(max_samples, len(validation))
    with torch.inference_mode():
        for record in validation[:total]:
            response = _generate_validation_response(
                model,
                tokenizer,
                record,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            result = verifier.verify(
                rollout_id=f"validation_{record.sample_id}",
                sample_id=record.sample_id,
                response=response,
                reference_answer=record.reference_answer,
            )
            correct += result.score
            formatted += result.extracted_answer is not None
    model.train()
    return correct / max(1, total), formatted / max(1, total)


def _instruction_constraints_pass(response: str, metadata: dict[str, Any]) -> bool:
    normalized = response.strip()
    case_sensitive = bool(metadata.get("case_sensitive", False))
    inspected = normalized if case_sensitive else normalized.lower()

    def normalize(value: Any) -> str:
        text = str(value)
        return text if case_sensitive else text.lower()

    exact_text = metadata.get("exact_text")
    if exact_text is not None and inspected != normalize(exact_text):
        return False
    if any(normalize(value) not in inspected for value in metadata.get("required_substrings", [])):
        return False
    if any(normalize(value) in inspected for value in metadata.get("forbidden_substrings", [])):
        return False
    starts_with = metadata.get("starts_with")
    if starts_with is not None and not inspected.startswith(normalize(starts_with)):
        return False
    ends_with = metadata.get("ends_with")
    if ends_with is not None and not inspected.endswith(normalize(ends_with)):
        return False
    word_count = len(normalized.split())
    if "min_words" in metadata and word_count < int(metadata["min_words"]):
        return False
    if "max_words" in metadata and word_count > int(metadata["max_words"]):
        return False
    return True


def _evaluate_instruction_validation(
    model: Any,
    tokenizer: Any,
    validation: list[PromptRecord],
    *,
    device: Any,
    max_samples: int,
    max_new_tokens: int,
) -> float:
    torch = __import__("torch")
    model.eval()
    total = min(max_samples, len(validation))
    passed = 0
    with torch.inference_mode():
        for record in validation[:total]:
            response = _generate_validation_response(
                model,
                tokenizer,
                record,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            passed += _instruction_constraints_pass(response, record.metadata)
    model.train()
    return passed / max(1, total)


def _save_checkpoint(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    global_step: int,
    early_state: EarlyStopState,
    resume_epoch: int,
    resume_batch_in_epoch: int,
    history: list[dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        write_json(
            output_dir / "checkpoint_state.json",
            {
                "global_step": global_step,
                "best_score": early_state.best_score,
                "best_step": early_state.best_step,
                "stale_evaluations": early_state.stale_evaluations,
                "resume_epoch": resume_epoch,
                "resume_batch_in_epoch": resume_batch_in_epoch,
                "history": history,
            },
        )
    accelerator.save_state(output_dir / "accelerate_state")


def train_hf(config: dict[str, Any]) -> Path:
    dependencies = _require_gpu_dependencies()
    torch = dependencies["torch"]
    training = config["training"]
    dependencies["set_seed"](int(config["project"]["seed"]), device_specific=False)
    accelerator = dependencies["Accelerator"](
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=training.get("mixed_precision", "bf16"),
    )
    model, tokenizer = _load_model(config, dependencies, accelerator)
    method = training["method"]
    input_rows = read_records(training["input_path"])
    records: list[Any]
    collate: Any
    if method == "sft":
        records = [PromptRecord.model_validate(row) for row in input_rows]
        collate = _sft_collator(tokenizer, int(training.get("max_length", 2048)))
    else:
        records = [TrainingRecord.model_validate(row) for row in input_rows]
        student_tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
        mismatched = [
            record.rollout_id
            for record in records
            if record.tokenizer_fingerprint != student_tokenizer_fingerprint
        ]
        if mismatched:
            raise ValueError(
                "Training view tokenizer does not match Student vocabulary; "
                f"first mismatched rollout: {mismatched[0]}"
            )
        collate = _opd_collator(tokenizer)
    max_records_value = training.get("max_records")
    max_records = int(max_records_value) if max_records_value is not None else None
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("training.max_records must be positive when configured")
        records = records[:max_records]
    random.Random(int(config["project"]["seed"])).shuffle(records)
    dataset = _ListDataset(records)
    dataloader = dependencies["DataLoader"](
        dataset,
        batch_size=int(training.get("batch_size", 1)),
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    max_steps = int(training["max_steps"])
    warmup_steps = int(training.get("warmup_steps", max(1, max_steps // 20)))
    scheduler = dependencies["get_cosine_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    validation_path = training.get("validation_path")
    validation = (
        [PromptRecord.model_validate(row) for row in read_records(validation_path)]
        if validation_path
        else []
    )
    early_config = training.get("early_stopping", {})
    instruction_validation_path = early_config.get("instruction_validation_path")
    instruction_validation = (
        [PromptRecord.model_validate(row) for row in read_records(instruction_validation_path)]
        if instruction_validation_path
        else []
    )
    eval_steps = int(early_config.get("eval_steps", 200))
    patience = int(early_config.get("patience", 3))
    min_delta = float(early_config.get("min_delta", 0.005))
    early_state = EarlyStopState()
    output_dir = Path(training["output_dir"])
    best_dir = output_dir / "best"
    latest_dir = output_dir / "latest"
    history: list[dict[str, float | int]] = []
    global_step = 0
    resume_epoch = 0
    resume_batch_in_epoch = 0
    resume_from = training.get("resume_from_checkpoint")
    if resume_from:
        resume_dir = Path(resume_from)
        state_path = resume_dir / "checkpoint_state.json"
        accelerate_state = resume_dir / "accelerate_state"
        if not state_path.exists() or not accelerate_state.exists():
            raise FileNotFoundError(f"Incomplete resume checkpoint: {resume_dir}")
        accelerator.load_state(accelerate_state)
        saved_state = read_json(state_path)
        global_step = int(saved_state["global_step"])
        early_state = EarlyStopState(
            best_score=float(saved_state["best_score"]),
            stale_evaluations=int(saved_state["stale_evaluations"]),
            best_step=int(saved_state["best_step"]),
        )
        resume_epoch = int(saved_state.get("resume_epoch", 0))
        resume_batch_in_epoch = int(saved_state.get("resume_batch_in_epoch", 0))
        history = [dict(item) for item in saved_state.get("history", [])]
    model.train()

    stop_training = global_step >= max_steps
    max_epochs = int(early_config.get("max_epochs_per_round", 1))
    next_resume_epoch = resume_epoch
    next_resume_batch = resume_batch_in_epoch
    for epoch in range(resume_epoch, max_epochs):
        if stop_training:
            break
        epoch_dataloader = (
            accelerator.skip_first_batches(dataloader, resume_batch_in_epoch)
            if epoch == resume_epoch and resume_batch_in_epoch
            else dataloader
        )
        initial_batch_index = resume_batch_in_epoch if epoch == resume_epoch else 0
        for data_batch_index, batch in enumerate(epoch_dataloader, start=initial_batch_index):
            with accelerator.accumulate(model):
                if method == "sft":
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                else:
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    sample_losses = []
                    for sample_index, record in enumerate(batch["records"]):
                        token_count = len(record.response_token_ids)
                        if token_count == 0 or record.verifier_weight <= 0:
                            sample_losses.append(output.logits[sample_index].sum() * 0.0)
                            continue
                        start, end = response_logit_bounds(
                            response_start=record.response_start,
                            response_tokens=token_count,
                            input_length=len(record.input_ids),
                        )
                        sample_logits = output.logits[sample_index, start:end]
                        topk_ids = torch.tensor(
                            record.topk_token_ids,
                            device=sample_logits.device,
                            dtype=torch.long,
                        )
                        topk_logprobs = torch.tensor(
                            record.topk_logprobs,
                            device=sample_logits.device,
                            dtype=torch.float32,
                        )
                        tail_mass = torch.tensor(
                            record.tail_mass,
                            device=sample_logits.device,
                            dtype=torch.float32,
                        )
                        weights = torch.tensor(
                            record.confidence_weights,
                            device=sample_logits.device,
                            dtype=torch.float32,
                        )
                        sample_losses.append(
                            sparse_kl_torch(
                                sample_logits,
                                topk_ids,
                                topk_logprobs,
                                tail_mass,
                                weights,
                            )
                            * float(record.verifier_weight)
                        )
                    loss = torch.stack(sample_losses).mean()
                if not bool(torch.isfinite(loss).all().item()):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
                accelerator.backward(loss)
                gradient_norm: float | None = None
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(training.get("max_grad_norm", 1.0))
                    )
                    gradient_norm = float(norm.detach().item())
                    if not math.isfinite(gradient_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at step {global_step + 1}"
                        )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                next_resume_epoch = epoch
                next_resume_batch = data_batch_index + 1
                if next_resume_batch >= len(dataloader):
                    next_resume_epoch = epoch + 1
                    next_resume_batch = 0
                history_entry: dict[str, float | int] = {
                    "step": global_step,
                    "loss": float(loss.detach().item()),
                }
                if gradient_norm is not None:
                    history_entry["gradient_norm"] = gradient_norm
                history.append(history_entry)
                if global_step % int(training.get("save_steps", eval_steps)) == 0:
                    _save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        latest_dir,
                        global_step=global_step,
                        early_state=early_state,
                        resume_epoch=next_resume_epoch,
                        resume_batch_in_epoch=next_resume_batch,
                        history=history,
                    )
                if validation and global_step % eval_steps == 0:
                    math_accuracy, format_pass_rate = _evaluate_math_validation(
                        accelerator.unwrap_model(model),
                        tokenizer,
                        validation,
                        device=accelerator.device,
                        max_samples=int(early_config.get("max_validation_samples", 200)),
                        max_new_tokens=int(early_config.get("max_new_tokens", 512)),
                    )
                    instruction_score = (
                        _evaluate_instruction_validation(
                            accelerator.unwrap_model(model),
                            tokenizer,
                            instruction_validation,
                            device=accelerator.device,
                            max_samples=int(
                                early_config.get("max_instruction_validation_samples", 32)
                            ),
                            max_new_tokens=int(early_config.get("instruction_max_new_tokens", 128)),
                        )
                        if instruction_validation
                        else 0.0
                    )
                    weights = early_config.get(
                        "weights",
                        {
                            "math_accuracy": 0.8,
                            "format_pass_rate": 0.1,
                            "instruction": 0.1,
                        },
                    )
                    score = (
                        float(weights["math_accuracy"]) * math_accuracy
                        + float(weights["format_pass_rate"]) * format_pass_rate
                        + float(weights["instruction"]) * instruction_score
                    )
                    history[-1]["validation_accuracy"] = math_accuracy
                    history[-1]["format_pass_rate"] = format_pass_rate
                    history[-1]["instruction_regression_score"] = instruction_score
                    history[-1]["composite_score"] = score
                    improved = early_state.update(score, step=global_step, min_delta=min_delta)
                    if improved:
                        _save_checkpoint(
                            accelerator,
                            model,
                            tokenizer,
                            best_dir,
                            global_step=global_step,
                            early_state=early_state,
                            resume_epoch=next_resume_epoch,
                            resume_batch_in_epoch=next_resume_batch,
                            history=history,
                        )
                    if early_state.stale_evaluations >= patience:
                        stop_training = True
                        break
                if global_step >= max_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    _save_checkpoint(
        accelerator,
        model,
        tokenizer,
        latest_dir,
        global_step=global_step,
        early_state=early_state,
        resume_epoch=next_resume_epoch,
        resume_batch_in_epoch=next_resume_batch,
        history=history,
    )
    if accelerator.is_main_process:
        write_json(
            output_dir / "training_summary.json",
            {
                "method": method,
                "global_step": global_step,
                "best_validation_score": early_state.best_score,
                "best_step": early_state.best_step,
                "history": history,
                "seed": int(config["project"]["seed"]),
            },
        )
    accelerator.wait_for_everyone()
    return best_dir if best_dir.exists() else latest_dir
