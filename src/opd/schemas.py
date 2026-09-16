from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    OOM = "oom"
    TRUNCATED = "truncated"


class VerificationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class PromptRecord(StrictModel):
    sample_id: str
    problem: str
    reference_answer: str
    reference_solution: str | None = None
    subject: str = "unknown"
    difficulty: str = "unknown"
    source: str
    source_dataset: str
    source_revision: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutRecord(StrictModel):
    rollout_id: str
    sample_id: str
    round_id: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    prompt: str
    response: str
    student_model: str
    student_revision: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    sampling_config_hash: str
    seed: int
    prompt_tokens: int = Field(ge=0)
    response_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    status: RecordStatus = RecordStatus.SUCCESS
    error: str | None = None


class VerificationRecord(StrictModel):
    rollout_id: str
    sample_id: str
    verifier_name: str
    verifier_version: str
    status: VerificationStatus
    score: float = Field(ge=0.0, le=1.0)
    reference_answer: str
    extracted_answer: str | None = None
    error_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TeacherAnnotationRecord(StrictModel):
    rollout_id: str
    sample_id: str
    teacher_model: str
    teacher_revision: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    input_ids: list[int]
    response_start: int = Field(ge=1)
    response_token_ids: list[int]
    topk_token_ids: list[list[int]]
    topk_logprobs: list[list[float]]
    tail_mass: list[float]
    token_entropy: list[float]
    teacher_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    status: RecordStatus = RecordStatus.SUCCESS
    error: str | None = None

    @model_validator(mode="after")
    def validate_token_shapes(self) -> TeacherAnnotationRecord:
        expected = len(self.response_token_ids)
        lengths = (
            len(self.topk_token_ids),
            len(self.topk_logprobs),
            len(self.tail_mass),
            len(self.token_entropy),
        )
        if any(length != expected for length in lengths):
            raise ValueError("Teacher token-level fields must match response_token_ids length")
        for ids, logprobs in zip(self.topk_token_ids, self.topk_logprobs, strict=True):
            if len(ids) != len(logprobs):
                raise ValueError("Each top-k id row must match its logprob row")
        return self


class TrainingRecord(StrictModel):
    rollout_id: str
    sample_id: str
    prompt: str
    response: str
    tokenizer_fingerprint: str
    input_ids: list[int]
    response_start: int
    response_token_ids: list[int]
    topk_token_ids: list[list[int]]
    topk_logprobs: list[list[float]]
    tail_mass: list[float]
    confidence_weights: list[float]
    verifier_status: VerificationStatus
    verifier_weight: float = Field(ge=0.0)
    method: str


class ArtifactManifest(StrictModel):
    artifact_id: str
    artifact_type: str
    stage: str
    created_at: datetime = Field(default_factory=utc_now)
    git_commit: str = "unknown"
    config_hash: str
    upstream_artifact_ids: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
    record_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobMetrics(StrictModel):
    job_type: str
    started_at: datetime
    finished_at: datetime
    wall_time_seconds: float = Field(ge=0.0)
    gpu_type: str = "cpu"
    gpu_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    response_tokens: int = Field(default=0, ge=0)
    teacher_tokens: int = Field(default=0, ge=0)
    records: int = Field(default=0, ge=0)
    estimated_gpu_hours: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
