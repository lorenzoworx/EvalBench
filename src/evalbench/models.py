from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExactGrader(BaseModel):
    type: Literal["exact"]
    case_sensitive: bool = False


class NumericGrader(BaseModel):
    type: Literal["numeric"]
    tolerance: float = Field(default=0.0, ge=0)


class ContainsAllGrader(BaseModel):
    type: Literal["contains_all"]
    values: list[str] = Field(min_length=1)
    case_sensitive: bool = False


class JsonSchemaGrader(BaseModel):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class LengthGrader(BaseModel):
    type: Literal["length"]
    unit: Literal["characters", "words", "lines"] = "words"
    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self) -> LengthGrader:
        if self.minimum is None and self.maximum is None:
            raise ValueError("length grader needs minimum and/or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("length minimum cannot exceed maximum")
        return self


class RefusalGrader(BaseModel):
    type: Literal["refusal"]
    expected: bool = True


class JudgeGrader(BaseModel):
    type: Literal["judge"]
    rubric: str = "overall_quality_v1"
    minimum_score: int = Field(default=2, ge=0, le=2)


GraderSpec = Annotated[
    ExactGrader
    | NumericGrader
    | ContainsAllGrader
    | JsonSchemaGrader
    | LengthGrader
    | RefusalGrader
    | JudgeGrader,
    Field(discriminator="type"),
]


class GenerationSettings(BaseModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0, le=32768)


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    system: str | None = None
    expected: Any | None = None
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    graders: list[GraderSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def grader_requirements(self) -> EvaluationCase:
        if not self.category.strip():
            raise ValueError("category cannot be blank")
        if any(g.type in {"exact", "numeric"} for g in self.graders) and self.expected is None:
            raise ValueError("exact and numeric graders require expected")
        return self


class EvaluationSuite(BaseModel):
    name: str = Field(min_length=1)
    version: str
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> EvaluationSuite:
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case IDs: {', '.join(duplicates)}")
        return self


class Generation(BaseModel):
    text: str
    done_reason: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None
    eval_duration_ns: int | None = None
    cached: bool = False

    @property
    def latency_ms(self) -> float | None:
        return None if self.total_duration_ns is None else self.total_duration_ns / 1_000_000

    @property
    def tokens_per_second(self) -> float | None:
        if not self.output_tokens or not self.eval_duration_ns:
            return None
        return self.output_tokens / (self.eval_duration_ns / 1_000_000_000)


class Judgment(BaseModel):
    grader_type: str
    passed: bool
    score: float
    rationale: str
    primary: bool = True


class CaseResult(BaseModel):
    case_id: str
    category: str
    prompt: str
    expected: Any | None
    generation: Generation
    judgments: list[Judgment]
    passed: bool


class RunSummary(BaseModel):
    id: str
    suite_name: str
    suite_version: str
    model: str
    provider: str
    status: Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    completed_cases: int = 0
    total_cases: int = 0
    accuracy: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    error: str | None = None
