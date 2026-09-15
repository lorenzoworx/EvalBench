from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalbench.models import EvaluationCase, JudgeGrader, Judgment


class RubricValidationError(ValueError):
    """Raised when a versioned judge rubric cannot be loaded safely."""


class JudgeEvaluationError(ValueError):
    """Raised when a judge response does not satisfy the scoring contract."""


class RubricLevel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: Literal[0, 1, 2]
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class JudgeRubric(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*_v[1-9][0-9]*$")
    version: str = Field(min_length=1)
    status: Literal["calibration", "frozen"]
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    instructions: list[str] = Field(min_length=1)
    criteria: list[str] = Field(min_length=1)
    scores: list[RubricLevel]

    @model_validator(mode="after")
    def complete_ordinal_scale(self) -> JudgeRubric:
        if sorted(level.score for level in self.scores) != [0, 1, 2]:
            raise ValueError(
                "rubric scores must define each ordinal level 0, 1, and 2 exactly once"
            )
        return self

    def system_prompt(self) -> str:
        levels = "\n".join(
            f"- {level.score} ({level.label}): {level.description}" for level in self.scores
        )
        instructions = "\n".join(f"- {item}" for item in self.instructions)
        criteria = "\n".join(f"- {item}" for item in self.criteria)
        return (
            f"You are an impartial evaluation judge using {self.id} ({self.version}).\n"
            f"{self.description}\n\nInstructions:\n{instructions}\n\n"
            f"Criteria:\n{criteria}\n\nOrdinal score levels:\n{levels}\n\n"
            "Return only the schema-constrained score and rationale."
        )


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: Literal[0, 1, 2]
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def rationale_cannot_be_blank(cls, value: str) -> str:
        rationale = value.strip()
        if not rationale:
            raise ValueError("rationale cannot be blank")
        return rationale


class StructuredChatProvider(Protocol):
    async def structured_chat(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...


def load_rubric(path: str | Path | None = None) -> JudgeRubric:
    """Load the bundled rubric or an explicitly supplied rubric for validation."""
    source = (
        Path(path)
        if path is not None
        else files("evalbench.rubrics").joinpath("overall_quality_v1.yaml")
    )
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RubricValidationError(f"Cannot read judge rubric {source}: {exc}") from exc
    try:
        return JudgeRubric.model_validate(raw)
    except ValidationError as exc:
        raise RubricValidationError(f"Invalid judge rubric {source}:\n{exc}") from exc


class JudgeEvaluator:
    """Apply one versioned rubric through a schema-constrained local judge."""

    def __init__(self, provider: StructuredChatProvider, rubric: JudgeRubric | None = None) -> None:
        self.provider = provider
        self.rubric = rubric or load_rubric()

    async def evaluate(
        self,
        *,
        model: str,
        case: EvaluationCase,
        answer: str,
        grader: JudgeGrader,
    ) -> Judgment:
        if grader.rubric != self.rubric.id:
            raise JudgeEvaluationError(
                f"Judge grader requests rubric {grader.rubric!r}, but {self.rubric.id!r} is loaded."
            )
        messages = [
            {"role": "system", "content": self.rubric.system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "evaluation_prompt": case.prompt,
                        "expected_reference": case.expected,
                        "candidate_answer": answer,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        payload = await self.provider.structured_chat(
            model,
            messages=messages,
            schema=JudgeResponse.model_json_schema(),
        )
        try:
            response = JudgeResponse.model_validate(payload)
        except ValidationError as exc:
            raise JudgeEvaluationError(f"Judge returned an invalid score response: {exc}") from exc
        return Judgment(
            grader_type="judge",
            passed=response.score >= grader.minimum_score,
            score=float(response.score),
            rationale=response.rationale,
        )
