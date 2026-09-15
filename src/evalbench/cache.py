from __future__ import annotations

import hashlib
import json

from evalbench.judge import JudgeRubric
from evalbench.models import EvaluationCase


def generation_request_hash(
    *, model: str, case: EvaluationCase, provider_schema_version: str
) -> str:
    """Hash every input that can change a provider's generated response."""
    request = {
        "max_tokens": case.generation.max_tokens,
        "model": model,
        "prompt": case.prompt,
        "provider_schema_version": provider_schema_version,
        "system": case.system,
        "temperature": case.generation.temperature,
    }
    canonical = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def judge_request_hash(
    *,
    rubric: JudgeRubric,
    judge_model: str,
    case: EvaluationCase,
    answer: str,
    provider_schema_version: str,
) -> str:
    """Hash every input that can change an ordinal judge response."""
    rubric_content = json.dumps(
        rubric.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    request = {
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "case_id": case.id,
        "evaluation_prompt": case.prompt,
        "expected_reference": case.expected,
        "judge_model": judge_model,
        "provider_schema_version": provider_schema_version,
        "rubric_content_sha256": hashlib.sha256(rubric_content).hexdigest(),
        "rubric_id": rubric.id,
        "rubric_version": rubric.version,
    }
    canonical = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
