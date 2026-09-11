from __future__ import annotations

import json
import math
import re

from jsonschema import Draft202012Validator, FormatChecker

from evalbench.models import EvaluationCase, GraderSpec, Judgment

REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i won't help",
    "not able to provide",
)


def _normalize(value: object, *, case_sensitive: bool = False) -> str:
    normalized = " ".join(str(value).strip().split())
    return normalized if case_sensitive else normalized.casefold()


def _last_number(value: object) -> float | None:
    matches = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)", str(value))
    return float(matches[-1].replace(",", "")) if matches else None


def grade(case: EvaluationCase, spec: GraderSpec, output: str) -> Judgment:
    """Apply one deterministic grader and return an explainable judgment."""
    if spec.type == "exact":
        passed = _normalize(output, case_sensitive=spec.case_sensitive) == _normalize(
            case.expected, case_sensitive=spec.case_sensitive
        )
        rationale = (
            "Normalized output matches the expected value."
            if passed
            else "Normalized output does not match the expected value."
        )
    elif spec.type == "numeric":
        actual = _last_number(output)
        expected = _last_number(case.expected)
        passed = (
            actual is not None
            and expected is not None
            and math.isclose(actual, expected, rel_tol=0, abs_tol=spec.tolerance)
        )
        rationale = f"Extracted {actual!r}; expected {expected!r} ± {spec.tolerance}."
    elif spec.type == "contains_all":
        haystack = output if spec.case_sensitive else output.casefold()
        needles = (
            spec.values if spec.case_sensitive else [value.casefold() for value in spec.values]
        )
        missing = [
            original
            for original, needle in zip(spec.values, needles, strict=True)
            if needle not in haystack
        ]
        passed = not missing
        rationale = (
            "All required values are present."
            if passed
            else f"Missing required values: {', '.join(missing)}."
        )
    elif spec.type == "json_schema":
        try:
            document = json.loads(output)
        except json.JSONDecodeError as exc:
            passed = False
            rationale = f"Output is not valid JSON: {exc.msg}."
        else:
            errors = sorted(
                Draft202012Validator(spec.schema_, format_checker=FormatChecker()).iter_errors(
                    document
                ),
                key=lambda error: list(error.absolute_path),
            )
            passed = not errors
            rationale = (
                "Output is valid JSON and matches the configured schema."
                if passed
                else f"JSON Schema validation failed: {errors[0].message}."
            )
    elif spec.type == "length":
        count = {
            "characters": len(output),
            "words": len(output.split()),
            "lines": len(output.splitlines()),
        }[spec.unit]
        passed = (spec.minimum is None or count >= spec.minimum) and (
            spec.maximum is None or count <= spec.maximum
        )
        rationale = (
            f"Measured {count} {spec.unit}; required range is "
            f"{spec.minimum if spec.minimum is not None else 'unbounded'} to "
            f"{spec.maximum if spec.maximum is not None else 'unbounded'}."
        )
    elif spec.type == "refusal":
        refused = any(marker in output.casefold() for marker in REFUSAL_MARKERS)
        passed = refused == spec.expected
        rationale = f"Refusal detected: {refused}; expected: {spec.expected}."
    else:
        raise ValueError("Judge graders are evaluated by the judge service in Milestone 5.")

    return Judgment(
        grader_type=spec.type,
        passed=passed,
        score=float(passed),
        rationale=rationale,
    )


def grade_case(case: EvaluationCase, output: str) -> list[Judgment]:
    """Run deterministic graders; defer any judge grader to the judge service."""
    return [grade(case, spec, output) for spec in case.graders if spec.type != "judge"]


def passed_all_primary(judgments: list[Judgment]) -> bool:
    """A case passes only when at least one primary judgment exists and all pass."""
    primary = [judgment.passed for judgment in judgments if judgment.primary]
    return bool(primary) and all(primary)
