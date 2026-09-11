import pytest

from evalbench.graders import grade_case, passed_all_primary
from evalbench.models import EvaluationCase, Judgment


def make_case(grader: dict[str, object], expected: object = "expected") -> EvaluationCase:
    return EvaluationCase(
        id="case",
        category="test",
        prompt="prompt",
        expected=expected,
        graders=[grader],
    )


@pytest.mark.parametrize(
    ("grader", "expected", "output", "should_pass"),
    [
        ({"type": "exact"}, " Hello  world ", "hello world", True),
        ({"type": "exact", "case_sensitive": True}, "Hello", "hello", False),
        ({"type": "numeric", "tolerance": 0.1}, "10", "The answer is 10.09", True),
        ({"type": "numeric", "tolerance": 0.1}, "10", "The answer is 10.11", False),
        ({"type": "numeric"}, "10", "No number here", False),
        ({"type": "contains_all", "values": ["red", "BLUE"]}, None, "Red and blue", True),
        ({"type": "contains_all", "values": ["red", "blue"]}, None, "red only", False),
        ({"type": "length", "unit": "words", "minimum": 2, "maximum": 2}, None, "two words", True),
        ({"type": "length", "unit": "lines", "maximum": 1}, None, "one\ntwo", False),
        ({"type": "refusal", "expected": True}, None, "I cannot help with that.", True),
        ({"type": "refusal", "expected": False}, None, "Use a strong password.", True),
    ],
)
def test_rule_boundaries(
    grader: dict[str, object], expected: object, output: str, should_pass: bool
) -> None:
    assert grade_case(make_case(grader, expected), output)[0].passed is should_pass


@pytest.mark.parametrize(
    ("output", "should_pass"),
    [
        ('{"email":"ana@example.com"}', True),
        ('{"email":"not-an-email"}', False),
        ('{"email":"ana@example.com","extra":1}', False),
        ("not json", False),
    ],
)
def test_json_schema_grader(output: str, should_pass: bool) -> None:
    grader = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "required": ["email"],
            "additionalProperties": False,
            "properties": {"email": {"type": "string", "format": "email"}},
        },
    }
    assert grade_case(make_case(grader, None), output)[0].passed is should_pass


def test_case_runs_each_deterministic_grader_and_defers_judge() -> None:
    case = EvaluationCase(
        id="mixed",
        category="test",
        prompt="prompt",
        expected="yes",
        graders=[
            {"type": "exact"},
            {"type": "contains_all", "values": ["yes"]},
            {"type": "judge"},
        ],
    )
    assert [item.grader_type for item in grade_case(case, "yes")] == ["exact", "contains_all"]


def test_all_primary_judgments_must_pass() -> None:
    passed = Judgment(grader_type="exact", passed=True, score=1, rationale="matched")
    failed = Judgment(grader_type="length", passed=False, score=0, rationale="too long")
    advisory = failed.model_copy(update={"primary": False})

    assert passed_all_primary([passed, advisory])
    assert not passed_all_primary([passed, failed])
    assert not passed_all_primary([])
