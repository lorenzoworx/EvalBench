from pathlib import Path

import pytest

from evalbench.models import EvaluationCase, Generation, LengthGrader
from evalbench.suite import SuiteValidationError, load_suite


def test_core_suite_has_balanced_60_cases() -> None:
    suite = load_suite(Path(__file__).parents[1] / "suites/core.yaml")
    assert len(suite.cases) == 60
    counts = {
        category: sum(case.category == category for case in suite.cases)
        for category in {case.category for case in suite.cases}
    }
    assert set(counts.values()) == {15}


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nversion: '1'\ncases:\n"
        "- {id: same, category: x, prompt: x, expected: x, graders: [{type: exact}]}\n"
        "- {id: same, category: x, prompt: x, expected: x, graders: [{type: exact}]}\n"
    )
    with pytest.raises(SuiteValidationError, match="duplicate case IDs"):
        load_suite(path)


@pytest.mark.parametrize(
    "grader",
    [
        "{type: numeric, tolerance: -1}",
        "{type: contains_all, values: []}",
        "{type: length}",
        "{type: made_up}",
    ],
)
def test_invalid_graders_are_rejected(tmp_path: Path, grader: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nversion: '1'\ncases:\n- id: x\n  category: x\n"
        f"  prompt: x\n  expected: x\n  graders: [{grader}]\n"
    )
    with pytest.raises(SuiteValidationError):
        load_suite(path)


def test_length_range_is_validated() -> None:
    with pytest.raises(ValueError, match="minimum cannot exceed"):
        LengthGrader(type="length", minimum=3, maximum=2)


def test_blank_categories_and_missing_expected_are_rejected() -> None:
    with pytest.raises(ValueError, match="category cannot be blank"):
        EvaluationCase(id="x", category=" ", prompt="x", expected="x", graders=[{"type": "exact"}])
    with pytest.raises(ValueError, match="require expected"):
        EvaluationCase(id="x", category="x", prompt="x", graders=[{"type": "numeric"}])


def test_generation_exposes_derived_ollama_metrics() -> None:
    generation = Generation(
        output_tokens=20, total_duration_ns=2_000_000, eval_duration_ns=1_000_000_000, text="ok"
    )
    assert generation.latency_ms == 2
    assert generation.tokens_per_second == 20
    assert Generation(text="ok").latency_ms is None
    assert Generation(text="ok").tokens_per_second is None
