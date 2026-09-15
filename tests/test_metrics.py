import pytest

from evalbench.metrics import (
    bootstrap_accuracy_ci,
    compare_paired_results,
    exact_mcnemar_test,
    percentile,
    summarize_results,
)
from evalbench.models import CaseResult, Generation


def result(
    case_id: str,
    category: str,
    passed: bool,
    *,
    latency_ms: float | None = None,
    output_tokens: int | None = None,
    eval_seconds: float | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        category=category,
        prompt="prompt",
        expected=None,
        generation=Generation(
            text="answer",
            total_duration_ns=None if latency_ms is None else int(latency_ms * 1_000_000),
            output_tokens=output_tokens,
            eval_duration_ns=None if eval_seconds is None else int(eval_seconds * 1_000_000_000),
        ),
        judgments=[],
        passed=passed,
    )


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([], 0.5) is None
    with pytest.raises(ValueError, match="probability"):
        percentile([1], 1.1)


def test_bootstrap_is_seeded_and_bounded() -> None:
    outcomes = [True, False, True, False, True]
    first = bootstrap_accuracy_ci(outcomes, samples=1_000, seed=9)
    second = bootstrap_accuracy_ci(outcomes, samples=1_000, seed=9)

    assert first == second
    assert 0 <= first[0] <= first[1] <= 1
    assert bootstrap_accuracy_ci([True] * 5) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("outcomes", "options", "message"),
    [
        ([], {}, "one outcome"),
        ([True], {"samples": 0}, "samples"),
        ([True], {"confidence": 1.0}, "confidence"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(
    outcomes: list[bool], options: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_accuracy_ci(outcomes, **options)


def test_summary_reports_accuracy_categories_latency_and_throughput() -> None:
    results = [
        result("a", "reasoning", True, latency_ms=1, output_tokens=10, eval_seconds=1),
        result("b", "reasoning", False, latency_ms=2, output_tokens=20, eval_seconds=1),
        result("c", "safety", True, latency_ms=3),
        result("d", "safety", True, latency_ms=4),
    ]

    metrics = summarize_results(results)

    assert metrics.case_count == 4
    assert metrics.accuracy == 0.75
    assert metrics.category_accuracy == {"reasoning": 0.5, "safety": 1.0}
    assert metrics.latency_ms_p50 == 2.5
    assert metrics.latency_ms_p95 == pytest.approx(3.85)
    assert metrics.tokens_per_second == 15
    assert metrics.ci_low is not None and metrics.ci_high is not None


def test_summary_handles_no_results_or_timing_metrics() -> None:
    empty = summarize_results([])
    untimed = summarize_results([result("a", "test", True)])

    assert empty.model_dump() == {
        "case_count": 0,
        "accuracy": None,
        "ci_low": None,
        "ci_high": None,
        "category_accuracy": {},
        "latency_ms_p50": None,
        "latency_ms_p95": None,
        "tokens_per_second": None,
    }
    assert untimed.latency_ms_p50 is None
    assert untimed.latency_ms_p95 is None
    assert untimed.tokens_per_second is None


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected_counts", "expected_p_value"),
    [
        ([True, False], [True, False], (1, 0, 0, 1), 1.0),
        ([True, True], [False, False], (0, 2, 0, 0), 0.5),
        ([True] * 5, [False] * 5, (0, 5, 0, 0), 0.0625),
        ([True] * 10, [False] * 10, (0, 10, 0, 0), 0.001953125),
    ],
)
def test_exact_mcnemar_contingencies_and_two_sided_p_value(
    baseline: list[bool],
    candidate: list[bool],
    expected_counts: tuple[int, int, int, int],
    expected_p_value: float,
) -> None:
    test = exact_mcnemar_test(baseline, candidate)

    assert (
        test.both_passed,
        test.baseline_only_passed,
        test.candidate_only_passed,
        test.both_failed,
    ) == expected_counts
    assert test.exact_p_value == expected_p_value
    assert test.significant is (expected_p_value < 0.05)


def test_exact_mcnemar_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        exact_mcnemar_test([True], [])
    with pytest.raises(ValueError, match="alpha"):
        exact_mcnemar_test([], [], alpha=1)


def test_paired_comparison_reports_delta_regressions_and_improvements() -> None:
    baseline = [
        result("a", "test", True),
        result("b", "test", True),
        result("c", "test", False),
        result("d", "test", False),
    ]
    candidate = [
        result("d", "test", False),
        result("c", "test", True),
        result("b", "test", False),
        result("a", "test", True),
    ]

    comparison = compare_paired_results("base", "candidate", baseline, candidate)

    assert comparison.case_count == 4
    assert comparison.baseline_accuracy == 0.5
    assert comparison.candidate_accuracy == 0.5
    assert comparison.accuracy_delta == 0
    assert comparison.regressions == ["b"]
    assert comparison.improvements == ["c"]
    assert comparison.mcnemar.baseline_only_passed == 1
    assert comparison.mcnemar.candidate_only_passed == 1


def test_paired_comparison_requires_matching_unique_case_ids() -> None:
    one = result("one", "test", True)
    two = result("two", "test", False)

    with pytest.raises(ValueError, match="at least one"):
        compare_paired_results("base", "candidate", [], [])
    with pytest.raises(ValueError, match="unique case IDs"):
        compare_paired_results("base", "candidate", [one, one], [one, one])
    with pytest.raises(ValueError, match="missing from candidate: one"):
        compare_paired_results("base", "candidate", [one], [two])
