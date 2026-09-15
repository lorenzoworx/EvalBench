from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import median

from evalbench.models import CaseResult, McNemarResult, RunComparison, RunMetrics


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile for a finite sequence."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_accuracy_ci(
    outcomes: Sequence[bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a reproducible nonparametric bootstrap interval for accuracy."""
    if not outcomes:
        raise ValueError("at least one outcome is required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    size = len(outcomes)
    estimates = [sum(rng.choice(outcomes) for _ in range(size)) / size for _ in range(samples)]
    tail = (1 - confidence) / 2
    low = percentile(estimates, tail)
    high = percentile(estimates, 1 - tail)
    if low is None or high is None:
        raise AssertionError("positive bootstrap samples must produce percentiles")
    return low, high


def exact_mcnemar_test(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    """Run the two-sided exact McNemar test over paired binary outcomes."""
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate outcomes must have equal lengths")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    both_passed = 0
    baseline_only = 0
    candidate_only = 0
    both_failed = 0
    for baseline_passed, candidate_passed in zip(baseline, candidate, strict=True):
        if baseline_passed and candidate_passed:
            both_passed += 1
        elif baseline_passed:
            baseline_only += 1
        elif candidate_passed:
            candidate_only += 1
        else:
            both_failed += 1

    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(baseline_only, candidate_only)
        tail_probability = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2 * tail_probability)

    return McNemarResult(
        both_passed=both_passed,
        baseline_only_passed=baseline_only,
        candidate_only_passed=candidate_only,
        both_failed=both_failed,
        exact_p_value=p_value,
        alpha=alpha,
        significant=p_value < alpha,
    )


def compare_paired_results(
    baseline_run_id: str,
    candidate_run_id: str,
    baseline: Sequence[CaseResult],
    candidate: Sequence[CaseResult],
    *,
    alpha: float = 0.05,
) -> RunComparison:
    """Compare two result sets paired by case ID."""
    if not baseline or not candidate:
        raise ValueError("paired comparison requires at least one result per run")

    baseline_by_id = {result.case_id: result for result in baseline}
    candidate_by_id = {result.case_id: result for result in candidate}
    if len(baseline_by_id) != len(baseline) or len(candidate_by_id) != len(candidate):
        raise ValueError("paired comparison requires unique case IDs within each run")
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing_candidate = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        missing_baseline = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        details = []
        if missing_candidate:
            details.append(f"missing from candidate: {', '.join(missing_candidate)}")
        if missing_baseline:
            details.append(f"missing from baseline: {', '.join(missing_baseline)}")
        raise ValueError(f"paired runs must contain identical case IDs ({'; '.join(details)})")

    case_ids = sorted(baseline_by_id)
    baseline_outcomes = [baseline_by_id[case_id].passed for case_id in case_ids]
    candidate_outcomes = [candidate_by_id[case_id].passed for case_id in case_ids]
    regressions = [
        case_id
        for case_id in case_ids
        if baseline_by_id[case_id].passed and not candidate_by_id[case_id].passed
    ]
    improvements = [
        case_id
        for case_id in case_ids
        if not baseline_by_id[case_id].passed and candidate_by_id[case_id].passed
    ]
    baseline_accuracy = sum(baseline_outcomes) / len(case_ids)
    candidate_accuracy = sum(candidate_outcomes) / len(case_ids)

    return RunComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        case_count=len(case_ids),
        baseline_accuracy=baseline_accuracy,
        candidate_accuracy=candidate_accuracy,
        accuracy_delta=candidate_accuracy - baseline_accuracy,
        regressions=regressions,
        improvements=improvements,
        mcnemar=exact_mcnemar_test(baseline_outcomes, candidate_outcomes, alpha=alpha),
    )


def summarize_results(results: Sequence[CaseResult]) -> RunMetrics:
    """Aggregate accuracy, latency, and Ollama throughput from stored case results."""
    if not results:
        return RunMetrics(case_count=0, category_accuracy={})

    outcomes = [result.passed for result in results]
    ci_low, ci_high = bootstrap_accuracy_ci(outcomes)
    by_category: dict[str, list[bool]] = defaultdict(list)
    latencies: list[float] = []
    throughputs: list[float] = []

    for result in results:
        by_category[result.category].append(result.passed)
        latency = result.generation.latency_ms
        throughput = result.generation.tokens_per_second
        if latency is not None:
            latencies.append(latency)
        if throughput is not None:
            throughputs.append(throughput)

    return RunMetrics(
        case_count=len(results),
        accuracy=sum(outcomes) / len(outcomes),
        ci_low=ci_low,
        ci_high=ci_high,
        category_accuracy={
            category: sum(values) / len(values) for category, values in sorted(by_category.items())
        },
        latency_ms_p50=percentile(latencies, 0.5),
        latency_ms_p95=percentile(latencies, 0.95),
        tokens_per_second=median(throughputs) if throughputs else None,
    )
