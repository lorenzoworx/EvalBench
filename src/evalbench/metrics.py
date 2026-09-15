from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import median

from evalbench.models import CaseResult, RunMetrics


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
