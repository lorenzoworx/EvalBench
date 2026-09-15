from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from evalbench.cache import generation_request_hash
from evalbench.graders import grade_case, passed_all_primary
from evalbench.metrics import bootstrap_accuracy_ci, compare_paired_results, summarize_results
from evalbench.models import CaseResult, EvaluationSuite, RunComparison, RunMetrics, RunSummary
from evalbench.providers import Provider
from evalbench.store import Database


class RunService:
    """Orchestrate deterministic evaluation independently of its interface."""

    def __init__(self, database: Database, provider: Provider | None = None) -> None:
        self.database = database
        self.provider = provider
        self.database.initialize()

    async def run(
        self,
        suite: EvaluationSuite,
        model: str,
        *,
        provider_name: str,
        run_id: str | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunSummary:
        if self.provider is None:
            raise RuntimeError("A provider is required to start an evaluation run.")
        run = self.create_run(
            suite,
            model,
            provider_name=provider_name,
            run_id=run_id,
        )
        return await self.execute(run, suite, should_cancel=should_cancel)

    def create_run(
        self,
        suite: EvaluationSuite,
        model: str,
        *,
        provider_name: str,
        run_id: str | None = None,
    ) -> RunSummary:
        """Persist a pending run before asynchronous execution begins."""
        run = RunSummary(
            id=run_id or str(uuid.uuid4()),
            suite_name=suite.name,
            suite_version=suite.version,
            model=model,
            provider=provider_name,
            status="pending",
            total_cases=len(suite.cases),
        )
        self.database.create_run(run)
        return run

    async def execute(
        self,
        run: RunSummary,
        suite: EvaluationSuite,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunSummary:
        """Execute a previously persisted run, checking cancellation between cases."""
        if self.provider is None:
            raise RuntimeError("A provider is required to start an evaluation run.")
        provider = self.provider
        if run.status != "pending":
            raise ValueError(f"Run {run.id!r} must be pending before execution.")
        run = run.model_copy(update={"status": "running"})
        self.database.update_run(run)

        outcomes: list[bool] = []
        try:
            if should_cancel is not None and should_cancel():
                return self._cancel(run, outcomes)

            unsupported = [
                case.id
                for case in suite.cases
                if any(spec.type == "judge" for spec in case.graders)
            ]
            if unsupported:
                raise ValueError(
                    "Judge graders are not available until Milestone 5; "
                    f"affected cases: {', '.join(unsupported)}."
                )

            await provider.preflight(run.model)

            for case in suite.cases:
                if should_cancel is not None and should_cancel():
                    return self._cancel(run, outcomes)
                request_hash = generation_request_hash(
                    model=run.model,
                    case=case,
                    provider_schema_version=provider.schema_version,
                )
                generation = self.database.get_cached_generation(request_hash)
                if generation is None:
                    generation = await provider.generate(run.model, case)
                    self.database.put_cached_generation(request_hash, generation)

                judgments = grade_case(case, generation.text)
                passed = passed_all_primary(judgments)
                result = CaseResult(
                    case_id=case.id,
                    category=case.category,
                    prompt=case.prompt,
                    expected=case.expected,
                    generation=generation,
                    judgments=judgments,
                    passed=passed,
                )
                self.database.save_case_result(run.id, case, result)
                outcomes.append(passed)
                run = run.model_copy(update={"completed_cases": len(outcomes)})
                self.database.update_run(run)

            ci_low, ci_high = bootstrap_accuracy_ci(outcomes)
            run = run.model_copy(
                update={
                    "status": "completed",
                    "completed_at": datetime.now(UTC),
                    "accuracy": sum(outcomes) / len(outcomes),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
            self.database.update_run(run)
            return run
        except Exception as exc:
            failed = run.model_copy(
                update={
                    "status": "failed",
                    "completed_at": datetime.now(UTC),
                    "error": str(exc),
                }
            )
            self.database.update_run(failed)
            raise

    def _cancel(self, run: RunSummary, outcomes: list[bool]) -> RunSummary:
        update: dict[str, object] = {
            "status": "cancelled",
            "completed_at": datetime.now(UTC),
        }
        if outcomes:
            ci_low, ci_high = bootstrap_accuracy_ci(outcomes)
            update.update(
                {
                    "accuracy": sum(outcomes) / len(outcomes),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        cancelled = run.model_copy(update=update)
        self.database.update_run(cancelled)
        return cancelled

    def metrics(self, run_id: str) -> RunMetrics:
        if self.database.get_run(run_id) is None:
            raise KeyError(f"Run {run_id!r} does not exist.")
        return summarize_results(
            [record.result for record in self.database.list_case_results(run_id)]
        )

    def compare(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        *,
        alpha: float = 0.05,
    ) -> RunComparison:
        if baseline_run_id == candidate_run_id:
            raise ValueError("baseline and candidate must be different runs")

        baseline_run = self.database.get_run(baseline_run_id)
        if baseline_run is None:
            raise KeyError(f"Baseline run {baseline_run_id!r} does not exist.")
        candidate_run = self.database.get_run(candidate_run_id)
        if candidate_run is None:
            raise KeyError(f"Candidate run {candidate_run_id!r} does not exist.")
        if baseline_run.status != "completed" or candidate_run.status != "completed":
            raise ValueError("paired comparison requires two completed runs")
        if (
            baseline_run.suite_name,
            baseline_run.suite_version,
        ) != (
            candidate_run.suite_name,
            candidate_run.suite_version,
        ):
            raise ValueError("paired runs must use the same suite name and version")

        baseline_records = self.database.list_case_results(baseline_run_id)
        candidate_records = self.database.list_case_results(candidate_run_id)
        for run, records in (
            (baseline_run, baseline_records),
            (candidate_run, candidate_records),
        ):
            if run.completed_cases != run.total_cases or len(records) != run.total_cases:
                raise ValueError(f"Run {run.id!r} does not contain a complete result set.")

        baseline_cases = {record.case.id: record.case for record in baseline_records}
        candidate_cases = {record.case.id: record.case for record in candidate_records}
        shared_case_ids = baseline_cases.keys() & candidate_cases.keys()
        changed_cases = sorted(
            case_id
            for case_id in shared_case_ids
            if baseline_cases[case_id] != candidate_cases[case_id]
        )
        if changed_cases:
            raise ValueError(
                "paired runs contain different snapshots for cases: " + ", ".join(changed_cases)
            )

        return compare_paired_results(
            baseline_run_id,
            candidate_run_id,
            [record.result for record in baseline_records],
            [record.result for record in candidate_records],
            alpha=alpha,
        )
