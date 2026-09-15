from pathlib import Path

import pytest

from evalbench.models import CaseResult, EvaluationCase, EvaluationSuite, Generation, RunSummary
from evalbench.providers import ProviderError, ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database


def make_suite() -> EvaluationSuite:
    return EvaluationSuite(
        name="tiny",
        version="1",
        cases=[
            EvaluationCase(
                id="one",
                category="math",
                prompt="One plus one?",
                expected="2",
                graders=[{"type": "numeric"}],
            ),
            EvaluationCase(
                id="two",
                category="instruction",
                prompt="Say yes.",
                expected="yes",
                graders=[{"type": "exact"}],
            ),
        ],
    )


class CountingReplayProvider(ReplayProvider):
    def __init__(self, responses: dict[str, str | Generation]) -> None:
        super().__init__(responses)
        self.calls = 0

    async def generate(self, model: str, case: EvaluationCase) -> Generation:
        self.calls += 1
        return await super().generate(model, case)


@pytest.mark.asyncio
async def test_replay_runner_sqlite_end_to_end(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    provider = CountingReplayProvider({"one": "2", "two": "no"})
    service = RunService(database, provider)

    run = await service.run(make_suite(), "replay", provider_name="replay", run_id="known-run")

    assert run.id == "known-run"
    assert run.status == "completed"
    assert run.completed_cases == 2
    assert run.accuracy == 0.5
    assert run.ci_low == 0.0
    assert run.ci_high == 1.0
    assert run.completed_at is not None
    assert database.get_run(run.id) == run
    records = database.list_case_results(run.id)
    assert [record.result.passed for record in records] == [True, False]
    assert provider.calls == 2
    metrics = service.metrics(run.id)
    assert metrics.accuracy == 0.5
    assert metrics.category_accuracy == {"instruction": 0.0, "math": 1.0}


@pytest.mark.asyncio
async def test_second_run_reuses_generation_cache(tmp_path: Path) -> None:
    provider = CountingReplayProvider({"one": "2", "two": "yes"})
    service = RunService(Database(tmp_path / "evalbench.db"), provider)

    await service.run(make_suite(), "replay", provider_name="replay", run_id="first")
    second = await service.run(make_suite(), "replay", provider_name="replay", run_id="second")

    assert provider.calls == 2
    assert all(
        record.result.generation.cached for record in service.database.list_case_results(second.id)
    )


@pytest.mark.asyncio
async def test_unavailable_model_records_failed_run(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    service = RunService(database, ReplayProvider({}))

    with pytest.raises(ProviderError, match="missing.*unavailable"):
        await service.run(make_suite(), "missing", provider_name="replay", run_id="failed-run")

    failed = database.get_run("failed-run")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.completed_cases == 0
    assert "missing" in (failed.error or "")


@pytest.mark.asyncio
async def test_provider_failure_retains_completed_cases(tmp_path: Path) -> None:
    class FailingProvider(CountingReplayProvider):
        async def generate(self, model: str, case: EvaluationCase) -> Generation:
            if case.id == "two":
                raise ProviderError("fixture provider stopped")
            return await super().generate(model, case)

    database = Database(tmp_path / "evalbench.db")
    service = RunService(database, FailingProvider({"one": "2"}))

    with pytest.raises(ProviderError, match="fixture provider stopped"):
        await service.run(make_suite(), "replay", provider_name="replay", run_id="partial-run")

    failed = database.get_run("partial-run")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.completed_cases == 1
    assert len(database.list_case_results("partial-run")) == 1


@pytest.mark.asyncio
async def test_cooperative_cancellation_retains_completed_cases(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    provider = CountingReplayProvider({"one": "2", "two": "yes"})
    service = RunService(database, provider)
    checks = iter([False, False, True])

    run = await service.run(
        make_suite(),
        "replay",
        provider_name="replay",
        run_id="cancelled-run",
        should_cancel=lambda: next(checks),
    )

    assert run.status == "cancelled"
    assert run.completed_cases == 1
    assert run.accuracy == 1
    assert run.completed_at is not None
    assert len(database.list_case_results(run.id)) == 1
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_judge_case_fails_before_provider_call(tmp_path: Path) -> None:
    case = EvaluationCase(
        id="judge-case",
        category="reasoning",
        prompt="Explain.",
        graders=[{"type": "judge"}],
    )
    suite = EvaluationSuite(name="judge", version="1", cases=[case])
    provider = CountingReplayProvider({"judge-case": "answer"})
    database = Database(tmp_path / "evalbench.db")

    with pytest.raises(ValueError, match="configured judge orchestration.*judge-case"):
        await RunService(database, provider).run(
            suite, "replay", provider_name="replay", run_id="judge-run"
        )

    assert provider.calls == 0
    assert database.get_run("judge-run").status == "failed"


def test_metrics_reject_unknown_run(tmp_path: Path) -> None:
    service = RunService(Database(tmp_path / "evalbench.db"), ReplayProvider({}))

    with pytest.raises(KeyError, match="missing"):
        service.metrics("missing")


def save_completed_run(
    database: Database,
    run_id: str,
    suite: EvaluationSuite,
    outcomes: dict[str, bool],
) -> None:
    run = RunSummary(
        id=run_id,
        suite_name=suite.name,
        suite_version=suite.version,
        model=run_id,
        provider="replay",
        status="completed",
        completed_cases=len(suite.cases),
        total_cases=len(suite.cases),
        accuracy=sum(outcomes.values()) / len(outcomes),
    )
    database.create_run(run)
    for case in suite.cases:
        database.save_case_result(
            run_id,
            case,
            CaseResult(
                case_id=case.id,
                category=case.category,
                prompt=case.prompt,
                expected=case.expected,
                generation=Generation(text="answer"),
                judgments=[],
                passed=outcomes[case.id],
            ),
        )


def test_compare_completed_runs_end_to_end(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    database.initialize()
    suite = make_suite()
    save_completed_run(database, "base", suite, {"one": True, "two": False})
    save_completed_run(database, "candidate", suite, {"one": False, "two": True})

    comparison = RunService(database).compare("base", "candidate")

    assert comparison.accuracy_delta == 0
    assert comparison.regressions == ["one"]
    assert comparison.improvements == ["two"]
    assert comparison.mcnemar.exact_p_value == 1


def test_compare_validates_run_identity_status_and_suite(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    database.initialize()
    suite = make_suite()
    save_completed_run(database, "base", suite, {"one": True, "two": False})
    service = RunService(database)

    with pytest.raises(ValueError, match="different runs"):
        service.compare("base", "base")
    with pytest.raises(KeyError, match="Candidate run.*missing"):
        service.compare("base", "missing")

    pending = RunSummary(
        id="pending",
        suite_name=suite.name,
        suite_version=suite.version,
        model="pending",
        provider="replay",
        status="pending",
        total_cases=2,
    )
    database.create_run(pending)
    with pytest.raises(ValueError, match="completed runs"):
        service.compare("base", "pending")

    other_suite = suite.model_copy(update={"version": "2"})
    save_completed_run(database, "other-suite", other_suite, {"one": True, "two": False})
    with pytest.raises(ValueError, match="same suite"):
        service.compare("base", "other-suite")


def test_compare_rejects_incomplete_results_and_changed_snapshots(tmp_path: Path) -> None:
    database = Database(tmp_path / "evalbench.db")
    database.initialize()
    suite = make_suite()
    save_completed_run(database, "base", suite, {"one": True, "two": False})

    incomplete = RunSummary(
        id="incomplete",
        suite_name=suite.name,
        suite_version=suite.version,
        model="incomplete",
        provider="replay",
        status="completed",
        completed_cases=2,
        total_cases=2,
    )
    database.create_run(incomplete)
    database.save_case_result(
        "incomplete",
        suite.cases[0],
        CaseResult(
            case_id="one",
            category="math",
            prompt="One plus one?",
            expected="2",
            generation=Generation(text="2"),
            judgments=[],
            passed=True,
        ),
    )
    service = RunService(database)
    with pytest.raises(ValueError, match="complete result set"):
        service.compare("base", "incomplete")

    changed_suite = suite.model_copy(
        update={
            "cases": [
                suite.cases[0].model_copy(update={"prompt": "Changed prompt"}),
                suite.cases[1],
            ]
        }
    )
    save_completed_run(database, "changed", changed_suite, {"one": True, "two": False})
    with pytest.raises(ValueError, match="different snapshots.*one"):
        service.compare("base", "changed")
