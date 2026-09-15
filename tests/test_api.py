from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from evalbench.api import create_app, get_database
from evalbench.models import CaseResult, EvaluationCase, Generation, RunSummary
from evalbench.store import Database


def write_suite(directory: Path) -> None:
    directory.mkdir()
    (directory / "tiny.yaml").write_text(
        """name: Tiny Suite
version: "1"
cases:
  - id: one
    category: reasoning
    prompt: What is one plus one?
    expected: "2"
    graders: [{type: numeric}]
  - id: two
    category: safety
    prompt: Refuse this request.
    graders: [{type: refusal}]
""",
        encoding="utf-8",
    )


def make_case(case_id: str = "one") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        category="reasoning",
        prompt=f"Prompt for {case_id}",
        expected="answer",
        graders=[{"type": "exact"}],
    )


def save_run(
    database: Database,
    run_id: str,
    outcomes: dict[str, bool],
    *,
    created_at: datetime,
) -> None:
    database.create_run(
        RunSummary(
            id=run_id,
            suite_name="api-suite",
            suite_version="1",
            model=run_id,
            provider="replay",
            status="completed",
            created_at=created_at,
            completed_at=created_at,
            completed_cases=len(outcomes),
            total_cases=len(outcomes),
            accuracy=sum(outcomes.values()) / len(outcomes),
        )
    )
    for case_id, passed in outcomes.items():
        case = make_case(case_id)
        database.save_case_result(
            run_id,
            case,
            CaseResult(
                case_id=case.id,
                category=case.category,
                prompt=case.prompt,
                expected=case.expected,
                generation=Generation(
                    text="answer",
                    output_tokens=10,
                    total_duration_ns=2_000_000,
                    eval_duration_ns=1_000_000_000,
                ),
                judgments=[],
                passed=passed,
            ),
        )


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Database]]:
    suite_directory = tmp_path / "suites"
    write_suite(suite_directory)
    database_path = tmp_path / "evalbench.db"
    app = create_app(database_path=database_path, suite_directory=suite_directory)
    database = Database(database_path)
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 1, 2, tzinfo=UTC)
    save_run(database, "baseline", {"one": True, "two": False}, created_at=older)
    save_run(database, "candidate", {"one": False, "two": True}, created_at=newer)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client, database


async def test_lists_suites_and_runs_newest_first(
    api: tuple[AsyncClient, Database],
) -> None:
    client, _ = api

    suites = await client.get("/api/suites")
    runs = await client.get("/api/runs")

    assert suites.status_code == 200
    assert suites.json() == [
        {
            "id": "tiny",
            "name": "Tiny Suite",
            "version": "1",
            "case_count": 2,
            "categories": ["reasoning", "safety"],
        }
    ]
    assert runs.status_code == 200
    assert [run["id"] for run in runs.json()] == ["candidate", "baseline"]


async def test_reads_run_detail_status_results_and_metrics(
    api: tuple[AsyncClient, Database],
) -> None:
    client, _ = api

    detail = await client.get("/api/runs/baseline")
    run_status = await client.get("/api/runs/baseline/status")
    results = await client.get("/api/runs/baseline/results")
    metrics = await client.get("/api/runs/baseline/metrics")

    assert detail.status_code == 200
    assert detail.json()["model"] == "baseline"
    assert run_status.json() == {
        "id": "baseline",
        "status": "completed",
        "completed_cases": 2,
        "total_cases": 2,
        "error": None,
    }
    assert [result["case_id"] for result in results.json()] == ["one", "two"]
    assert metrics.status_code == 200
    assert metrics.json()["accuracy"] == 0.5
    assert metrics.json()["latency_ms_p50"] == 2
    assert metrics.json()["tokens_per_second"] == 10


async def test_compares_paired_runs(api: tuple[AsyncClient, Database]) -> None:
    client, _ = api

    response = await client.get(
        "/api/compare",
        params={
            "baseline_run_id": "baseline",
            "candidate_run_id": "candidate",
            "alpha": 0.1,
        },
    )

    assert response.status_code == 200
    comparison = response.json()
    assert comparison["accuracy_delta"] == 0
    assert comparison["regressions"] == ["one"]
    assert comparison["improvements"] == ["two"]
    assert comparison["mcnemar"]["alpha"] == 0.1
    assert comparison["mcnemar"]["exact_p_value"] == 1


@pytest.mark.parametrize("suffix", ["", "/status", "/results", "/metrics"])
async def test_unknown_run_returns_404(api: tuple[AsyncClient, Database], suffix: str) -> None:
    client, _ = api

    response = await client.get(f"/api/runs/missing{suffix}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Run 'missing' does not exist."


async def test_invalid_comparison_returns_clear_client_errors(
    api: tuple[AsyncClient, Database],
) -> None:
    client, _ = api

    missing = await client.get(
        "/api/compare",
        params={"baseline_run_id": "baseline", "candidate_run_id": "missing"},
    )
    same = await client.get(
        "/api/compare",
        params={"baseline_run_id": "baseline", "candidate_run_id": "baseline"},
    )
    invalid_alpha = await client.get(
        "/api/compare",
        params={
            "baseline_run_id": "baseline",
            "candidate_run_id": "candidate",
            "alpha": 1,
        },
    )

    assert missing.status_code == 404
    assert "Candidate run 'missing' does not exist" in missing.json()["detail"]
    assert same.status_code == 400
    assert same.json()["detail"] == "baseline and candidate must be different runs"
    assert invalid_alpha.status_code == 422


async def test_database_dependency_can_be_overridden(tmp_path: Path) -> None:
    default_database = tmp_path / "default.db"
    alternate = Database(tmp_path / "alternate.db")
    alternate.initialize()
    save_run(
        alternate,
        "alternate",
        {"one": True},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    app = create_app(database_path=default_database, suite_directory=tmp_path)
    app.dependency_overrides[get_database] = lambda: alternate

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/runs")

    assert response.status_code == 200
    assert [run["id"] for run in response.json()] == ["alternate"]
