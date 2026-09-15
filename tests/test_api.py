import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from evalbench.api import create_app, get_database
from evalbench.jobs import RunCoordinator
from evalbench.models import CaseResult, EvaluationCase, Generation, RunSummary
from evalbench.providers import ProviderError, ReplayProvider
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
    app = create_app(
        database_path=database_path,
        suite_directory=suite_directory,
        provider=ReplayProvider({}, model_names=("fixture",)),
        provider_name="replay",
    )
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
    app = create_app(
        database_path=default_database,
        suite_directory=tmp_path,
        provider=ReplayProvider({}),
        provider_name="replay",
    )
    app.dependency_overrides[get_database] = lambda: alternate

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/runs")

    assert response.status_code == 200
    assert [run["id"] for run in response.json()] == ["alternate"]


async def test_built_frontend_serves_assets_and_spa_routes_without_masking_api(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "dist"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (frontend / "index.html").write_text("<main>EvalBench shell</main>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('loaded')", encoding="utf-8")
    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=tmp_path,
        frontend_directory=frontend,
        provider=ReplayProvider({}),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        root = await client.get("/")
        client_route = await client.get("/runs/example")
        asset = await client.get("/assets/app.js")
        missing_asset = await client.get("/assets/missing.js")
        missing_api = await client.get("/api/not-a-route")

    assert app.state.frontend_mounted is True
    assert root.status_code == 200
    assert root.text == "<main>EvalBench shell</main>"
    assert client_route.status_code == 200
    assert client_route.text == root.text
    assert asset.status_code == 200
    assert asset.text == "console.log('loaded')"
    assert missing_asset.status_code == 404
    assert missing_api.status_code == 404
    assert "EvalBench shell" not in missing_api.text


def test_missing_frontend_build_leaves_api_only(tmp_path: Path) -> None:
    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=tmp_path,
        frontend_directory=tmp_path / "missing",
        provider=ReplayProvider({}),
    )

    assert app.state.frontend_mounted is False


async def test_models_and_background_run_completion(tmp_path: Path) -> None:
    suite_directory = tmp_path / "suites"
    write_suite(suite_directory)
    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=suite_directory,
        provider=ReplayProvider(
            {"one": "2", "two": "I cannot help with that request."},
            model_names=("fixture-model",),
        ),
        provider_name="replay",
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        models = await client.get("/api/models")
        started = await client.post(
            "/api/runs",
            json={"suite_id": "tiny", "model": "fixture-model"},
        )
        await app.state.coordinator.wait_until_idle()
        run_id = started.json()["id"]
        completed = await client.get(f"/api/runs/{run_id}/status")

    assert models.status_code == 200
    assert models.json() == ["fixture-model"]
    assert started.status_code == 202
    assert started.json()["status"] == "pending"
    assert completed.json()["status"] == "completed"
    assert completed.json()["completed_cases"] == 2


class BlockingProvider:
    schema_version = "blocking-v1"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def models(self) -> list[str]:
        return ["blocking-model"]

    async def preflight(self, model: str) -> None:
        assert model == "blocking-model"

    async def generate(self, model: str, case: EvaluationCase) -> Generation:
        if case.id == "one":
            self.started.set()
            await self.release.wait()
            return Generation(text="2")
        return Generation(text="I cannot help with that request.")


async def test_rejects_concurrent_run_and_cancels_between_cases(tmp_path: Path) -> None:
    suite_directory = tmp_path / "suites"
    write_suite(suite_directory)
    provider = BlockingProvider()
    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=suite_directory,
        provider=provider,
        provider_name="test",
    )
    coordinator: RunCoordinator = app.state.coordinator

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        started = await client.post(
            "/api/runs",
            json={"suite_id": "tiny", "model": "blocking-model"},
        )
        run_id = started.json()["id"]
        await provider.started.wait()

        conflict = await client.post(
            "/api/runs",
            json={"suite_id": "tiny", "model": "blocking-model"},
        )
        cancellation = await client.post(f"/api/runs/{run_id}/cancel")
        provider.release.set()
        await coordinator.wait_until_idle()
        cancelled = await client.get(f"/api/runs/{run_id}/status")
        results = await client.get(f"/api/runs/{run_id}/results")
        repeated_cancel = await client.post(f"/api/runs/{run_id}/cancel")
        next_started = await client.post(
            "/api/runs",
            json={"suite_id": "tiny", "model": "blocking-model"},
        )
        await coordinator.wait_until_idle()
        next_status = await client.get(f"/api/runs/{next_started.json()['id']}/status")

    assert conflict.status_code == 409
    assert run_id in conflict.json()["detail"]
    assert cancellation.status_code == 202
    assert cancellation.json() == {"id": run_id, "cancellation_requested": True}
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["completed_cases"] == 1
    assert [result["case_id"] for result in results.json()] == ["one"]
    assert repeated_cancel.status_code == 409
    assert next_started.status_code == 202
    assert next_status.json()["status"] == "completed"


async def test_start_validates_request_and_suite(tmp_path: Path) -> None:
    suite_directory = tmp_path / "suites"
    write_suite(suite_directory)
    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=suite_directory,
        provider=ReplayProvider({}),
        provider_name="replay",
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        missing = await client.post(
            "/api/runs",
            json={"suite_id": "missing", "model": "replay"},
        )
        invalid = await client.post(
            "/api/runs",
            json={"suite_id": "", "model": ""},
        )
        missing_cancel = await client.post("/api/runs/missing/cancel")

    assert missing.status_code == 404
    assert missing.json()["detail"] == "Suite 'missing' does not exist."
    assert invalid.status_code == 422
    assert missing_cancel.status_code == 404


async def test_model_discovery_failure_returns_503(tmp_path: Path) -> None:
    class FailingModelsProvider(ReplayProvider):
        async def models(self) -> list[str]:
            raise ProviderError("model service unavailable")

    app = create_app(
        database_path=tmp_path / "runs.db",
        suite_directory=tmp_path,
        provider=FailingModelsProvider({}),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/models")

    assert response.status_code == 503
    assert response.json()["detail"] == "model service unavailable"


def test_app_startup_marks_abandoned_runs_interrupted(tmp_path: Path) -> None:
    database_path = tmp_path / "runs.db"
    database = Database(database_path)
    database.initialize()
    database.create_run(
        RunSummary(
            id="abandoned",
            suite_name="tiny",
            suite_version="1",
            model="fixture",
            provider="replay",
            status="running",
            total_cases=2,
        )
    )

    create_app(
        database_path=database_path,
        suite_directory=tmp_path,
        provider=ReplayProvider({}),
    )

    recovered = database.get_run("abandoned")
    assert recovered is not None
    assert recovered.status == "interrupted"
    assert recovered.completed_at is not None
