from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from evalbench.jobs import ActiveRunError, RunCoordinator
from evalbench.models import (
    CancellationAccepted,
    CaseResult,
    EvaluationSuite,
    RunAccepted,
    RunComparison,
    RunMetrics,
    RunRequest,
    RunStatus,
    RunSummary,
    SuiteSummary,
)
from evalbench.providers import OllamaProvider, Provider, ProviderError
from evalbench.service import RunService
from evalbench.store import Database
from evalbench.suite import discover_suites
from evalbench.web import mount_frontend


def get_database(request: Request) -> Database:
    """Return the application database; replaceable through FastAPI dependencies."""
    return cast(Database, request.app.state.database)


def get_suite_directory(request: Request) -> Path:
    """Return the configured suite directory."""
    return cast(Path, request.app.state.suite_directory)


def get_coordinator(request: Request) -> RunCoordinator:
    """Return the process-local background run coordinator."""
    return cast(RunCoordinator, request.app.state.coordinator)


DatabaseDependency = Annotated[Database, Depends(get_database)]
SuiteDirectoryDependency = Annotated[Path, Depends(get_suite_directory)]
CoordinatorDependency = Annotated[RunCoordinator, Depends(get_coordinator)]


def create_app(
    *,
    database_path: str | Path = "results/evalbench.db",
    suite_directory: str | Path = "suites",
    provider: Provider | None = None,
    provider_name: str = "ollama",
    frontend_directory: str | Path | None = "frontend/dist",
) -> FastAPI:
    """Build the local EvalBench API with explicit filesystem dependencies."""
    database = Database(database_path)
    database.initialize()
    database.interrupt_unfinished_runs()
    owns_provider = provider is None
    selected_provider: Provider = provider if provider is not None else OllamaProvider()
    coordinator = RunCoordinator(
        database,
        selected_provider,
        provider_name=provider_name,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await coordinator.shutdown()
            if owns_provider:
                await cast(OllamaProvider, selected_provider).aclose()

    app = FastAPI(title="EvalBench API", version="0.1.0", lifespan=lifespan)
    app.state.database = database
    app.state.suite_directory = Path(suite_directory)
    app.state.coordinator = coordinator

    @app.get("/api/models", response_model=list[str])
    async def list_models(coordinator: CoordinatorDependency) -> list[str]:
        try:
            return await coordinator.models()
        except ProviderError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.get("/api/suites", response_model=list[SuiteSummary])
    def list_suites(directory: SuiteDirectoryDependency) -> list[SuiteSummary]:
        return [
            SuiteSummary(
                id=path.stem,
                name=suite.name,
                version=suite.version,
                case_count=len(suite.cases),
                categories=sorted({case.category for case in suite.cases}),
            )
            for path, suite in discover_suites(directory)
        ]

    @app.post(
        "/api/runs",
        response_model=RunAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_run(
        request: RunRequest,
        directory: SuiteDirectoryDependency,
        coordinator: CoordinatorDependency,
    ) -> RunAccepted:
        suite = _require_suite(directory, request.suite_id)
        try:
            run = await coordinator.start(suite, request.model)
        except ActiveRunError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return RunAccepted(id=run.id)

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs(database: DatabaseDependency) -> list[RunSummary]:
        return database.list_runs()

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str, database: DatabaseDependency) -> RunSummary:
        return _require_run(database, run_id)

    @app.get("/api/runs/{run_id}/status", response_model=RunStatus)
    def get_run_status(run_id: str, database: DatabaseDependency) -> RunStatus:
        run = _require_run(database, run_id)
        return RunStatus(
            id=run.id,
            status=run.status,
            completed_cases=run.completed_cases,
            total_cases=run.total_cases,
            error=run.error,
        )

    @app.post(
        "/api/runs/{run_id}/cancel",
        response_model=CancellationAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_run(
        run_id: str,
        coordinator: CoordinatorDependency,
    ) -> CancellationAccepted:
        try:
            await coordinator.cancel(run_id)
        except KeyError as exc:
            raise _not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return CancellationAccepted(id=run_id)

    @app.get("/api/runs/{run_id}/results", response_model=list[CaseResult])
    def list_run_results(run_id: str, database: DatabaseDependency) -> list[CaseResult]:
        _require_run(database, run_id)
        return [record.result for record in database.list_case_results(run_id)]

    @app.get("/api/runs/{run_id}/metrics", response_model=RunMetrics)
    def get_run_metrics(run_id: str, database: DatabaseDependency) -> RunMetrics:
        try:
            return RunService(database).metrics(run_id)
        except KeyError as exc:
            raise _not_found(exc) from exc

    @app.get("/api/compare", response_model=RunComparison)
    def compare_runs(
        database: DatabaseDependency,
        baseline_run_id: Annotated[str, Query(min_length=1)],
        candidate_run_id: Annotated[str, Query(min_length=1)],
        alpha: Annotated[float, Query(gt=0, lt=1)] = 0.05,
    ) -> RunComparison:
        try:
            return RunService(database).compare(
                baseline_run_id,
                candidate_run_id,
                alpha=alpha,
            )
        except KeyError as exc:
            raise _not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    app.state.frontend_mounted = mount_frontend(app, frontend_directory)
    return app


def _require_run(database: Database, run_id: str) -> RunSummary:
    run = database.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id!r} does not exist.",
        )
    return run


def _require_suite(directory: Path, suite_id: str) -> EvaluationSuite:
    for path, suite in discover_suites(directory):
        if path.stem == suite_id:
            return suite
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Suite {suite_id!r} does not exist.",
    )


def _not_found(exc: KeyError) -> HTTPException:
    detail = str(exc.args[0]) if exc.args else "Resource does not exist."
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
