from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import ValidationError

from evalbench.models import EvaluationSuite, Generation, RunSummary
from evalbench.providers import OllamaProvider, ProviderError, ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database
from evalbench.suite import SuiteValidationError, load_suite

app = typer.Typer(help="Explainable local-model evaluation.", no_args_is_help=True)
suite_app = typer.Typer(help="Validate and inspect evaluation suites.", no_args_is_help=True)
app.add_typer(suite_app, name="suite")


def _load_replay(path: Path) -> dict[str, str | Generation]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Replay data must be a JSON object keyed by case ID.")
    responses: dict[str, str | Generation] = {}
    for case_id, value in raw.items():
        if not isinstance(case_id, str):
            raise ValueError("Every replay case ID must be a string.")
        responses[case_id] = value if isinstance(value, str) else Generation.model_validate(value)
    return responses


@suite_app.command("validate")
def validate_suite(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a YAML suite and report its dimensions."""
    try:
        suite = load_suite(path)
    except SuiteValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    categories = sorted({case.category for case in suite.cases})
    typer.echo(
        f"Valid: {suite.name} v{suite.version} — {len(suite.cases)} cases "
        f"across {len(categories)} categories."
    )


@app.command("run")
def run_evaluation(
    provider_name: Annotated[Literal["ollama", "replay"], typer.Option("--provider")] = "ollama",
    replay: Annotated[
        Path | None, typer.Option("--replay", exists=True, dir_okay=False, readable=True)
    ] = None,
    suite_path: Annotated[
        Path, typer.Option("--suite", exists=True, dir_okay=False, readable=True)
    ] = Path("suites/core.yaml"),
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
    model: Annotated[str | None, typer.Option("--model")] = None,
) -> None:
    """Evaluate a suite through local Ollama or deterministic replay data."""
    try:
        suite = load_suite(suite_path)
        selected_model = model or ("replay" if provider_name == "replay" else "qwen3:0.6b")
        run = asyncio.run(
            _execute_run(
                suite=suite,
                model=selected_model,
                provider_name=provider_name,
                replay_path=replay,
                database_path=database_path,
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        SuiteValidationError,
        ProviderError,
        ValueError,
    ) as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(run.model_dump_json(indent=2))


async def _execute_run(
    *,
    suite: EvaluationSuite,
    model: str,
    provider_name: Literal["ollama", "replay"],
    replay_path: Path | None,
    database_path: Path,
) -> RunSummary:
    if provider_name == "replay":
        if replay_path is None:
            raise ValueError("Replay runs require --replay PATH.")
        replay_provider = ReplayProvider(_load_replay(replay_path), model_names=(model,))
        return await RunService(Database(database_path), replay_provider).run(
            suite, model, provider_name="replay"
        )

    if replay_path is not None:
        raise ValueError("--replay can only be used with --provider replay.")
    async with OllamaProvider() as ollama_provider:
        return await RunService(Database(database_path), ollama_provider).run(
            suite, model, provider_name="ollama"
        )


@app.command("runs")
def list_runs(
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
) -> None:
    """List recorded runs newest first."""
    database = Database(database_path)
    database.initialize()
    typer.echo(
        json.dumps(
            [run.model_dump(mode="json") for run in database.list_runs()],
            indent=2,
        )
    )


@app.command("compare")
def compare_runs(
    baseline_run_id: Annotated[str, typer.Argument(help="Run ID used as the baseline.")],
    candidate_run_id: Annotated[str, typer.Argument(help="Run ID evaluated as the candidate.")],
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
) -> None:
    """Compare two completed runs over the same suite and case snapshots."""
    try:
        comparison = RunService(Database(database_path)).compare(baseline_run_id, candidate_run_id)
    except (KeyError, ValueError) as exc:
        typer.echo(f"Comparison failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(comparison.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
