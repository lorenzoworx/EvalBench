from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import uvicorn
from pydantic import ValidationError

from evalbench.api import create_app
from evalbench.judge import JudgeEvaluator
from evalbench.models import EvaluationSuite, Generation, RunSummary
from evalbench.providers import OllamaProvider, ProviderError, ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database
from evalbench.study import (
    JudgeStudy,
    StudySplit,
    StudyValidationError,
    build_agreement_report,
    build_judge_study,
    freeze_study_rubric,
    load_study,
    record_human_score,
    save_agreement_report,
    save_study,
    score_study_split,
    validate_study_split,
)
from evalbench.suite import SuiteValidationError, load_suite

app = typer.Typer(help="Explainable local-model evaluation.", no_args_is_help=True)
suite_app = typer.Typer(help="Validate and inspect evaluation suites.", no_args_is_help=True)
judge_app = typer.Typer(
    help="Prepare and label the local judge-validation study.", no_args_is_help=True
)
app.add_typer(suite_app, name="suite")
app.add_typer(judge_app, name="judge")


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
    judge_model: Annotated[str | None, typer.Option("--judge-model")] = None,
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
                judge_model=judge_model,
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
    judge_model: str | None,
) -> RunSummary:
    if provider_name == "replay":
        if replay_path is None:
            raise ValueError("Replay runs require --replay PATH.")
        replay_provider = ReplayProvider(_load_replay(replay_path), model_names=(model,))
        if judge_model is None:
            return await RunService(Database(database_path), replay_provider).run(
                suite, model, provider_name="replay"
            )
        async with OllamaProvider() as judge_provider:
            return await RunService(
                Database(database_path),
                replay_provider,
                JudgeEvaluator(judge_provider),
            ).run(
                suite,
                model,
                provider_name="replay",
                judge_model=judge_model,
            )

    if replay_path is not None:
        raise ValueError("--replay can only be used with --provider replay.")
    async with OllamaProvider() as ollama_provider:
        return await RunService(
            Database(database_path),
            ollama_provider,
            JudgeEvaluator(ollama_provider) if judge_model is not None else None,
        ).run(
            suite,
            model,
            provider_name="ollama",
            judge_model=judge_model,
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


@judge_app.command("select")
def select_judge_examples(
    baseline_run_id: Annotated[str, typer.Argument(help="First target-model run ID.")],
    candidate_run_id: Annotated[str, typer.Argument(help="Second target-model run ID.")],
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = Path(
        "labels/judge-study.json"
    ),
    sample_size: Annotated[int, typer.Option("--sample-size", min=2)] = 60,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Select balanced outputs and freeze their calibration/holdout split."""
    if output.exists():
        typer.echo(f"Study already exists at {output}; choose a new output path.", err=True)
        raise typer.Exit(code=1)
    try:
        study = build_judge_study(
            Database(database_path),
            baseline_run_id,
            candidate_run_id,
            sample_size=sample_size,
            seed=seed,
        )
        save_study(output, study)
    except (KeyError, ValueError) as exc:
        typer.echo(f"Study selection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Selected {study.sample_size} blinded outputs: "
        f"{study.sample_size // 2} calibration and {study.sample_size // 2} holdout.\n"
        f"Saved {output}"
    )


@judge_app.command("label")
def label_judge_examples(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    split: Annotated[StudySplit, typer.Option("--split")] = "calibration",
) -> None:
    """Blindly score one study split, saving after every response."""
    try:
        study = load_study(path)
        rubric = validate_study_split(study, split)
    except StudyValidationError as exc:
        typer.echo(f"Labeling failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    pending = [
        example
        for example in study.examples
        if example.split == split and example.human_score is None
    ]
    total = sum(example.split == split for example in study.examples)
    completed = total - len(pending)
    if not pending:
        typer.echo(f"{split.title()} labeling is already complete ({total}/{total}).")
        return

    typer.echo(rubric.system_prompt())
    typer.echo(
        f"\nLabeling {split}: {completed}/{total} complete. "
        "Model identity is hidden. Enter 0, 1, 2, or q to stop."
    )
    for example in pending:
        typer.echo("\n" + "=" * 72)
        typer.echo(f"Sample {example.sample_id} · Category: {example.category}")
        typer.echo(f"\nPROMPT\n{example.prompt}")
        reference = (
            json.dumps(example.expected, ensure_ascii=False)
            if example.expected is not None
            else "—"
        )
        typer.echo(f"\nEXPECTED REFERENCE\n{reference}")
        typer.echo(f"\nCANDIDATE ANSWER\n{example.answer}")
        score: Literal[0, 1, 2]
        while True:
            value = typer.prompt("Score").strip().casefold()
            if value == "q":
                typer.echo(f"Stopped. Progress saved at {completed}/{total}.")
                return
            if value == "0":
                score = 0
                break
            if value == "1":
                score = 1
                break
            if value == "2":
                score = 2
                break
            typer.echo("Enter 0, 1, 2, or q.", err=True)
        study = record_human_score(study, example.sample_id, score)
        save_study(path, study)
        completed += 1
        typer.echo(f"Saved {completed}/{total}.")
    typer.echo(f"{split.title()} labeling complete ({total}/{total}).")


@judge_app.command("score")
def score_judge_study(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    judge_model: Annotated[str, typer.Option("--judge-model")] = "gemma3:4b",
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
    split: Annotated[StudySplit, typer.Option("--split")] = "calibration",
) -> None:
    """Score one study split with an independent local judge, saving each result."""
    try:
        asyncio.run(
            _score_judge_study_file(
                path=path,
                split=split,
                judge_model=judge_model,
                database_path=database_path,
            )
        )
    except (ProviderError, StudyValidationError, ValueError) as exc:
        typer.echo(f"Judge scoring failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _score_judge_study_file(
    *,
    path: Path,
    split: StudySplit,
    judge_model: str,
    database_path: Path,
) -> None:
    study = load_study(path)
    total = sum(example.split == split for example in study.examples)
    completed_before = sum(
        example.split == split and example.judge_score is not None for example in study.examples
    )
    async with OllamaProvider() as provider:
        evaluator = JudgeEvaluator(provider)

        def persist(updated_study: JudgeStudy, completed: int, expected: int) -> None:
            save_study(path, updated_study)
            typer.echo(f"Saved judge score {completed}/{expected}.")

        await score_study_split(
            study,
            split,
            database=Database(database_path),
            evaluator=evaluator,
            judge_model=judge_model,
            on_progress=persist,
        )
    if completed_before == total:
        typer.echo(f"{split.title()} judge scoring is already complete ({total}/{total}).")
    else:
        typer.echo(f"{split.title()} judge scoring complete ({total}/{total}).")


@judge_app.command("report")
def report_judge_study(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    split: Annotated[StudySplit, typer.Option("--split")] = "calibration",
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=1)] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Report human/judge agreement for one complete, preassigned study split."""
    if output is not None and output.exists():
        typer.echo(f"Report already exists at {output}; choose a new output path.", err=True)
        raise typer.Exit(code=1)
    try:
        report = build_agreement_report(
            load_study(path),
            split,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed,
        )
    except (StudyValidationError, ValueError) as exc:
        typer.echo(f"Agreement report failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output is not None:
        save_agreement_report(output, report)
        typer.echo(f"Saved {output}", err=True)
    typer.echo(report.model_dump_json(indent=2))


@judge_app.command("freeze")
def freeze_judge_study(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Adopt the bundled frozen rubric after calibration is complete."""
    try:
        study = freeze_study_rubric(load_study(path))
        save_study(path, study)
    except StudyValidationError as exc:
        typer.echo(f"Rubric freeze failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    calibration_count = sum(
        example.split == "calibration" and example.human_score is not None
        for example in study.examples
    )
    holdout_count = sum(example.split == "holdout" for example in study.examples)
    typer.echo(
        f"Frozen {study.rubric_id} at {study.rubric_version}; preserved "
        f"{calibration_count} calibration pairs and unlocked {holdout_count} holdout examples."
    )


@app.command("serve")
def serve(
    database_path: Annotated[Path, typer.Option("--database", dir_okay=False)] = Path(
        "results/evalbench.db"
    ),
    suite_directory: Annotated[Path, typer.Option("--suites", file_okay=False)] = Path("suites"),
    frontend_directory: Annotated[Path, typer.Option("--frontend", file_okay=False)] = Path(
        "frontend/dist"
    ),
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
) -> None:
    """Serve EvalBench locally with Ollama-backed run control."""
    web_app = create_app(
        database_path=database_path,
        suite_directory=suite_directory,
        frontend_directory=frontend_directory,
    )
    if not web_app.state.frontend_mounted:
        typer.echo(
            "Frontend build not found; serving the API only. "
            "Run `npm run build` in frontend/ for the integrated dashboard.",
            err=True,
        )
    typer.echo(f"EvalBench listening at http://127.0.0.1:{port}")
    uvicorn.run(web_app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
