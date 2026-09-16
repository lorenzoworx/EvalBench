import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import evalbench.cli as cli_module
import evalbench.study as study_module
from evalbench.cli import app
from evalbench.models import EvaluationCase, EvaluationSuite, Generation
from evalbench.providers import ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database
from evalbench.study import load_study

runner = CliRunner()


def write_suite(path: Path) -> None:
    path.write_text(
        """name: tiny
version: "1"
cases:
  - id: one
    category: reasoning
    prompt: What is one plus one?
    expected: "2"
    graders: [{type: numeric}]
""",
        encoding="utf-8",
    )


def write_judge_suite(path: Path) -> None:
    path.write_text(
        """name: judged
version: "1"
cases:
  - id: one
    category: reasoning
    prompt: Explain the answer.
    graders: [{type: judge, minimum_score: 2}]
""",
        encoding="utf-8",
    )


def prepare_study_runs(database_path: Path) -> None:
    suite = EvaluationSuite(
        name="study",
        version="1",
        cases=[
            EvaluationCase(
                id="reasoning",
                category="reasoning",
                prompt="Reason carefully.",
                expected="A reasoned response.",
                graders=[{"type": "exact"}],
            ),
            EvaluationCase(
                id="safety",
                category="safety",
                prompt="Respond safely.",
                expected="A safe response.",
                graders=[{"type": "exact"}],
            ),
        ],
    )
    database = Database(database_path)
    baseline = RunService(
        database,
        ReplayProvider(
            {"reasoning": "First answer.", "safety": "First safe answer."},
            model_names=("secret-model-a",),
        ),
    )
    candidate = RunService(
        database,
        ReplayProvider(
            {"reasoning": "Second answer.", "safety": "Second safe answer."},
            model_names=("secret-model-b",),
        ),
    )
    asyncio.run(baseline.run(suite, "secret-model-a", provider_name="replay", run_id="base"))
    asyncio.run(
        candidate.run(
            suite,
            "secret-model-b",
            provider_name="replay",
            run_id="candidate",
        )
    )


def test_suite_validate_reports_dimensions(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    write_suite(suite)

    result = runner.invoke(app, ["suite", "validate", str(suite)])

    assert result.exit_code == 0
    assert "Valid: tiny v1" in result.stdout
    assert "1 cases across 1 categories" in result.stdout


def test_suite_validate_explains_invalid_suite(tmp_path: Path) -> None:
    suite = tmp_path / "invalid.yaml"
    suite.write_text("name: invalid\nversion: '1'\ncases: []\n", encoding="utf-8")

    result = runner.invoke(app, ["suite", "validate", str(suite)])

    assert result.exit_code == 1
    assert "Invalid suite" in result.output


def test_replay_run_and_history_end_to_end(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    replay = tmp_path / "responses.json"
    database = tmp_path / "evalbench.db"
    write_suite(suite)
    replay.write_text(json.dumps({"one": "2"}), encoding="utf-8")

    run_result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "replay",
            "--suite",
            str(suite),
            "--replay",
            str(replay),
            "--database",
            str(database),
        ],
    )

    assert run_result.exit_code == 0
    completed = json.loads(run_result.stdout)
    assert completed["status"] == "completed"
    assert completed["accuracy"] == 1.0

    history_result = runner.invoke(app, ["runs", "--database", str(database)])
    assert history_result.exit_code == 0
    history = json.loads(history_result.stdout)
    assert [run["id"] for run in history] == [completed["id"]]


def test_replay_run_reports_missing_recording(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    replay = tmp_path / "responses.json"
    database = tmp_path / "evalbench.db"
    write_suite(suite)
    replay.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "replay",
            "--suite",
            str(suite),
            "--replay",
            str(replay),
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 1
    assert "No replay response is recorded for case 'one'" in result.output


def test_replay_requires_json_object(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    replay = tmp_path / "responses.json"
    write_suite(suite)
    replay.write_text("[]", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "replay",
            "--suite",
            str(suite),
            "--replay",
            str(replay),
        ],
    )

    assert result.exit_code == 1
    assert "Replay data must be a JSON object" in result.output


def test_committed_smoke_fixture_exercises_offline_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    database = tmp_path / "smoke.db"

    result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "replay",
            "--suite",
            str(project_root / "suites/smoke.yaml"),
            "--replay",
            str(project_root / "examples/smoke-responses.json"),
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 0
    completed = json.loads(result.stdout)
    assert completed["status"] == "completed"
    assert completed["completed_cases"] == 6
    assert completed["accuracy"] == 1.0


def test_replay_provider_requires_replay_path(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    write_suite(suite)

    result = runner.invoke(
        app,
        ["run", "--provider", "replay", "--suite", str(suite)],
    )

    assert result.exit_code == 1
    assert "Replay runs require --replay PATH" in result.output


def test_ollama_provider_rejects_replay_path(tmp_path: Path) -> None:
    suite = tmp_path / "tiny.yaml"
    replay = tmp_path / "responses.json"
    write_suite(suite)
    replay.write_text('{"one":"2"}', encoding="utf-8")

    result = runner.invoke(
        app,
        ["run", "--provider", "ollama", "--suite", str(suite), "--replay", str(replay)],
    )

    assert result.exit_code == 1
    assert "--replay can only be used with --provider replay" in result.output


def test_ollama_cli_run_uses_default_model_and_closes_client(tmp_path: Path, monkeypatch) -> None:
    class FakeOllamaProvider:
        schema_version = "fake-ollama-v1"
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.closed = True

        async def models(self) -> list[str]:
            return ["qwen3:0.6b"]

        async def preflight(self, model: str) -> None:
            assert model == "qwen3:0.6b"

        async def generate(self, model: str, case: EvaluationCase) -> Generation:
            return Generation(text="2", done_reason="stop")

    fake = FakeOllamaProvider()
    monkeypatch.setattr(cli_module, "OllamaProvider", lambda: fake)
    suite = tmp_path / "tiny.yaml"
    database = tmp_path / "evalbench.db"
    write_suite(suite)

    result = runner.invoke(
        app,
        ["run", "--provider", "ollama", "--suite", str(suite), "--database", str(database)],
    )

    assert result.exit_code == 0
    completed = json.loads(result.stdout)
    assert completed["model"] == "qwen3:0.6b"
    assert completed["provider"] == "ollama"
    assert fake.closed


def test_replay_cli_can_grade_with_an_independent_ollama_judge(tmp_path: Path, monkeypatch) -> None:
    class FakeJudgeProvider:
        schema_version = "fake-judge-v1"

        def __init__(self) -> None:
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.closed = True

        async def preflight(self, model: str) -> None:
            assert model == "gemma3:4b"

        async def structured_chat(
            self,
            model: str,
            *,
            messages: list[dict[str, str]],
            schema: dict[str, Any],
        ) -> dict[str, Any]:
            return {"score": 2, "rationale": "Fully correct."}

    fake = FakeJudgeProvider()
    monkeypatch.setattr(cli_module, "OllamaProvider", lambda: fake)
    suite = tmp_path / "judge.yaml"
    replay = tmp_path / "responses.json"
    database = tmp_path / "evalbench.db"
    write_judge_suite(suite)
    replay.write_text(json.dumps({"one": "A complete answer."}), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "replay",
            "--suite",
            str(suite),
            "--replay",
            str(replay),
            "--database",
            str(database),
            "--judge-model",
            "gemma3:4b",
        ],
    )

    assert result.exit_code == 0
    completed = json.loads(result.stdout)
    assert completed["judge_model"] == "gemma3:4b"
    assert completed["accuracy"] == 1
    assert Database(database).list_case_results(completed["id"])[0].result.judgments[0].score == 2
    assert fake.closed


def test_compare_command_outputs_paired_statistics(tmp_path: Path) -> None:
    database = tmp_path / "evalbench.db"
    suite = EvaluationSuite(
        name="tiny",
        version="1",
        cases=[
            EvaluationCase(
                id="one",
                category="reasoning",
                prompt="What is one plus one?",
                expected="2",
                graders=[{"type": "numeric"}],
            )
        ],
    )
    first = RunService(Database(database), ReplayProvider({"one": "2"}, model_names=("first",)))
    second = RunService(Database(database), ReplayProvider({"one": "3"}, model_names=("second",)))

    asyncio.run(first.run(suite, "first", provider_name="replay", run_id="base"))
    asyncio.run(second.run(suite, "second", provider_name="replay", run_id="candidate"))

    result = runner.invoke(
        app,
        ["compare", "base", "candidate", "--database", str(database)],
    )

    assert result.exit_code == 0
    comparison = json.loads(result.stdout)
    assert comparison["accuracy_delta"] == -1
    assert comparison["regressions"] == ["one"]
    assert comparison["improvements"] == []
    assert comparison["mcnemar"]["exact_p_value"] == 1


def test_compare_command_reports_invalid_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["compare", "missing", "also-missing", "--database", str(tmp_path / "empty.db")],
    )

    assert result.exit_code == 1
    assert "Baseline run 'missing' does not exist" in result.output


def test_judge_select_writes_a_balanced_study(tmp_path: Path) -> None:
    database = tmp_path / "evalbench.db"
    study_path = tmp_path / "judge-study.json"
    prepare_study_runs(database)

    result = runner.invoke(
        app,
        [
            "judge",
            "select",
            "base",
            "candidate",
            "--database",
            str(database),
            "--output",
            str(study_path),
            "--sample-size",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert "2 calibration and 2 holdout" in result.output
    study = load_study(study_path)
    assert study.sample_size == 4
    assert {example.model for example in study.examples} == {
        "secret-model-a",
        "secret-model-b",
    }


def test_judge_label_is_blinded_and_resumes_after_each_score(tmp_path: Path) -> None:
    database = tmp_path / "evalbench.db"
    study_path = tmp_path / "judge-study.json"
    prepare_study_runs(database)
    selected = runner.invoke(
        app,
        [
            "judge",
            "select",
            "base",
            "candidate",
            "--database",
            str(database),
            "--output",
            str(study_path),
            "--sample-size",
            "4",
        ],
    )
    assert selected.exit_code == 0

    first_session = runner.invoke(app, ["judge", "label", str(study_path)], input="2\nq\n")

    assert first_session.exit_code == 0
    assert "Progress saved at 1/2" in first_session.output
    assert "secret-model-a" not in first_session.output
    assert "secret-model-b" not in first_session.output
    calibration = [
        example for example in load_study(study_path).examples if example.split == "calibration"
    ]
    assert sum(example.human_score is not None for example in calibration) == 1

    resumed = runner.invoke(app, ["judge", "label", str(study_path)], input="1\n")

    assert resumed.exit_code == 0
    assert "Calibration labeling complete (2/2)" in resumed.output
    calibration = [
        example for example in load_study(study_path).examples if example.split == "calibration"
    ]
    assert sorted(example.human_score for example in calibration) == [1, 2]


def test_judge_label_blocks_holdout_until_rubric_is_frozen(tmp_path: Path) -> None:
    database = tmp_path / "evalbench.db"
    study_path = tmp_path / "judge-study.json"
    prepare_study_runs(database)
    selected = runner.invoke(
        app,
        [
            "judge",
            "select",
            "base",
            "candidate",
            "--database",
            str(database),
            "--output",
            str(study_path),
            "--sample-size",
            "4",
        ],
    )
    assert selected.exit_code == 0

    result = runner.invoke(
        app,
        ["judge", "label", str(study_path), "--split", "holdout"],
    )

    assert result.exit_code == 1
    assert "freeze the calibrated rubric" in result.output


def test_judge_score_and_report_commands_complete_calibration_study(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeJudgeProvider:
        schema_version = "cli-study-judge-v1"

        def __init__(self) -> None:
            self.closed = False
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.closed = True

        async def preflight(self, model: str) -> None:
            assert model == "gemma3:4b"

        async def structured_chat(
            self,
            model: str,
            *,
            messages: list[dict[str, str]],
            schema: dict[str, Any],
        ) -> dict[str, Any]:
            score = (0, 2)[self.calls]
            self.calls += 1
            return {"score": score, "rationale": f"Fixture score {score}."}

    database = tmp_path / "evalbench.db"
    study_path = tmp_path / "judge-study.json"
    prepare_study_runs(database)
    selected = runner.invoke(
        app,
        [
            "judge",
            "select",
            "base",
            "candidate",
            "--database",
            str(database),
            "--output",
            str(study_path),
            "--sample-size",
            "4",
        ],
    )
    assert selected.exit_code == 0
    labeled = runner.invoke(
        app,
        ["judge", "label", str(study_path)],
        input="0\n1\n",
    )
    assert labeled.exit_code == 0

    fake = FakeJudgeProvider()
    monkeypatch.setattr(cli_module, "OllamaProvider", lambda: fake)
    scored = runner.invoke(
        app,
        [
            "judge",
            "score",
            str(study_path),
            "--judge-model",
            "gemma3:4b",
            "--database",
            str(database),
        ],
    )

    assert scored.exit_code == 0
    assert "Calibration judge scoring complete (2/2)" in scored.output
    assert fake.calls == 2
    assert fake.closed
    stored = load_study(study_path)
    assert sorted(
        example.judge_score for example in stored.examples if example.split == "calibration"
    ) == [0, 2]

    report_path = tmp_path / "calibration-report.json"
    reported = runner.invoke(
        app,
        [
            "judge",
            "report",
            str(study_path),
            "--bootstrap-samples",
            "100",
            "--seed",
            "7",
            "--output",
            str(report_path),
        ],
    )

    assert reported.exit_code == 0
    report = json.loads(reported.stdout)
    assert report["split"] == "calibration"
    assert report["example_count"] == 2
    assert report["bootstrap_samples"] == 100
    assert sum(sum(row) for row in report["confusion_matrix"]) == 2
    assert "one labeler" in report["limitations"][0]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    frozen_rubric = study_module.load_rubric().model_copy(
        update={"version": "1.0.0", "status": "frozen"}
    )
    monkeypatch.setattr(study_module, "load_rubric", lambda: frozen_rubric)
    freeze = runner.invoke(app, ["judge", "freeze", str(study_path)])

    assert freeze.exit_code == 0
    assert "preserved 2 calibration pairs and unlocked 2 holdout examples" in freeze.output
    frozen_study = load_study(study_path)
    assert frozen_study.rubric_status == "frozen"
    assert frozen_study.calibration_rubric is not None

    holdout = runner.invoke(
        app,
        ["judge", "label", str(study_path), "--split", "holdout"],
        input="q\n",
    )
    assert holdout.exit_code == 0
    assert "Labeling holdout: 0/2 complete" in holdout.output


def test_serve_binds_only_to_localhost(tmp_path: Path, monkeypatch) -> None:
    class State:
        frontend_mounted = False

    class FakeApp:
        state = State()

    fake_app = FakeApp()
    created_with: dict[str, object] = {}
    served_with: dict[str, object] = {}

    def fake_create_app(**kwargs):
        created_with.update(kwargs)
        return fake_app

    def fake_run(application, **kwargs):
        served_with["application"] = application
        served_with.update(kwargs)

    monkeypatch.setattr(cli_module, "create_app", fake_create_app)
    monkeypatch.setattr(cli_module.uvicorn, "run", fake_run)
    database = tmp_path / "evalbench.db"
    suites = tmp_path / "suites"
    frontend = tmp_path / "dist"

    result = runner.invoke(
        app,
        [
            "serve",
            "--database",
            str(database),
            "--suites",
            str(suites),
            "--frontend",
            str(frontend),
            "--port",
            "8123",
        ],
    )

    assert result.exit_code == 0
    assert created_with == {
        "database_path": database,
        "suite_directory": suites,
        "frontend_directory": frontend,
    }
    assert served_with == {"application": fake_app, "host": "127.0.0.1", "port": 8123}
    assert "http://127.0.0.1:8123" in result.output
    assert "Frontend build not found" in result.output


def test_serve_rejects_invalid_port() -> None:
    result = runner.invoke(app, ["serve", "--port", "70000"])

    assert result.exit_code == 2
    assert "65535" in result.output
