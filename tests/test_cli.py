import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

import evalbench.cli as cli_module
from evalbench.cli import app
from evalbench.models import EvaluationCase, EvaluationSuite, Generation
from evalbench.providers import ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database

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
