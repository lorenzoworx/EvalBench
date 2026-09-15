import json
from pathlib import Path

from typer.testing import CliRunner

from evalbench.cli import app

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
        ["run", "--suite", str(suite), "--replay", str(replay)],
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
