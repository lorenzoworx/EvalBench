import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalbench.models import CaseResult, EvaluationCase, Generation, Judgment, RunSummary
from evalbench.store import Database


def make_run(run_id: str = "run-1", *, created_at: datetime | None = None) -> RunSummary:
    return RunSummary(
        id=run_id,
        suite_name="tiny",
        suite_version="1",
        model="replay",
        provider="replay",
        status="pending",
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        total_cases=1,
    )


def make_case() -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        category="reasoning",
        prompt="What is one plus one?",
        expected="2",
        graders=[{"type": "numeric"}],
    )


def make_result(*, answer: str = "2", passed: bool = True) -> CaseResult:
    case = make_case()
    return CaseResult(
        case_id=case.id,
        category=case.category,
        prompt=case.prompt,
        expected=case.expected,
        generation=Generation(
            text=answer,
            done_reason="stop",
            prompt_tokens=7,
            output_tokens=1,
            total_duration_ns=2_000_000,
        ),
        judgments=[
            Judgment(
                grader_type="numeric",
                passed=passed,
                score=float(passed),
                rationale="Matched expected number.",
            )
        ],
        passed=passed,
    )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "nested" / "evalbench.db")
    value.initialize()
    return value


def test_initialize_creates_wal_database(database: Database) -> None:
    assert database.path.exists()
    with database.connect() as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert mode == "wal"
    assert {"runs", "case_results", "judgments", "generation_cache"} <= tables


def test_create_get_update_and_list_runs(database: Database) -> None:
    older = make_run()
    newer = make_run("run-2", created_at=datetime(2026, 1, 2, tzinfo=UTC))
    database.create_run(older)
    database.create_run(newer)

    completed = older.model_copy(
        update={
            "status": "completed",
            "completed_at": datetime(2026, 1, 1, 1, tzinfo=UTC),
            "completed_cases": 1,
            "accuracy": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
        }
    )
    database.update_run(completed)

    assert database.get_run("run-1") == completed
    assert [run.id for run in database.list_runs()] == ["run-2", "run-1"]
    assert database.get_run("missing") is None


def test_interrupt_unfinished_runs_preserves_terminal_history(database: Database) -> None:
    pending = make_run("pending")
    running = make_run("running").model_copy(update={"status": "running"})
    completed = make_run("completed").model_copy(update={"status": "completed"})
    for run in (pending, running, completed):
        database.create_run(run)

    assert database.interrupt_unfinished_runs() == 2
    assert database.get_run("pending").status == "interrupted"
    interrupted = database.get_run("running")
    assert interrupted is not None
    assert interrupted.status == "interrupted"
    assert interrupted.completed_at is not None
    assert interrupted.error == "Server stopped before the run completed."
    assert database.get_run("completed") == completed


def test_update_rejects_unknown_run(database: Database) -> None:
    with pytest.raises(KeyError, match="missing"):
        database.update_run(make_run("missing"))


def test_case_snapshot_generation_and_judgments_round_trip(database: Database) -> None:
    run = make_run()
    case = make_case()
    result = make_result()
    database.create_run(run)

    database.save_case_result(run.id, case, result)

    assert database.list_case_results(run.id) == [(case, result)]


def test_replacing_result_replaces_judgments_atomically(database: Database) -> None:
    run = make_run()
    case = make_case()
    database.create_run(run)
    database.save_case_result(run.id, case, make_result())

    replacement = make_result(answer="3", passed=False)
    database.save_case_result(run.id, case, replacement)

    assert database.list_case_results(run.id) == [(case, replacement)]


def test_result_identity_must_match_snapshot(database: Database) -> None:
    run = make_run()
    database.create_run(run)
    result = make_result().model_copy(update={"case_id": "other"})

    with pytest.raises(ValueError, match="identity"):
        database.save_case_result(run.id, make_case(), result)


def test_foreign_keys_prevent_orphaned_results(database: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        database.save_case_result("missing", make_case(), make_result())
