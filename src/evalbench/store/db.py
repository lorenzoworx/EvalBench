from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from evalbench.models import CaseResult, EvaluationCase, Generation, Judgment, RunSummary


class StoredCaseResult(NamedTuple):
    case: EvaluationCase
    result: CaseResult


class Database:
    """Synchronous SQLite persistence with case-level atomic writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as connection:
            connection.executescript(schema_path.read_text(encoding="utf-8"))

    def create_run(self, run: RunSummary) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO runs (
                    id, suite_name, suite_version, model, provider, status,
                    created_at, completed_at, completed_cases, total_cases,
                    accuracy, ci_low, ci_high, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._run_values(run),
            )

    def update_run(self, run: RunSummary) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE runs SET
                    suite_name=?, suite_version=?, model=?, provider=?, status=?,
                    created_at=?, completed_at=?, completed_cases=?, total_cases=?,
                    accuracy=?, ci_low=?, ci_high=?, error=?
                WHERE id=?""",
                (*self._run_values(run)[1:], run.id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Run {run.id!r} does not exist.")

    def get_run(self, run_id: str) -> RunSummary | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else RunSummary.model_validate(dict(row))

    def list_runs(self) -> list[RunSummary]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [RunSummary.model_validate(dict(row)) for row in rows]

    def save_case_result(self, run_id: str, case: EvaluationCase, result: CaseResult) -> None:
        if result.case_id != case.id or result.category != case.category:
            raise ValueError("Result identity must match its case snapshot.")
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO case_results (
                    run_id, case_id, category, case_snapshot, generation, passed
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, case_id) DO UPDATE SET
                    category=excluded.category,
                    case_snapshot=excluded.case_snapshot,
                    generation=excluded.generation,
                    passed=excluded.passed""",
                (
                    run_id,
                    case.id,
                    case.category,
                    case.model_dump_json(by_alias=True),
                    result.generation.model_dump_json(),
                    int(result.passed),
                ),
            )
            connection.execute(
                "DELETE FROM judgments WHERE run_id=? AND case_id=?", (run_id, case.id)
            )
            connection.executemany(
                """INSERT INTO judgments (
                    run_id, case_id, grader_index, grader_type,
                    passed, score, rationale, is_primary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        case.id,
                        index,
                        judgment.grader_type,
                        int(judgment.passed),
                        judgment.score,
                        judgment.rationale,
                        int(judgment.primary),
                    )
                    for index, judgment in enumerate(result.judgments)
                ],
            )

    def list_case_results(self, run_id: str) -> list[StoredCaseResult]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT case_id, case_snapshot, generation, passed
                FROM case_results WHERE run_id=? ORDER BY case_id""",
                (run_id,),
            ).fetchall()
            judgment_rows = connection.execute(
                """SELECT case_id, grader_type, passed, score, rationale, is_primary
                FROM judgments WHERE run_id=? ORDER BY case_id, grader_index""",
                (run_id,),
            ).fetchall()

        judgments_by_case: dict[str, list[Judgment]] = {}
        for row in judgment_rows:
            judgments_by_case.setdefault(row["case_id"], []).append(
                Judgment(
                    grader_type=row["grader_type"],
                    passed=bool(row["passed"]),
                    score=row["score"],
                    rationale=row["rationale"],
                    primary=bool(row["is_primary"]),
                )
            )

        records: list[StoredCaseResult] = []
        for row in rows:
            case = EvaluationCase.model_validate_json(row["case_snapshot"])
            records.append(
                StoredCaseResult(
                    case=case,
                    result=CaseResult(
                        case_id=case.id,
                        category=case.category,
                        prompt=case.prompt,
                        expected=case.expected,
                        generation=Generation.model_validate_json(row["generation"]),
                        judgments=judgments_by_case.get(case.id, []),
                        passed=bool(row["passed"]),
                    ),
                )
            )
        return records

    @staticmethod
    def _run_values(run: RunSummary) -> tuple[object, ...]:
        return (
            run.id,
            run.suite_name,
            run.suite_version,
            run.model,
            run.provider,
            run.status,
            run.created_at.isoformat(),
            run.completed_at.isoformat() if run.completed_at else None,
            run.completed_cases,
            run.total_cases,
            run.accuracy,
            run.ci_low,
            run.ci_high,
            run.error,
        )
