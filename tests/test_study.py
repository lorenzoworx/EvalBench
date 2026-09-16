from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalbench.judge import load_rubric
from evalbench.models import (
    CaseResult,
    EvaluationCase,
    Generation,
    Judgment,
    RunSummary,
)
from evalbench.store import Database
from evalbench.study import (
    StudyValidationError,
    build_judge_study,
    load_study,
    record_human_score,
    save_study,
    validate_study_rubric,
)

CATEGORIES = ("reasoning", "structured", "instruction", "safety")


def study_database(tmp_path: Path, *, same_model: bool = False) -> Database:
    database = Database(tmp_path / "study.db")
    database.initialize()
    cases = [
        EvaluationCase(
            id=f"{category}-{index:02d}",
            category=category,
            prompt=f"Prompt {category} {index}",
            expected=f"Expected {category} {index}",
            graders=[{"type": "exact"}],
        )
        for category in CATEGORIES
        for index in range(15)
    ]
    for run_id, model in (
        ("baseline", "qwen3:0.6b"),
        ("candidate", "qwen3:0.6b" if same_model else "qwen3:4b"),
    ):
        database.create_run(
            RunSummary(
                id=run_id,
                suite_name="Core Assistant",
                suite_version="1.0",
                model=model,
                provider="replay",
                status="completed",
                completed_cases=len(cases),
                total_cases=len(cases),
                accuracy=1,
            )
        )
        for case in cases:
            database.save_case_result(
                run_id,
                case,
                CaseResult(
                    case_id=case.id,
                    category=case.category,
                    prompt=case.prompt,
                    expected=case.expected,
                    generation=Generation(text=f"{model} answer to {case.id}"),
                    judgments=[
                        Judgment(
                            grader_type="exact",
                            passed=True,
                            score=1,
                            rationale="Fixture pass.",
                        )
                    ],
                    passed=True,
                ),
            )
    return database


def test_selection_is_reproducible_and_balanced_before_labeling(tmp_path: Path) -> None:
    database = study_database(tmp_path)

    study = build_judge_study(database, "baseline", "candidate", seed=17)
    repeated = build_judge_study(database, "baseline", "candidate", seed=17)

    assert study == repeated
    assert len(study.examples) == 60
    assert Counter(example.model for example in study.examples) == {
        "qwen3:0.6b": 30,
        "qwen3:4b": 30,
    }
    assert Counter(example.category for example in study.examples) == {
        category: 15 for category in CATEGORIES
    }
    assert Counter(example.split for example in study.examples) == {
        "calibration": 30,
        "holdout": 30,
    }
    assert Counter((example.split, example.model) for example in study.examples) == {
        ("calibration", "qwen3:0.6b"): 15,
        ("calibration", "qwen3:4b"): 15,
        ("holdout", "qwen3:0.6b"): 15,
        ("holdout", "qwen3:4b"): 15,
    }
    stratum_counts = Counter((example.category, example.model) for example in study.examples)
    assert set(stratum_counts.values()) == {7, 8}
    assert all(example.human_score is None for example in study.examples)


def test_selection_rejects_unbalanced_inputs(tmp_path: Path) -> None:
    database = study_database(tmp_path, same_model=True)

    with pytest.raises(StudyValidationError, match="two different models"):
        build_judge_study(database, "baseline", "candidate")
    with pytest.raises(StudyValidationError, match="divisible by four"):
        build_judge_study(database, "baseline", "candidate", sample_size=62)


def test_label_progress_round_trips_atomically_and_checks_rubric(tmp_path: Path) -> None:
    study = build_judge_study(study_database(tmp_path), "baseline", "candidate")
    first = study.examples[0]
    labeled = record_human_score(
        study,
        first.sample_id,
        2,
        labeled_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    path = tmp_path / "labels" / "study.json"

    save_study(path, labeled)
    loaded = load_study(path)

    assert loaded == labeled
    assert not path.with_suffix(".json.tmp").exists()
    assert loaded.examples[0].human_score == 2
    assert validate_study_rubric(loaded) == load_rubric()
    changed = load_rubric().model_copy(update={"version": "changed"})
    with pytest.raises(StudyValidationError, match="provenance"):
        validate_study_rubric(loaded, changed)
    with pytest.raises(StudyValidationError, match="does not exist"):
        record_human_score(study, "missing", 1)
