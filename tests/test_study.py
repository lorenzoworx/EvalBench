import asyncio
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evalbench.judge import JudgeEvaluator, JudgeResponse, load_rubric
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
    build_agreement_report,
    build_judge_study,
    freeze_study_rubric,
    load_study,
    record_human_score,
    record_judge_response,
    save_study,
    score_study_split,
    validate_study_rubric,
    validate_study_split,
)

CATEGORIES = ("reasoning", "structured", "instruction", "safety")


class StudyJudgeProvider:
    schema_version = "study-judge-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.preflight_models: list[str] = []

    async def preflight(self, model: str) -> None:
        self.preflight_models.append(model)

    async def structured_chat(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        return {"score": 2, "rationale": "Fully correct."}


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


def complete_calibration(study):
    calibration = [example for example in study.examples if example.split == "calibration"]
    for index, example in enumerate(calibration):
        study = record_human_score(study, example.sample_id, index % 3)
        study = record_judge_response(
            study,
            example.sample_id,
            JudgeResponse(score=index % 3, rationale="Fixture score."),
            judge_model="gemma3:4b",
            provider_schema_version="study-judge-v1",
        )
    return study


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


def test_study_scoring_is_resumable_cached_and_independent(tmp_path: Path) -> None:
    database = study_database(tmp_path)
    study = build_judge_study(database, "baseline", "candidate", sample_size=4)
    provider = StudyJudgeProvider()
    evaluator = JudgeEvaluator(provider)
    progress: list[tuple[int, int]] = []

    scored = asyncio.run(
        score_study_split(
            study,
            "calibration",
            database=database,
            evaluator=evaluator,
            judge_model="gemma3:4b",
            on_progress=lambda _study, completed, total: progress.append((completed, total)),
        )
    )

    assert provider.calls == 2
    assert provider.preflight_models == ["gemma3:4b"]
    assert progress == [(1, 2), (2, 2)]
    assert scored.judge_model == "gemma3:4b"
    assert scored.judge_provider_schema_version == "study-judge-v1"
    assert all(
        example.judge_score == 2 for example in scored.examples if example.split == "calibration"
    )
    assert all(
        example.judge_score is None for example in scored.examples if example.split == "holdout"
    )

    cached_copy = asyncio.run(
        score_study_split(
            study,
            "calibration",
            database=database,
            evaluator=evaluator,
            judge_model="gemma3:4b",
        )
    )
    assert provider.calls == 2
    assert [example.judge_score for example in cached_copy.examples] == [
        example.judge_score for example in scored.examples
    ]
    assert cached_copy.judge_model == scored.judge_model

    with pytest.raises(StudyValidationError, match="was scored with judge model"):
        asyncio.run(
            score_study_split(
                scored,
                "calibration",
                database=database,
                evaluator=evaluator,
                judge_model="other-judge",
            )
        )

    with pytest.raises(StudyValidationError, match="independent"):
        asyncio.run(
            score_study_split(
                study,
                "calibration",
                database=database,
                evaluator=evaluator,
                judge_model="qwen3:0.6b",
            )
        )


def test_agreement_report_is_reproducible_and_split_specific(tmp_path: Path) -> None:
    study = build_judge_study(study_database(tmp_path), "baseline", "candidate")
    calibration = [example for example in study.examples if example.split == "calibration"]
    human_pattern = [0, 0, 1, 1, 2, 2]
    judge_pattern = [0, 1, 1, 2, 2, 2]
    for index, example in enumerate(calibration):
        study = record_human_score(
            study,
            example.sample_id,
            human_pattern[index % len(human_pattern)],
        )
        study = record_judge_response(
            study,
            example.sample_id,
            JudgeResponse(
                score=judge_pattern[index % len(judge_pattern)],
                rationale="Fixture score.",
            ),
            judge_model="gemma3:4b",
            provider_schema_version="study-judge-v1",
        )

    report = build_agreement_report(
        study,
        "calibration",
        bootstrap_samples=500,
        bootstrap_seed=9,
    )
    repeated = build_agreement_report(
        study,
        "calibration",
        bootstrap_samples=500,
        bootstrap_seed=9,
    )

    assert report == repeated
    assert report.example_count == 30
    assert report.raw_agreement == pytest.approx(2 / 3)
    assert report.quadratic_weighted_kappa == pytest.approx(0.75)
    assert report.confusion_labels == [0, 1, 2]
    assert report.confusion_matrix == [[5, 5, 0], [0, 5, 5], [0, 0, 10]]
    assert set(report.category_breakdown) == set(CATEGORIES)
    assert report.kappa_ci_low is not None
    assert report.kappa_ci_high is not None
    assert "one labeler" in report.limitations[0]


def test_agreement_report_requires_complete_scores_and_seals_holdout(tmp_path: Path) -> None:
    study = build_judge_study(study_database(tmp_path), "baseline", "candidate")

    with pytest.raises(StudyValidationError, match="complete scores"):
        build_agreement_report(study, "calibration", bootstrap_samples=10)
    with pytest.raises(StudyValidationError, match="freeze the calibrated rubric"):
        build_agreement_report(study, "holdout", bootstrap_samples=10)


def test_freeze_preserves_calibration_provenance_and_unlocks_holdout(tmp_path: Path) -> None:
    study = complete_calibration(
        build_judge_study(study_database(tmp_path), "baseline", "candidate", sample_size=4)
    )
    calibration_hash = study.rubric_content_sha256
    frozen_rubric = load_rubric().model_copy(update={"version": "1.0.0", "status": "frozen"})

    frozen = freeze_study_rubric(study, frozen_rubric)

    assert frozen.rubric_status == "frozen"
    assert frozen.rubric_version == "1.0.0"
    assert frozen.rubric_content_sha256 != calibration_hash
    assert frozen.calibration_rubric is not None
    assert frozen.calibration_rubric.rubric_status == "calibration"
    assert frozen.calibration_rubric.rubric_content_sha256 == calibration_hash
    assert [example.human_score for example in frozen.examples] == [
        example.human_score for example in study.examples
    ]
    assert [example.judge_score for example in frozen.examples] == [
        example.judge_score for example in study.examples
    ]
    assert validate_study_split(frozen, "holdout", frozen_rubric) == frozen_rubric
    assert freeze_study_rubric(frozen, frozen_rubric) == frozen

    calibration_report = build_agreement_report(
        frozen,
        "calibration",
        bootstrap_samples=10,
        rubric=frozen_rubric,
    )
    assert calibration_report.rubric_status == "calibration"
    assert calibration_report.rubric_content_sha256 == calibration_hash

    completed_holdout = frozen
    for example in [item for item in frozen.examples if item.split == "holdout"]:
        completed_holdout = record_human_score(completed_holdout, example.sample_id, 2)
        completed_holdout = record_judge_response(
            completed_holdout,
            example.sample_id,
            JudgeResponse(score=2, rationale="Fixture holdout score."),
            judge_model="gemma3:4b",
            provider_schema_version="study-judge-v1",
        )
    holdout_report = build_agreement_report(
        completed_holdout,
        "holdout",
        bootstrap_samples=10,
        rubric=frozen_rubric,
    )
    assert holdout_report.rubric_status == "frozen"
    assert holdout_report.rubric_content_sha256 == frozen.rubric_content_sha256


def test_freeze_requires_complete_calibration_and_untouched_holdout(tmp_path: Path) -> None:
    study = build_judge_study(study_database(tmp_path), "baseline", "candidate", sample_size=4)
    calibration_rubric = load_rubric()
    frozen_rubric = calibration_rubric.model_copy(update={"version": "1.0.0", "status": "frozen"})

    with pytest.raises(StudyValidationError, match="status 'frozen'"):
        freeze_study_rubric(study, calibration_rubric)
    with pytest.raises(StudyValidationError, match="calibration is complete"):
        freeze_study_rubric(study, frozen_rubric)
    with pytest.raises(StudyValidationError, match="rubric ID"):
        freeze_study_rubric(
            study,
            frozen_rubric.model_copy(update={"id": "different_v1"}),
        )

    completed = complete_calibration(study)
    first_holdout = next(example for example in completed.examples if example.split == "holdout")
    touched = record_human_score(completed, first_holdout.sample_id, 1)
    with pytest.raises(StudyValidationError, match="after holdout scores exist"):
        freeze_study_rubric(touched, frozen_rubric)
