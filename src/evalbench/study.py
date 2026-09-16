from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalbench.cache import judge_request_hash, rubric_content_hash
from evalbench.judge import JudgeEvaluator, JudgeResponse, JudgeRubric, load_rubric
from evalbench.metrics import bootstrap_kappa_ci, ordinal_confusion_matrix, quadratic_weighted_kappa
from evalbench.models import EvaluationCase
from evalbench.service import RunService
from evalbench.store import Database, StoredCaseResult

StudySplit = Literal["calibration", "holdout"]


class StudyValidationError(ValueError):
    """Raised when a judge-study artifact or source runs are invalid."""


class StudyRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model: str


class RubricProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    rubric_version: str
    rubric_status: Literal["calibration", "frozen"]
    rubric_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StudyExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    split: StudySplit
    run_id: str
    model: str
    case_id: str
    category: str
    prompt: str
    expected: Any | None
    answer: str
    human_score: Literal[0, 1, 2] | None = None
    labeled_at: datetime | None = None
    judge_score: Literal[0, 1, 2] | None = None
    judge_rationale: str | None = None
    judged_at: datetime | None = None


class JudgeStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judge-study-v1"] = "judge-study-v1"
    rubric_id: str
    rubric_version: str
    rubric_status: Literal["calibration", "frozen"] = "calibration"
    rubric_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_rubric: RubricProvenance | None = None
    seed: int
    sample_size: int = Field(gt=0)
    source_runs: list[StudyRun]
    examples: list[StudyExample]
    judge_model: str | None = None
    judge_provider_schema_version: str | None = None

    @model_validator(mode="after")
    def consistent_manifest(self) -> JudgeStudy:
        if len(self.source_runs) != 2:
            raise ValueError("judge study requires exactly two source runs")
        if len(self.examples) != self.sample_size:
            raise ValueError("sample_size must equal the number of examples")
        if len({example.sample_id for example in self.examples}) != len(self.examples):
            raise ValueError("study sample IDs must be unique")
        source_by_id = {run.run_id: run.model for run in self.source_runs}
        if len(source_by_id) != 2 or len(set(source_by_id.values())) != 2:
            raise ValueError("source runs and models must be unique")
        source_ids = set(source_by_id)
        if any(example.run_id not in source_ids for example in self.examples):
            raise ValueError("every example must reference a declared source run")
        if any(example.model != source_by_id[example.run_id] for example in self.examples):
            raise ValueError("example model provenance must match its source run")
        calibration = sum(example.split == "calibration" for example in self.examples)
        if calibration * 2 != self.sample_size:
            raise ValueError("study must have equal calibration and holdout splits")
        if any(
            sum(example.run_id == run_id for example in self.examples) * 2 != self.sample_size
            for run_id in source_ids
        ):
            raise ValueError("study must have equal examples from both source models")
        if any(
            sum(example.run_id == run_id and example.split == split for example in self.examples)
            * 4
            != self.sample_size
            for run_id in source_ids
            for split in ("calibration", "holdout")
        ):
            raise ValueError("each split must balance both source models")
        category_counts = {
            category: sum(example.category == category for example in self.examples)
            for category in {example.category for example in self.examples}
        }
        if not category_counts or len(set(category_counts.values())) != 1:
            raise ValueError("study must balance examples across categories")
        if any(
            (example.human_score is None) != (example.labeled_at is None)
            for example in self.examples
        ):
            raise ValueError("human score and labeled timestamp must be recorded together")
        if (self.judge_model is None) != (self.judge_provider_schema_version is None):
            raise ValueError("judge model and provider schema provenance must be recorded together")
        if any(
            len(
                {
                    example.judge_score is None,
                    example.judge_rationale is None,
                    example.judged_at is None,
                }
            )
            != 1
            for example in self.examples
        ):
            raise ValueError("judge score, rationale, and timestamp must be recorded together")
        has_judge_scores = any(example.judge_score is not None for example in self.examples)
        if has_judge_scores and not self.judge_model:
            raise ValueError("scored examples require judge provenance")
        if self.calibration_rubric is not None:
            if self.calibration_rubric.rubric_status != "calibration":
                raise ValueError("saved calibration provenance must describe a calibration rubric")
            if self.calibration_rubric.rubric_id != self.rubric_id:
                raise ValueError("calibration and frozen rubric IDs must match")
            if self.rubric_status != "frozen":
                raise ValueError("saved calibration provenance requires a frozen active rubric")
        if self.rubric_status == "frozen" and self.calibration_rubric is None:
            raise ValueError("a frozen study must preserve its calibration rubric provenance")
        return self


class CategoryAgreement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_count: int = Field(gt=0)
    raw_agreement: float = Field(ge=0, le=1)
    quadratic_weighted_kappa: float | None = Field(default=None, ge=-1, le=1)
    confusion_matrix: list[list[int]]


class StudyAgreementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judge-report-v1"] = "judge-report-v1"
    study_schema_version: Literal["judge-study-v1"]
    split: StudySplit
    rubric_id: str
    rubric_version: str
    rubric_status: Literal["calibration", "frozen"]
    rubric_content_sha256: str
    judge_model: str
    source_runs: list[StudyRun]
    example_count: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    bootstrap_seed: int
    confidence: float = Field(gt=0, lt=1)
    raw_agreement: float = Field(ge=0, le=1)
    quadratic_weighted_kappa: float | None = Field(default=None, ge=-1, le=1)
    kappa_ci_low: float | None = Field(default=None, ge=-1, le=1)
    kappa_ci_high: float | None = Field(default=None, ge=-1, le=1)
    confusion_labels: list[Literal[0, 1, 2]]
    confusion_matrix: list[list[int]]
    category_breakdown: dict[str, CategoryAgreement]
    limitations: list[str]


def build_judge_study(
    database: Database,
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    sample_size: int = 60,
    seed: int = 42,
    rubric: JudgeRubric | None = None,
) -> JudgeStudy:
    """Select model- and category-balanced outputs, then split before labeling."""
    if sample_size <= 0 or sample_size % 4:
        raise StudyValidationError("sample size must be positive and divisible by four")
    RunService(database).compare(baseline_run_id, candidate_run_id)
    source_runs = [database.get_run(baseline_run_id), database.get_run(candidate_run_id)]
    if any(run is None for run in source_runs):
        raise AssertionError("comparison validation guarantees both runs exist")
    baseline, candidate = source_runs
    if baseline is None or candidate is None:
        raise AssertionError("source runs were validated above")
    if baseline.model == candidate.model:
        raise StudyValidationError("judge study source runs must use two different models")

    records_by_run = {
        baseline.id: database.list_case_results(baseline.id),
        candidate.id: database.list_case_results(candidate.id),
    }
    categories = sorted({record.case.category for record in records_by_run[baseline.id]})
    if not categories or sample_size % len(categories):
        raise StudyValidationError(
            "sample size must divide evenly across the source suite's categories"
        )

    per_category = sample_size // len(categories)
    per_model = sample_size // 2
    lower_per_stratum = per_category // 2
    baseline_extra_count = per_model - lower_per_stratum * len(categories)
    if not 0 <= baseline_extra_count <= len(categories):
        raise StudyValidationError("sample cannot be balanced across both models and categories")

    rng = random.Random(seed)
    extra_categories = categories.copy()
    rng.shuffle(extra_categories)
    baseline_extra = set(extra_categories[:baseline_extra_count])
    selected: list[tuple[str, StoredCaseResult]] = []
    for category in categories:
        for index, run in enumerate((baseline, candidate)):
            count = lower_per_stratum
            if per_category % 2:
                gets_extra = category in baseline_extra
                if (index == 0 and gets_extra) or (index == 1 and not gets_extra):
                    count += 1
            available = [
                record for record in records_by_run[run.id] if record.case.category == category
            ]
            if len(available) < count:
                raise StudyValidationError(
                    f"Run {run.id!r} has only {len(available)} {category!r} outputs; "
                    f"{count} required."
                )
            rng.shuffle(available)
            selected.extend((run.id, record) for record in available[:count])

    by_stratum: dict[tuple[str, str], list[int]] = {}
    for index, (run_id, record) in enumerate(selected):
        by_stratum.setdefault((run_id, record.case.category), []).append(index)
    calibration_target = sample_size // 2
    calibration_indices: set[int] = set()
    odd_strata_by_run: dict[str, list[tuple[str, str]]] = {}
    for stratum, indices in sorted(by_stratum.items()):
        rng.shuffle(indices)
        calibration_indices.update(indices[: len(indices) // 2])
        if len(indices) % 2:
            odd_strata_by_run.setdefault(stratum[0], []).append(stratum)
    for run in (baseline, candidate):
        odd_strata = odd_strata_by_run.get(run.id, [])
        rng.shuffle(odd_strata)
        current = sum(
            index in calibration_indices
            for stratum, indices in by_stratum.items()
            if stratum[0] == run.id
            for index in indices
        )
        needed = per_model // 2 - current
        for stratum in odd_strata[:needed]:
            indices = by_stratum[stratum]
            calibration_indices.add(indices[len(indices) // 2])
    if len(calibration_indices) != calibration_target:
        raise StudyValidationError("could not create equal balanced study splits")

    models = {baseline.id: baseline.model, candidate.id: candidate.model}
    examples = []
    for index, (run_id, record) in enumerate(selected):
        sample_key = f"{run_id}\0{record.case.id}".encode()
        examples.append(
            StudyExample(
                sample_id=hashlib.sha256(sample_key).hexdigest()[:16],
                split="calibration" if index in calibration_indices else "holdout",
                run_id=run_id,
                model=models[run_id],
                case_id=record.case.id,
                category=record.case.category,
                prompt=record.case.prompt,
                expected=record.case.expected,
                answer=record.result.generation.text,
            )
        )
    rng.shuffle(examples)
    selected_rubric = rubric or load_rubric()
    if selected_rubric.status != "calibration":
        raise StudyValidationError("a new judge study must start with a calibration rubric")
    return JudgeStudy(
        rubric_id=selected_rubric.id,
        rubric_version=selected_rubric.version,
        rubric_status=selected_rubric.status,
        rubric_content_sha256=rubric_content_hash(selected_rubric),
        seed=seed,
        sample_size=sample_size,
        source_runs=[
            StudyRun(run_id=baseline.id, model=baseline.model),
            StudyRun(run_id=candidate.id, model=candidate.model),
        ],
        examples=examples,
    )


def load_study(path: str | Path) -> JudgeStudy:
    study_path = Path(path)
    try:
        raw = json.loads(study_path.read_text(encoding="utf-8"))
        return JudgeStudy.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise StudyValidationError(f"Cannot load judge study {study_path}: {exc}") from exc


def save_study(path: str | Path, study: JudgeStudy) -> None:
    """Atomically persist progress so an interrupted labeling session can resume."""
    study_path = Path(path)
    study_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = study_path.with_suffix(f"{study_path.suffix}.tmp")
    temporary.write_text(study.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(study_path)


def save_agreement_report(path: str | Path, report: StudyAgreementReport) -> None:
    """Atomically persist a reproducible agreement report."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(f"{report_path.suffix}.tmp")
    temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def validate_study_rubric(study: JudgeStudy, rubric: JudgeRubric | None = None) -> JudgeRubric:
    selected = rubric or load_rubric()
    if (
        study.rubric_id,
        study.rubric_version,
        study.rubric_status,
        study.rubric_content_sha256,
    ) != (selected.id, selected.version, selected.status, rubric_content_hash(selected)):
        raise StudyValidationError("study rubric provenance does not match the current rubric")
    return selected


def freeze_study_rubric(
    study: JudgeStudy,
    rubric: JudgeRubric | None = None,
) -> JudgeStudy:
    """Adopt a frozen rubric after complete calibration without touching holdout data."""
    selected = rubric or load_rubric()
    if selected.status != "frozen":
        raise StudyValidationError("current rubric must have status 'frozen'")
    if study.rubric_status == "frozen":
        validate_study_rubric(study, selected)
        return study
    if selected.id != study.rubric_id:
        raise StudyValidationError("frozen rubric ID must match the calibration rubric ID")

    calibration = [example for example in study.examples if example.split == "calibration"]
    missing_human = sum(example.human_score is None for example in calibration)
    missing_judge = sum(example.judge_score is None for example in calibration)
    if missing_human or missing_judge:
        raise StudyValidationError(
            "rubric cannot be frozen before calibration is complete: "
            f"{missing_human} human and {missing_judge} judge scores are missing"
        )
    holdout = [example for example in study.examples if example.split == "holdout"]
    if any(
        example.human_score is not None or example.judge_score is not None for example in holdout
    ):
        raise StudyValidationError("rubric cannot be frozen after holdout scores exist")

    previous = RubricProvenance(
        rubric_id=study.rubric_id,
        rubric_version=study.rubric_version,
        rubric_status=study.rubric_status,
        rubric_content_sha256=study.rubric_content_sha256,
    )
    payload = study.model_dump(mode="python")
    payload.update(
        {
            "rubric_id": selected.id,
            "rubric_version": selected.version,
            "rubric_status": selected.status,
            "rubric_content_sha256": rubric_content_hash(selected),
            "calibration_rubric": previous.model_dump(mode="python"),
        }
    )
    return JudgeStudy.model_validate(payload)


def validate_study_split(
    study: JudgeStudy,
    split: StudySplit,
    rubric: JudgeRubric | None = None,
) -> JudgeRubric:
    """Validate rubric provenance and keep holdout sealed during calibration."""
    selected = validate_study_rubric(study, rubric)
    if split == "holdout" and selected.status != "frozen":
        raise StudyValidationError(
            "freeze the calibrated rubric before using the untouched holdout split"
        )
    return selected


def record_human_score(
    study: JudgeStudy,
    sample_id: str,
    score: Literal[0, 1, 2],
    *,
    labeled_at: datetime | None = None,
) -> JudgeStudy:
    """Return a validated copy with one human score recorded."""
    found = False
    examples = []
    for example in study.examples:
        if example.sample_id == sample_id:
            found = True
            examples.append(
                example.model_copy(
                    update={"human_score": score, "labeled_at": labeled_at or datetime.now(UTC)}
                )
            )
        else:
            examples.append(example)
    if not found:
        raise StudyValidationError(f"Study sample {sample_id!r} does not exist")
    return study.model_copy(update={"examples": examples})


def record_judge_response(
    study: JudgeStudy,
    sample_id: str,
    response: JudgeResponse,
    *,
    judge_model: str,
    provider_schema_version: str,
    judged_at: datetime | None = None,
) -> JudgeStudy:
    """Attach one judge response while enforcing consistent inference provenance."""
    if study.judge_model not in (None, judge_model):
        raise StudyValidationError(
            f"study was scored with judge model {study.judge_model!r}, not {judge_model!r}"
        )
    if study.judge_provider_schema_version not in (None, provider_schema_version):
        raise StudyValidationError("study was scored with a different judge provider schema")
    found = False
    examples = []
    for example in study.examples:
        if example.sample_id == sample_id:
            found = True
            examples.append(
                example.model_copy(
                    update={
                        "judge_score": response.score,
                        "judge_rationale": response.rationale,
                        "judged_at": judged_at or datetime.now(UTC),
                    }
                )
            )
        else:
            examples.append(example)
    if not found:
        raise StudyValidationError(f"Study sample {sample_id!r} does not exist")
    return study.model_copy(
        update={
            "examples": examples,
            "judge_model": judge_model,
            "judge_provider_schema_version": provider_schema_version,
        }
    )


async def score_study_split(
    study: JudgeStudy,
    split: StudySplit,
    *,
    database: Database,
    evaluator: JudgeEvaluator,
    judge_model: str,
    on_progress: Callable[[JudgeStudy, int, int], None] | None = None,
) -> JudgeStudy:
    """Score one preassigned split, reusing the content-addressed judge cache."""
    rubric = validate_study_split(study, split, evaluator.rubric)
    if judge_model in {run.model for run in study.source_runs}:
        raise StudyValidationError("judge model must be independent from both target models")
    if study.judge_model not in (None, judge_model):
        raise StudyValidationError(
            f"study was scored with judge model {study.judge_model!r}, not {judge_model!r}"
        )
    schema_version = evaluator.provider.schema_version
    if study.judge_provider_schema_version not in (None, schema_version):
        raise StudyValidationError("study was scored with a different judge provider schema")

    pending = [
        example
        for example in study.examples
        if example.split == split and example.judge_score is None
    ]
    total = sum(example.split == split for example in study.examples)
    completed = total - len(pending)
    if not pending:
        return study

    database.initialize()
    await evaluator.preflight(judge_model)
    updated = study
    for example in pending:
        case = EvaluationCase(
            id=example.case_id,
            category=example.category,
            prompt=example.prompt,
            expected=example.expected,
            graders=[{"type": "judge", "rubric": rubric.id}],
        )
        request_hash = judge_request_hash(
            rubric=rubric,
            judge_model=judge_model,
            case=case,
            answer=example.answer,
            provider_schema_version=schema_version,
        )
        response = database.get_cached_judge_response(request_hash)
        if response is None:
            response = await evaluator.score(
                model=judge_model,
                case=case,
                answer=example.answer,
                rubric_id=rubric.id,
            )
            database.put_cached_judge_response(request_hash, response)
        updated = record_judge_response(
            updated,
            example.sample_id,
            response,
            judge_model=judge_model,
            provider_schema_version=schema_version,
        )
        completed += 1
        if on_progress is not None:
            on_progress(updated, completed, total)
    return updated


def build_agreement_report(
    study: JudgeStudy,
    split: StudySplit,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
    rubric: JudgeRubric | None = None,
) -> StudyAgreementReport:
    """Summarize complete human/judge pairs for exactly one preassigned split."""
    validate_study_split(study, split, rubric)
    report_rubric = (
        study.calibration_rubric
        if split == "calibration" and study.calibration_rubric is not None
        else RubricProvenance(
            rubric_id=study.rubric_id,
            rubric_version=study.rubric_version,
            rubric_status=study.rubric_status,
            rubric_content_sha256=study.rubric_content_sha256,
        )
    )
    examples = [example for example in study.examples if example.split == split]
    missing_human = sum(example.human_score is None for example in examples)
    missing_judge = sum(example.judge_score is None for example in examples)
    if missing_human or missing_judge:
        raise StudyValidationError(
            f"{split} report requires complete scores: "
            f"{missing_human} human and {missing_judge} judge scores are missing"
        )
    if study.judge_model is None:
        raise StudyValidationError("study does not record judge model provenance")

    human = [example.human_score for example in examples]
    judged = [example.judge_score for example in examples]
    if any(score is None for score in human) or any(score is None for score in judged):
        raise AssertionError("complete-score validation guarantees ordinal values")
    human_scores = [int(score) for score in human if score is not None]
    judge_scores = [int(score) for score in judged if score is not None]
    ci_low, ci_high = bootstrap_kappa_ci(
        human_scores,
        judge_scores,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    categories: dict[str, CategoryAgreement] = {}
    for category in sorted({example.category for example in examples}):
        category_examples = [example for example in examples if example.category == category]
        category_human = [
            int(example.human_score)
            for example in category_examples
            if example.human_score is not None
        ]
        category_judged = [
            int(example.judge_score)
            for example in category_examples
            if example.judge_score is not None
        ]
        categories[category] = CategoryAgreement(
            example_count=len(category_examples),
            raw_agreement=sum(
                expected == actual
                for expected, actual in zip(category_human, category_judged, strict=True)
            )
            / len(category_examples),
            quadratic_weighted_kappa=quadratic_weighted_kappa(category_human, category_judged),
            confusion_matrix=ordinal_confusion_matrix(category_human, category_judged),
        )

    return StudyAgreementReport(
        study_schema_version=study.schema_version,
        split=split,
        rubric_id=report_rubric.rubric_id,
        rubric_version=report_rubric.rubric_version,
        rubric_status=report_rubric.rubric_status,
        rubric_content_sha256=report_rubric.rubric_content_sha256,
        judge_model=study.judge_model,
        source_runs=study.source_runs,
        example_count=len(examples),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        raw_agreement=sum(
            expected == actual for expected, actual in zip(human_scores, judge_scores, strict=True)
        )
        / len(examples),
        quadratic_weighted_kappa=quadratic_weighted_kappa(human_scores, judge_scores),
        kappa_ci_low=ci_low,
        kappa_ci_high=ci_high,
        confusion_labels=[0, 1, 2],
        confusion_matrix=ordinal_confusion_matrix(human_scores, judge_scores),
        category_breakdown=categories,
        limitations=[
            "Human reference scores come from one labeler, so inter-rater reliability "
            "cannot be estimated."
        ],
    )
