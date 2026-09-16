from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalbench.cache import rubric_content_hash
from evalbench.judge import JudgeRubric, load_rubric
from evalbench.service import RunService
from evalbench.store import Database, StoredCaseResult

StudySplit = Literal["calibration", "holdout"]


class StudyValidationError(ValueError):
    """Raised when a judge-study artifact or source runs are invalid."""


class StudyRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model: str


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


class JudgeStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["judge-study-v1"] = "judge-study-v1"
    rubric_id: str
    rubric_version: str
    rubric_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    sample_size: int = Field(gt=0)
    source_runs: list[StudyRun]
    examples: list[StudyExample]

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
        return self


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
    return JudgeStudy(
        rubric_id=selected_rubric.id,
        rubric_version=selected_rubric.version,
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


def validate_study_rubric(study: JudgeStudy, rubric: JudgeRubric | None = None) -> JudgeRubric:
    selected = rubric or load_rubric()
    if (
        study.rubric_id,
        study.rubric_version,
        study.rubric_content_sha256,
    ) != (selected.id, selected.version, rubric_content_hash(selected)):
        raise StudyValidationError("study rubric provenance does not match the current rubric")
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
