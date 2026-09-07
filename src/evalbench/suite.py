from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from evalbench.models import EvaluationSuite


class SuiteValidationError(ValueError):
    pass


def load_suite(path: str | Path) -> EvaluationSuite:
    suite_path = Path(path)
    try:
        raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteValidationError(f"Cannot read suite {suite_path}: {exc}") from exc
    try:
        return EvaluationSuite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteValidationError(f"Invalid suite {suite_path}:\n{exc}") from exc


def discover_suites(directory: str | Path) -> list[tuple[Path, EvaluationSuite]]:
    found: list[tuple[Path, EvaluationSuite]] = []
    for path in sorted(Path(directory).glob("*.yaml")):
        found.append((path, load_suite(path)))
    return found
