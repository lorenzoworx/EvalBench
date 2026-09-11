from pathlib import Path

import pytest

from evalbench.cache import generation_request_hash
from evalbench.models import EvaluationCase, Generation, GenerationSettings
from evalbench.store import Database


def make_case(**changes: object) -> EvaluationCase:
    values: dict[str, object] = {
        "id": "case",
        "category": "test",
        "prompt": "Explain caching.",
        "system": "Be concise.",
        "expected": "answer",
        "generation": GenerationSettings(temperature=0.2, max_tokens=128),
        "graders": [{"type": "exact"}],
    }
    values.update(changes)
    return EvaluationCase.model_validate(values)


def request_hash(
    case: EvaluationCase | None = None,
    *,
    model: str = "model-a",
    schema: str = "provider-v1",
) -> str:
    return generation_request_hash(
        model=model,
        case=case or make_case(),
        provider_schema_version=schema,
    )


@pytest.mark.parametrize(
    "changed_hash",
    [
        request_hash(model="model-b"),
        request_hash(make_case(prompt="A different prompt.")),
        request_hash(make_case(system="A different system message.")),
        request_hash(make_case(generation=GenerationSettings(temperature=0.3, max_tokens=128))),
        request_hash(make_case(generation=GenerationSettings(temperature=0.2, max_tokens=256))),
        request_hash(schema="provider-v2"),
    ],
)
def test_every_generation_input_invalidates_hash(changed_hash: str) -> None:
    assert changed_hash != request_hash()


def test_non_generation_metadata_does_not_invalidate_hash() -> None:
    assert request_hash(make_case(id="renamed", category="other")) == request_hash()


def test_hash_is_deterministic_sha256() -> None:
    assert request_hash() == request_hash(make_case())
    assert len(request_hash()) == 64


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "evalbench.db")
    value.initialize()
    return value


def test_cache_miss_returns_none(database: Database) -> None:
    assert database.get_cached_generation("missing") is None


def test_generation_cache_round_trip_marks_hit(database: Database) -> None:
    generation = Generation(
        text="cached answer",
        done_reason="stop",
        prompt_tokens=10,
        output_tokens=3,
        total_duration_ns=20_000,
        eval_duration_ns=10_000,
    )
    database.put_cached_generation("request", generation)

    cached = database.get_cached_generation("request")

    assert cached == generation.model_copy(update={"cached": True})
    assert not generation.cached


def test_cache_entries_are_immutable(database: Database) -> None:
    first = Generation(text="first")
    database.put_cached_generation("same-request", first)
    database.put_cached_generation("same-request", Generation(text="second", cached=True))

    assert database.get_cached_generation("same-request") == first.model_copy(
        update={"cached": True}
    )
