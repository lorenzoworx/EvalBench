import pytest

from evalbench.models import EvaluationCase, Generation
from evalbench.providers import Provider, ProviderError, ReplayProvider


def make_case(case_id: str = "case") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        category="test",
        prompt="prompt",
        expected="expected",
        graders=[{"type": "exact"}],
    )


def test_replay_provider_satisfies_protocol() -> None:
    assert isinstance(ReplayProvider({}), Provider)


@pytest.mark.asyncio
async def test_replay_normalizes_text_response() -> None:
    provider = ReplayProvider({"case": "recorded answer"})

    result = await provider.generate("replay", make_case())

    assert result == Generation(text="recorded answer", done_reason="stop")


@pytest.mark.asyncio
async def test_replay_preserves_normalized_generation_metadata() -> None:
    recorded = Generation(
        text="answer",
        done_reason="length",
        prompt_tokens=8,
        output_tokens=4,
        total_duration_ns=2_000_000,
    )
    provider = ReplayProvider({"case": recorded})

    result = await provider.generate("replay", make_case())

    assert result == recorded
    assert result is not recorded


@pytest.mark.asyncio
async def test_replay_requires_recorded_case() -> None:
    provider = ReplayProvider({})

    with pytest.raises(ProviderError, match="No replay response.*missing"):
        await provider.generate("replay", make_case("missing"))


@pytest.mark.asyncio
async def test_replay_rejects_unknown_model() -> None:
    provider = ReplayProvider({"case": "answer"}, model_names=("fixture-a", "fixture-b"))

    with pytest.raises(ProviderError, match="unknown.*unavailable"):
        await provider.generate("unknown", make_case())


@pytest.mark.asyncio
async def test_models_returns_defensive_copy() -> None:
    provider = ReplayProvider({}, model_names=("fixture-a", "fixture-b"))

    models = await provider.models()
    models.append("mutated")

    assert await provider.models() == ["fixture-a", "fixture-b"]
