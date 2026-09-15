import json

import httpx
import pytest

from evalbench.models import EvaluationCase, Generation
from evalbench.providers import OllamaProvider, Provider, ProviderError, ReplayProvider


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


def ollama(transport: httpx.AsyncBaseTransport, *, timeout: float = 120) -> OllamaProvider:
    return OllamaProvider(transport=transport, timeout=timeout)


@pytest.mark.asyncio
async def test_ollama_lists_installed_models() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:0.6b"}, {"name": "gemma3:4b"}]},
        )

    async with ollama(httpx.MockTransport(handler)) as provider:
        assert await provider.models() == ["qwen3:0.6b", "gemma3:4b"]


@pytest.mark.asyncio
async def test_ollama_missing_model_has_pull_instruction() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "other:1b"}]})

    async with ollama(httpx.MockTransport(handler)) as provider:
        with pytest.raises(ProviderError, match=r"ollama pull qwen3:0\.6b"):
            await provider.preflight("qwen3:0.6b")


@pytest.mark.asyncio
async def test_ollama_maps_chat_payload_and_metrics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        request_body = json.loads(request.content)
        assert request_body == {
            "model": "qwen3:0.6b",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "prompt"},
            ],
            "stream": False,
            "options": {"temperature": 0.25, "num_predict": 64},
        }
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "answer"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 9,
                "eval_count": 3,
                "total_duration": 2_000_000,
                "eval_duration": 1_000_000,
            },
        )

    case = make_case().model_copy(
        update={
            "system": "Be terse.",
            "generation": make_case().generation.model_copy(
                update={"temperature": 0.25, "max_tokens": 64}
            ),
        }
    )
    async with ollama(httpx.MockTransport(handler)) as provider:
        result = await provider.generate("qwen3:0.6b", case)

    assert result == Generation(
        text="answer",
        done_reason="stop",
        prompt_tokens=9,
        output_tokens=3,
        total_duration_ns=2_000_000,
        eval_duration_ns=1_000_000,
    )


@pytest.mark.asyncio
async def test_ollama_requests_schema_constrained_json_at_temperature_zero() -> None:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }
    messages = [{"role": "user", "content": "Evaluate this."}]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        assert json.loads(request.content) == {
            "model": "gemma3:4b",
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"score": 2}'}, "done": True},
        )

    async with ollama(httpx.MockTransport(handler)) as provider:
        assert await provider.structured_chat("gemma3:4b", messages=messages, schema=schema) == {
            "score": 2
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
async def test_ollama_rejects_malformed_structured_output(content: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}, "done": True},
        )

    async with ollama(httpx.MockTransport(handler)) as provider:
        with pytest.raises(ProviderError, match="malformed structured output"):
            await provider.structured_chat("gemma3:4b", messages=[], schema={})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("connect", "ollama serve"),
        ("timeout", "timed out after 3 seconds"),
        ("http", "HTTP 500"),
        ("invalid_json", "request failed"),
    ],
)
async def test_ollama_transport_errors_are_actionable(failure: str, message: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("offline", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if failure == "http":
            return httpx.Response(500, text="engine failed")
        return httpx.Response(200, text="not-json")

    async with ollama(httpx.MockTransport(handler), timeout=3) as provider:
        with pytest.raises(ProviderError, match=message):
            await provider.models()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "response", "message"),
    [
        ("models", {}, "malformed model list"),
        ("chat", {"done": True}, "malformed chat response"),
    ],
)
async def test_ollama_rejects_malformed_success_response(
    endpoint: str, response: dict[str, object], message: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with ollama(httpx.MockTransport(handler)) as provider:
        with pytest.raises(ProviderError, match=message):
            if endpoint == "models":
                await provider.models()
            else:
                await provider.generate("qwen3:0.6b", make_case())
