from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from evalbench.models import EvaluationCase, Generation


class ProviderError(RuntimeError):
    """An actionable error raised by an inference provider."""


@runtime_checkable
class Provider(Protocol):
    """The inference boundary consumed by the future run service."""

    schema_version: str

    async def generate(self, model: str, case: EvaluationCase) -> Generation:
        """Generate one normalized response for an evaluation case."""
        ...

    async def preflight(self, model: str) -> None:
        """Raise an actionable error when a requested model cannot run."""
        ...

    async def models(self) -> list[str]:
        """Return model identifiers available through this provider."""
        ...


class ReplayProvider:
    """Return recorded responses without network or model dependencies."""

    schema_version = "replay-v1"

    def __init__(
        self,
        responses: Mapping[str, str | Generation],
        *,
        model_names: tuple[str, ...] = ("replay",),
    ) -> None:
        self._responses = dict(responses)
        self._model_names = list(model_names)

    async def generate(self, model: str, case: EvaluationCase) -> Generation:
        await self.preflight(model)
        try:
            response = self._responses[case.id]
        except KeyError as exc:
            raise ProviderError(f"No replay response is recorded for case {case.id!r}.") from exc
        if isinstance(response, Generation):
            return response.model_copy(deep=True)
        return Generation(text=response, done_reason="stop")

    async def models(self) -> list[str]:
        return self._model_names.copy()

    async def preflight(self, model: str) -> None:
        if model not in self._model_names:
            raise ProviderError(
                f"Replay model {model!r} is unavailable; choose one of {self._model_names!r}."
            )


class _OllamaModel(BaseModel):
    name: str


class _OllamaTagsResponse(BaseModel):
    models: list[_OllamaModel]


class _OllamaMessage(BaseModel):
    content: str


class _OllamaChatResponse(BaseModel):
    message: _OllamaMessage
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration: int | None = None
    eval_duration: int | None = None


class OllamaProvider:
    """Direct adapter for Ollama's local, non-streaming chat API."""

    schema_version = "ollama-chat-v1"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: float = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )
        self._timeout = timeout

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def models(self) -> list[str]:
        payload = await self._request("GET", "/api/tags")
        try:
            response = _OllamaTagsResponse.model_validate(payload)
        except ValidationError as exc:
            raise ProviderError(f"Ollama returned a malformed model list: {exc}") from exc
        return [model.name for model in response.models]

    async def preflight(self, model: str) -> None:
        installed = await self.models()
        if model not in installed:
            raise ProviderError(
                f"Ollama model {model!r} is not installed. Run `ollama pull {model}` first. "
                f"Installed models: {', '.join(installed) or 'none'}."
            )

    async def generate(self, model: str, case: EvaluationCase) -> Generation:
        messages: list[dict[str, str]] = []
        if case.system:
            messages.append({"role": "system", "content": case.system})
        messages.append({"role": "user", "content": case.prompt})
        payload = await self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": case.generation.temperature,
                    "num_predict": case.generation.max_tokens,
                },
            },
        )
        try:
            response = _OllamaChatResponse.model_validate(payload)
        except ValidationError as exc:
            raise ProviderError(f"Ollama returned a malformed chat response: {exc}") from exc
        return Generation(
            text=response.message.content,
            done_reason=response.done_reason,
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            total_duration_ns=response.total_duration,
            eval_duration_ns=response.eval_duration,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is not reachable at 127.0.0.1:11434. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Ollama timed out after {self._timeout:g} seconds.") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            suffix = f" Response: {detail}" if detail else ""
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}.{suffix}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
