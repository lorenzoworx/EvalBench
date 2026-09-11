from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

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
        if model not in self._model_names:
            raise ProviderError(
                f"Replay model {model!r} is unavailable; choose one of {self._model_names!r}."
            )
        try:
            response = self._responses[case.id]
        except KeyError as exc:
            raise ProviderError(f"No replay response is recorded for case {case.id!r}.") from exc
        if isinstance(response, Generation):
            return response.model_copy(deep=True)
        return Generation(text=response, done_reason="stop")

    async def models(self) -> list[str]:
        return self._model_names.copy()
