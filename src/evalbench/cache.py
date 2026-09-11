from __future__ import annotations

import hashlib
import json

from evalbench.models import EvaluationCase


def generation_request_hash(
    *, model: str, case: EvaluationCase, provider_schema_version: str
) -> str:
    """Hash every input that can change a provider's generated response."""
    request = {
        "max_tokens": case.generation.max_tokens,
        "model": model,
        "prompt": case.prompt,
        "provider_schema_version": provider_schema_version,
        "system": case.system,
        "temperature": case.generation.temperature,
    }
    canonical = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
