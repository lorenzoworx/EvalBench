from __future__ import annotations

import uuid
from datetime import UTC, datetime

from evalbench.cache import generation_request_hash
from evalbench.graders import grade_case, passed_all_primary
from evalbench.models import CaseResult, EvaluationSuite, RunSummary
from evalbench.providers import Provider, ProviderError
from evalbench.store import Database


class RunService:
    """Orchestrate deterministic evaluation independently of its interface."""

    def __init__(self, database: Database, provider: Provider) -> None:
        self.database = database
        self.provider = provider
        self.database.initialize()

    async def run(
        self,
        suite: EvaluationSuite,
        model: str,
        *,
        provider_name: str,
        run_id: str | None = None,
    ) -> RunSummary:
        run = RunSummary(
            id=run_id or str(uuid.uuid4()),
            suite_name=suite.name,
            suite_version=suite.version,
            model=model,
            provider=provider_name,
            status="pending",
            total_cases=len(suite.cases),
        )
        self.database.create_run(run)
        run = run.model_copy(update={"status": "running"})
        self.database.update_run(run)

        outcomes: list[bool] = []
        try:
            unsupported = [
                case.id
                for case in suite.cases
                if any(spec.type == "judge" for spec in case.graders)
            ]
            if unsupported:
                raise ValueError(
                    "Judge graders are not available until Milestone 5; "
                    f"affected cases: {', '.join(unsupported)}."
                )

            available_models = await self.provider.models()
            if model not in available_models:
                raise ProviderError(
                    f"Model {model!r} is unavailable from {provider_name}; "
                    f"choose one of {available_models!r}."
                )

            for case in suite.cases:
                request_hash = generation_request_hash(
                    model=model,
                    case=case,
                    provider_schema_version=self.provider.schema_version,
                )
                generation = self.database.get_cached_generation(request_hash)
                if generation is None:
                    generation = await self.provider.generate(model, case)
                    self.database.put_cached_generation(request_hash, generation)

                judgments = grade_case(case, generation.text)
                passed = passed_all_primary(judgments)
                result = CaseResult(
                    case_id=case.id,
                    category=case.category,
                    prompt=case.prompt,
                    expected=case.expected,
                    generation=generation,
                    judgments=judgments,
                    passed=passed,
                )
                self.database.save_case_result(run.id, case, result)
                outcomes.append(passed)
                run = run.model_copy(update={"completed_cases": len(outcomes)})
                self.database.update_run(run)

            run = run.model_copy(
                update={
                    "status": "completed",
                    "completed_at": datetime.now(UTC),
                    "accuracy": sum(outcomes) / len(outcomes),
                }
            )
            self.database.update_run(run)
            return run
        except Exception as exc:
            failed = run.model_copy(
                update={
                    "status": "failed",
                    "completed_at": datetime.now(UTC),
                    "error": str(exc),
                }
            )
            self.database.update_run(failed)
            raise
