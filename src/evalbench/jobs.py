from __future__ import annotations

import asyncio
from contextlib import suppress

from evalbench.models import EvaluationSuite, RunSummary
from evalbench.providers import Provider
from evalbench.service import RunService
from evalbench.store import Database


class ActiveRunError(RuntimeError):
    """Raised when local inference is already occupied by another run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run {run_id!r} is already active.")


class RunCoordinator:
    """Own at most one background evaluation and its cooperative cancellation signal."""

    def __init__(
        self,
        database: Database,
        provider: Provider,
        *,
        provider_name: str,
    ) -> None:
        self.database = database
        self.provider = provider
        self.provider_name = provider_name
        self.service = RunService(database, provider)
        self._lock = asyncio.Lock()
        self._active_run_id: str | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._cancel_event: asyncio.Event | None = None

    async def models(self) -> list[str]:
        return await self.provider.models()

    async def start(self, suite: EvaluationSuite, model: str) -> RunSummary:
        async with self._lock:
            if self._active_task is not None and not self._active_task.done():
                if self._active_run_id is None:
                    raise AssertionError("active task must have a run ID")
                raise ActiveRunError(self._active_run_id)

            run = self.service.create_run(
                suite,
                model,
                provider_name=self.provider_name,
            )
            cancel_event = asyncio.Event()
            task = asyncio.create_task(
                self._execute(run, suite, cancel_event),
                name=f"evalbench-run-{run.id}",
            )
            self._active_run_id = run.id
            self._active_task = task
            self._cancel_event = cancel_event
            task.add_done_callback(self._task_finished)
            return run

    async def cancel(self, run_id: str) -> None:
        async with self._lock:
            run = self.database.get_run(run_id)
            if run is None:
                raise KeyError(f"Run {run_id!r} does not exist.")
            if (
                self._active_run_id != run_id
                or self._active_task is None
                or self._active_task.done()
                or self._cancel_event is None
            ):
                raise ValueError(f"Run {run_id!r} is not active and cannot be cancelled.")
            self._cancel_event.set()

    async def wait_until_idle(self) -> None:
        """Wait for the current run, primarily for orderly tests and shutdown."""
        task = self._active_task
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        """Stop in-process work and persist restart-safe interrupted states."""
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.database.interrupt_unfinished_runs()

    async def _execute(
        self,
        run: RunSummary,
        suite: EvaluationSuite,
        cancel_event: asyncio.Event,
    ) -> None:
        # RunService records actionable failures in SQLite for polling clients.
        with suppress(Exception):
            await self.service.execute(run, suite, should_cancel=cancel_event.is_set)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
        if self._active_task is task:
            self._active_run_id = None
            self._active_task = None
            self._cancel_event = None
