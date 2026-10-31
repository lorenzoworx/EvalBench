"""Export explicitly fictional response fixtures through the real evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evalbench.providers import ReplayProvider
from evalbench.service import RunService
from evalbench.store import Database
from evalbench.suite import load_suite


async def build_replay(root: Path) -> dict[str, Any]:
    suite_path = root / "suites/smoke.yaml"
    suite = load_suite(suite_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "provenance": (
            "Fictional response fixtures graded by the real offline pipeline. "
            "No model-performance claims."
        ),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "runs": [],
        "metrics": {},
        "results": {},
        "comparisons": [],
    }
    with TemporaryDirectory() as temporary:
        database = Database(Path(temporary) / "replay.db")
        for name, filename in (
            ("Baseline fixture", "smoke-responses-baseline.json"),
            ("Candidate fixture", "smoke-responses.json"),
        ):
            responses = json.loads((root / "examples" / filename).read_text())
            provider = ReplayProvider(responses, model_names=(name,))
            service = RunService(database, provider)
            run_id = "baseline-fixture" if name.startswith("Baseline") else "candidate-fixture"
            run = await service.run(suite, name, provider_name="replay", run_id=run_id)
            exported = run.model_dump(mode="json")
            # Explicit synthetic fixture timestamp, not a claimed model run date.
            exported["created_at"] = exported["completed_at"] = "2026-10-30T00:00:00Z"
            payload["runs"].append(exported)
            payload["metrics"][run_id] = service.metrics(run_id).model_dump(mode="json")
            payload["results"][run_id] = [
                record.result.model_dump(mode="json")
                for record in database.list_case_results(run_id)
            ]
        service = RunService(database)
        for baseline, candidate in (
            ("baseline-fixture", "candidate-fixture"),
            ("candidate-fixture", "baseline-fixture"),
        ):
            payload["comparisons"].append(
                service.compare(baseline, candidate).model_dump(mode="json")
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "frontend/public/replay-data.json"
    rendered = json.dumps(asyncio.run(build_replay(root)), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not output.is_file() or output.read_text() != rendered:
            raise SystemExit("Replay export is stale; run python -m evalbench.replay_export")
        print("Replay export matches the suite, fixtures, graders, and comparison pipeline.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print("Exported fictional replay fixtures; no model inference was used.")


if __name__ == "__main__":
    main()
