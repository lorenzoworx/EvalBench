# EvalBench

EvalBench is an explainable evaluation workbench for local language models. It is
being built in small, verified milestones; the current implementation supports both
deterministic offline replay and direct local Ollama inference.

## Current capabilities

- A balanced 60-case general-assistant suite covering reasoning, structured
  extraction, instruction following, and safety/refusal.
- Typed Pydantic models and strict YAML validation.
- Exact, numeric, contains-all, JSON Schema, length, and refusal graders with
  evidence-based rationales.
- A provider protocol and deterministic replay provider for network-free tests.
- Direct non-streaming Ollama inference with local model preflight checks.
- SQLite storage for runs, immutable case snapshots, generations, and judgments.
- Content-addressed generation caching keyed by the complete inference request.
- One shared run service used by the CLI and future web API.

Ollama integration, statistical comparison, and the functional dashboard are later
milestones and are intentionally absent today.

## Setup

Requirements: Python 3.12+ and Node 20+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

cd frontend
npm ci
cd ..
```

## Offline smoke test

The committed smoke fixture exercises every deterministic grader without Ollama or
network access:

```bash
evalbench suite validate suites/smoke.yaml
evalbench run \
  --provider replay \
  --suite suites/smoke.yaml \
  --replay examples/smoke-responses.json \
  --database results/evalbench.db
evalbench runs --database results/evalbench.db
```

The run should complete six cases with `accuracy` equal to `1.0`. Running it again
uses the persistent generation cache; stored case results identify cache hits.

## Ollama runs

With Ollama already running and a model already downloaded:

```bash
ollama serve
ollama pull qwen3:0.6b
evalbench run \
  --provider ollama \
  --model qwen3:0.6b \
  --suite suites/core.yaml \
  --database results/evalbench.db
```

EvalBench connects only to `127.0.0.1:11434`, performs non-streaming chat calls,
and records Ollama's finish reason, token counts, total duration, and evaluation
duration. No hosted provider or API key is used.

Validate the full suite independently:

```bash
evalbench suite validate suites/core.yaml
```

## Architecture

```text
suite YAML
    │
    ▼
Pydantic validation ──► RunService ──► Provider protocol ──┬─► ReplayProvider
                            │                              └─► OllamaProvider
                            │                 │
                            │                 └── generation request hash
                            │                              │
                            ▼                              ▼
                      rule graders ◄────────────── SQLite cache
                            │
                            ▼
                SQLite run + case evidence
```

`RunService` owns orchestration so CLI and future API behavior cannot diverge. The
provider protocol keeps inference replaceable: CI uses replay data now, while the
next milestone adds Ollama without changing grading or persistence.

## Development checks

```bash
pytest --cov=evalbench --cov-report=term-missing --cov-fail-under=85
ruff check .
ruff format --check .
mypy src

cd frontend
npm run lint
npm test
npm run build
```

CI never requires Ollama or network access after dependencies are installed.
Runtime databases under `results/` are ignored by Git.

## Domain model

- `EvaluationSuite` owns a name, version, and cases with unique IDs.
- `EvaluationCase` owns its prompt, optional system/reference data, generation
  settings, and discriminated grader specifications.
- `Generation`, `Judgment`, `CaseResult`, and `RunSummary` form the evidence trail.
- A case passes only when at least one primary judgment exists and all primary
  judgments pass.

See [PLAN.md](PLAN.md) for the complete roadmap. MIT licensed.
