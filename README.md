# EvalBench

EvalBench is an explainable evaluation workbench for local language models.

This repository is being built in verified milestones. Milestone 1 establishes the
typed evaluation domain, YAML suite validation, a hand-authored 60-case general
assistant suite, Python packaging, and the React/TypeScript application shell.
Evaluation execution, persistence, Ollama integration, statistics, API workflows,
and judge validation intentionally arrive in later milestones.

## Milestone 1 setup

Requirements: Python 3.12+ and Node 20+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
mypy src

cd frontend
npm ci
npm test
npm run lint
npm run build
```

The suite lives at `suites/core.yaml`. It covers reasoning, structured extraction,
instruction following, and safety/refusal with 15 cases in each category.

## Domain model

- `EvaluationSuite` owns a name, semantic version, and unique cases.
- `EvaluationCase` owns identity, category, prompt, optional system/reference data,
  generation settings, and one or more discriminated grader specifications.
- Grader specs currently describe exact, numeric, contains-all, JSON Schema, length,
  refusal, and future judge evaluation.
- `Generation`, `Judgment`, and `RunSummary` define the records later milestones will
  create and persist.

See [PLAN.md](PLAN.md) for the full roadmap. MIT licensed.
