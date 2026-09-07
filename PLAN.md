# Rebuild EvalBench as an Explainable ML Portfolio Project

## Summary

Rebuild EvalBench from scratch in `/Users/lorenzoworx/Projects/EvalBench` as a public `evalbench` GitHub repository over 4–6 weeks.

The finished project will evaluate local Ollama models across a curated general-assistant suite, store reproducible results in SQLite, compare models statistically, launch runs from a React dashboard, and validate a local LLM judge against 60 human-labeled examples.

The existing `/Users/lorenzoworx/Downloads/evalbench` snapshot remains untouched as a private requirements reference. Code will be implemented fresh in small working slices, with genuine commits and no backdated or fabricated history.

## Implementation Milestones

### 1. Repository and evaluation domain

- Re-authenticate GitHub CLI with the user’s personal account, verify Git author identity, initialize `main`, create the public `evalbench` repository, and add MIT license, Python 3.12 packaging, React/TypeScript/Vite, linting, tests, and GitHub Actions.
- Define Pydantic domain models:
  - `EvaluationSuite`: name, version, cases.
  - `EvaluationCase`: id, category, prompt, optional system/expected values, generation settings, and graders.
  - Discriminated grader specifications for `exact`, `numeric`, `contains_all`, `json_schema`, `length`, `refusal`, and later `judge`.
  - `Generation`, `Judgment`, and `RunSummary`.
- Implement YAML loading and validation for duplicate IDs, missing grader parameters, invalid thresholds, empty categories, and unsupported grader types.
- Begin a hand-authored 60-case suite covering reasoning, structured extraction, instruction following, and safety/refusal.

### 2. Minimal end-to-end evaluator

- Implement the six deterministic rule graders and define a case as passed only when every configured primary grader passes.
- Add a `Provider` protocol and deterministic `ReplayProvider` so tests and CI require neither Ollama nor network access.
- Build a shared `RunService` used by both CLI and web API.
- Store runs, case snapshots, generations, judgments, and generation cache entries in SQLite.
- Hash the complete generation request—model, prompt, system message, temperature, token limit, and provider schema version—to prevent stale cache hits.
- Add CLI commands:
  - `evalbench suite validate`
  - `evalbench run`
  - `evalbench runs`
  - `evalbench compare`
  - `evalbench serve`

### 3. Ollama integration and statistical comparison

- Install Ollama with explicit approval and implement a direct `httpx` adapter to local `/api/chat`; do not add hosted providers.
- Use non-streaming calls and record output, `done_reason`, prompt/output token counts, total duration, and evaluation duration.
- Add actionable preflight errors for a stopped Ollama service or missing model.
- Use configurable models, with documented baselines:
  - Targets: `qwen3:0.6b` and `qwen3:4b-instruct`.
  - Independent judge: `gemma3:4b`.
- Compute seeded bootstrap 95% confidence intervals, category accuracy, p50/p95 latency, and Ollama-reported tokens per second.
- Compare paired runs using exact McNemar testing, accuracy delta, and explicit lists of regressions and improvements.
- Exclude batching, hosted APIs, streaming/TTFT, cost accounting, direct Transformers loading, and CUDA-specific quantization.

### 4. Local FastAPI and React application

- Build a local-only FastAPI service bound to `127.0.0.1`; Vite uses an API proxy during development, and FastAPI serves the production frontend build.
- Expose:
  - `GET /api/models` and `GET /api/suites`
  - `POST /api/runs`
  - `GET /api/runs` and run detail/results endpoints
  - `GET /api/runs/{id}/status`
  - `POST /api/runs/{id}/cancel`
  - `GET /api/compare`
- Permit one active Ollama run. Starting another returns `409`; starting a valid run returns `202` with its run ID.
- React polls run status once per second. Cancellation is cooperative between cases, completed results are retained, and server restart changes unfinished runs to `interrupted`.
- Dashboard views:
  - New-run form with suite and installed-model selection.
  - Live phase/progress and cancellation.
  - Accuracy, confidence intervals, latency, throughput, and category charts.
  - Filterable case explorer with prompts, outputs, expected values, and grader rationales.
  - Paired run comparison with regressions and McNemar result.
- Keep the application local-only; no GitHub Pages or hosted backend.

### 5. Validated local LLM judge

- Create a versioned `overall_quality_v1` rubric with ordinal scores 0, 1, and 2.
- Use Ollama structured outputs with a JSON Schema for `{score, rationale}` and temperature zero.
- Key judge-cache entries by rubric content/version, judge model, case ID, and answer hash.
- Add an interactive labeling command and select 60 balanced outputs across categories and the two target models.
- Split before tuning:
  - 30 calibration examples used to clarify the rubric once.
  - Freeze `overall_quality_v1`.
  - 30 untouched holdout examples used for the final report.
- Report raw agreement, quadratic-weighted Cohen’s kappa, bootstrap confidence interval, confusion matrix, and category breakdown.
- Publish the result even if agreement is weak; document the single-labeler limitation and do not tune against the holdout set.

### 6. Portfolio and interview readiness

- Finish the README with the problem statement, quick start, architecture diagram, dashboard screenshots, example comparison, judge-study result, limitations, and reproducibility instructions.
- Add concise architecture and methodology documents plus decision records for Ollama, SQLite, polling, caching, pass aggregation, McNemar testing, and judge validation.
- Tag a real `v0.1.0` release only after the clean-install smoke test and two real model runs succeed.
- Use one GitHub issue and short-lived branch per milestone, with 1–3 focused commits and a PR containing test evidence. Commit dates and progress remain genuine.

## Test and Acceptance Plan

- Unit-test suite validation, every grader’s boundaries, hashing, pass aggregation, bootstrap calculations, McNemar contingencies, and kappa calculations.
- Mock Ollama responses for token/timing mapping, missing models, HTTP failures, timeouts, malformed structured judge output, and finish reasons.
- Exercise replay-provider → runner → SQLite → API end to end.
- Test job lifecycle: `202` start, `409` concurrent start, polling transitions, cancellation, partial-result retention, and restart interruption.
- Test React run creation, polling, cancellation, failure states, result filtering, and comparisons with mocked API responses.
- CI runs backend lint/type checks/tests with at least 85% core coverage plus frontend lint, TypeScript build, and Vitest. CI never requires Ollama.
- Final acceptance requires:
  - A fresh clone installs and passes CI.
  - The dashboard can launch and cancel a local evaluation.
  - Two Ollama models can be compared over all 60 cases.
  - The holdout judge report is reproducible from committed labels.
  - No API key is needed after models are downloaded.

## Learning and Ownership Workflow

- The assistant implements one milestone at a time, then provides a code walkthrough, data-flow trace, debugging exercise, and interview questions before beginning the next milestone.
- The user performs the 60 human labels and runs each milestone locally.
- Knowledge checks cover:
  - Tracing a case from YAML through Ollama, grading, SQLite, API, and React.
  - Explaining dependency inversion through the provider protocol.
  - Explaining cache invalidation and reproducibility.
  - Defending bootstrap confidence intervals and exact McNemar testing.
  - Explaining calibration/holdout separation and Cohen’s kappa.
  - Describing background-run cancellation and restart behavior.
- The final handoff includes a three-minute demo script, resume bullets, an architecture explanation, likely interviewer questions, and one mock project interview.

## Assumptions

- The public repository will use the name `evalbench`; the package and CLI use lowercase `evalbench`, while the product title is “EvalBench.”
- Creating `/Users/lorenzoworx/Projects/EvalBench`, installing Ollama, authenticating GitHub, and creating the public remote will require user approval or interaction during execution.
- Ollama is accessed only through its unauthenticated local API. Its current chat response exposes the token and timing fields needed for metrics, and its structured-output interface accepts JSON Schema. [Ollama chat API](https://docs.ollama.com/api/chat), [usage metrics](https://docs.ollama.com/api/usage), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- The named Qwen and Gemma models are documented defaults rather than hardcoded requirements. [Qwen 3 models](https://ollama.com/library/qwen3), [Gemma 3 models](https://ollama.com/library/gemma3)
