import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  cancelRun,
  getRunStatus,
  listModels,
  listSuites,
  RunStatus,
  startRun,
  SuiteSummary,
} from "./api";
import ResultsWorkspace from "./ResultsWorkspace";

const TERMINAL_STATES = new Set(["completed", "failed", "cancelled", "interrupted"]);

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function displayStatus(status: RunStatus["status"]): string {
  return status.replace("_", " ").toUpperCase();
}

function fetchOptions(): Promise<[string[], SuiteSummary[]]> {
  return Promise.all([listModels(), listSuites()]);
}

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [suites, setSuites] = useState<SuiteSummary[]>([]);
  const [model, setModel] = useState("");
  const [suiteId, setSuiteId] = useState("");
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [run, setRun] = useState<RunStatus | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resultsRefreshKey, setResultsRefreshKey] = useState(0);

  function applyOptions(availableModels: string[], availableSuites: SuiteSummary[]) {
    setModels(availableModels);
    setSuites(availableSuites);
    setModel((current) =>
      availableModels.includes(current) ? current : (availableModels[0] ?? ""),
    );
    setSuiteId((current) =>
      availableSuites.some((suite) => suite.id === current)
        ? current
        : (availableSuites[0]?.id ?? ""),
    );
  }

  async function reloadOptions() {
    setLoadingOptions(true);
    setOptionsError(null);
    try {
      const [availableModels, availableSuites] = await fetchOptions();
      applyOptions(availableModels, availableSuites);
    } catch (error) {
      setOptionsError(errorMessage(error));
    } finally {
      setLoadingOptions(false);
    }
  }

  useEffect(() => {
    let stopped = false;
    void fetchOptions()
      .then(([availableModels, availableSuites]) => {
        if (stopped) return;
        setModels(availableModels);
        setSuites(availableSuites);
        setModel(availableModels[0] ?? "");
        setSuiteId(availableSuites[0]?.id ?? "");
      })
      .catch((error: unknown) => {
        if (!stopped) setOptionsError(errorMessage(error));
      })
      .finally(() => {
        if (!stopped) setLoadingOptions(false);
      });
    return () => {
      stopped = true;
    };
  }, []);

  const isActive = run !== null && !TERMINAL_STATES.has(run.status);

  useEffect(() => {
    if (!run || TERMINAL_STATES.has(run.status)) return;

    let stopped = false;
    const interval = window.setInterval(() => {
      void getRunStatus(run.id)
        .then((status) => {
          if (stopped) return;
          setRun(status);
          if (TERMINAL_STATES.has(status.status)) {
            window.clearInterval(interval);
            setCancelling(false);
            if (status.status === "completed") {
              setResultsRefreshKey((key) => key + 1);
            }
          }
        })
        .catch((error: unknown) => {
          if (stopped) return;
          window.clearInterval(interval);
          setRunError(`Status polling stopped: ${errorMessage(error)}`);
        });
    }, 1000);

    return () => {
      stopped = true;
      window.clearInterval(interval);
    };
  }, [run]);

  const selectedSuite = useMemo(
    () => suites.find((suite) => suite.id === suiteId) ?? null,
    [suiteId, suites],
  );
  const progress = run && run.total_cases > 0 ? (run.completed_cases / run.total_cases) * 100 : 0;

  async function handleStart(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!suiteId || !model || isActive) return;
    setStarting(true);
    setRunError(null);
    setCancelling(false);
    try {
      const accepted = await startRun(suiteId, model);
      setRun({
        id: accepted.id,
        status: accepted.status,
        completed_cases: 0,
        total_cases: selectedSuite?.case_count ?? 0,
        error: null,
      });
    } catch (error) {
      setRunError(errorMessage(error));
    } finally {
      setStarting(false);
    }
  }

  async function handleCancel() {
    if (!run || !isActive || cancelling) return;
    setCancelling(true);
    setRunError(null);
    try {
      await cancelRun(run.id);
    } catch (error) {
      setCancelling(false);
      setRunError(errorMessage(error));
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="EvalBench home">
          <span className="brand-mark">EB</span>
          <span className="brand-copy">
            <strong>EvalBench</strong>
            <small>LOCAL MODEL LAB</small>
          </span>
        </a>
        <div className="local-status"><i /> LOCAL WORKBENCH</div>
      </header>

      <div className="workspace">
        <section className="page-heading">
          <div>
            <p className="eyebrow">Evaluation control</p>
            <h1>Run a reproducible model evaluation.</h1>
          </div>
          <p>
            Select an installed Ollama model and a versioned suite. EvalBench records every
            generation and grader decision locally.
          </p>
        </section>

        {optionsError && (
          <div className="alert alert-error" role="alert">
            <div>
              <strong>Couldn’t load the local workbench.</strong>
              <span>{optionsError}</span>
            </div>
            <button className="button button-secondary" type="button" onClick={reloadOptions}>
              Retry connection
            </button>
          </div>
        )}

        <div className="control-grid">
          <section className="panel" aria-labelledby="new-run-title">
            <div className="panel-heading">
              <span className="step-number">01</span>
              <div>
                <p className="eyebrow">Configure</p>
                <h2 id="new-run-title">New evaluation</h2>
              </div>
            </div>

            <form onSubmit={handleStart}>
              <label htmlFor="suite">Evaluation suite</label>
              <select
                id="suite"
                value={suiteId}
                onChange={(event) => setSuiteId(event.target.value)}
                disabled={loadingOptions || isActive}
              >
                {suites.length === 0 && <option value="">No suites available</option>}
                {suites.map((suite) => (
                  <option key={suite.id} value={suite.id}>
                    {suite.name} · v{suite.version} · {suite.case_count} cases
                  </option>
                ))}
              </select>

              <label htmlFor="model">Installed model</label>
              <select
                id="model"
                value={model}
                onChange={(event) => setModel(event.target.value)}
                disabled={loadingOptions || isActive}
              >
                {models.length === 0 && <option value="">No Ollama models found</option>}
                {models.map((availableModel) => (
                  <option key={availableModel} value={availableModel}>
                    {availableModel}
                  </option>
                ))}
              </select>

              {selectedSuite && (
                <div className="suite-meta">
                  <span>{selectedSuite.categories.length} categories</span>
                  <span>{selectedSuite.case_count} prompts</span>
                  <span>temperature 0</span>
                </div>
              )}

              <button
                className="button button-primary"
                type="submit"
                disabled={loadingOptions || starting || isActive || !suiteId || !model}
              >
                {starting ? "Starting…" : isActive ? "Evaluation in progress" : "Run evaluation"}
              </button>
            </form>
          </section>

          <section className="panel monitor" aria-labelledby="monitor-title" aria-live="polite">
            <div className="panel-heading">
              <span className="step-number">02</span>
              <div>
                <p className="eyebrow">Observe</p>
                <h2 id="monitor-title">Run monitor</h2>
              </div>
              {run && (
                <span className={`status-badge status-${run.status}`}>
                  {displayStatus(run.status)}
                </span>
              )}
            </div>

            {!run ? (
              <div className="empty-state">
                <span className="pulse-ring" />
                <h3>Ready for a local run</h3>
                <p>Progress and cancellation controls will appear here after launch.</p>
                {runError && (
                  <p className="inline-error" role="alert">{runError}</p>
                )}
              </div>
            ) : (
              <div className="run-state">
                <div className="run-identity">
                  <span>RUN ID</span>
                  <code>{run.id}</code>
                </div>
                <div className="progress-copy">
                  <strong>
                    {run.completed_cases} / {run.total_cases}
                  </strong>
                  <span>{Math.round(progress)}% complete</span>
                </div>
                <progress value={run.completed_cases} max={run.total_cases || 1}>
                  {Math.round(progress)}%
                </progress>

                {cancelling && (
                  <p className="inline-note">
                    Cancellation requested. The current case will finish first.
                  </p>
                )}
                {run.error && (
                  <p className="inline-error" role="alert">{run.error}</p>
                )}
                {runError && (
                  <p className="inline-error" role="alert">{runError}</p>
                )}

                {isActive && (
                  <button
                    className="button button-danger"
                    type="button"
                    onClick={handleCancel}
                    disabled={cancelling}
                  >
                    {cancelling ? "Cancellation pending…" : "Cancel after current case"}
                  </button>
                )}
              </div>
            )}
          </section>
        </div>

        <ResultsWorkspace
          preferredRunId={run?.status === "completed" ? run.id : null}
          refreshKey={resultsRefreshKey}
        />

        <footer>
          <span>SQLite persistence</span>
          <span>Deterministic grading</span>
          <span>No API key</span>
        </footer>
      </div>
    </main>
  );
}
