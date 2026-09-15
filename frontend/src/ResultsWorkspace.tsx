import { useEffect, useMemo, useState } from "react";
import {
  CaseResult,
  getRunMetrics,
  listRunResults,
  listRuns,
  RunMetrics,
  RunSummary,
} from "./api";

interface ResultsWorkspaceProps {
  preferredRunId?: string | null;
  refreshKey?: number;
}

interface Evidence {
  runId: string;
  metrics: RunMetrics;
  results: CaseResult[];
}

interface LoadError {
  runId: string;
  message: string;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function percent(value: number | null): string {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

function decimal(value: number | null, suffix: string): string {
  return value === null ? "—" : `${value.toFixed(1)} ${suffix}`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatExpected(value: unknown | null): string {
  if (value === null) return "No explicit reference value";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

export default function ResultsWorkspace({
  preferredRunId = null,
  refreshKey = 0,
}: ResultsWorkspaceProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyRevision, setHistoryRevision] = useState(0);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [evidenceError, setEvidenceError] = useState<LoadError | null>(null);
  const [evidenceRevision, setEvidenceRevision] = useState(0);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [outcome, setOutcome] = useState("all");

  useEffect(() => {
    let stopped = false;
    void listRuns()
      .then((storedRuns) => {
        if (stopped) return;
        const completedRuns = storedRuns.filter((run) => run.status === "completed");
        setRuns(completedRuns);
        setSelectedRunId((current) => {
          if (preferredRunId && completedRuns.some((run) => run.id === preferredRunId)) {
            return preferredRunId;
          }
          if (current && completedRuns.some((run) => run.id === current)) return current;
          return completedRuns[0]?.id ?? null;
        });
        setHistoryError(null);
      })
      .catch((error: unknown) => {
        if (!stopped) setHistoryError(errorMessage(error));
      })
      .finally(() => {
        if (!stopped) setHistoryLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [historyRevision, preferredRunId, refreshKey]);

  useEffect(() => {
    if (!selectedRunId) return;
    let stopped = false;
    void Promise.all([getRunMetrics(selectedRunId), listRunResults(selectedRunId)])
      .then(([metrics, results]) => {
        if (stopped) return;
        setEvidence({ runId: selectedRunId, metrics, results });
        setEvidenceError(null);
      })
      .catch((error: unknown) => {
        if (!stopped) setEvidenceError({ runId: selectedRunId, message: errorMessage(error) });
      });
    return () => {
      stopped = true;
    };
  }, [evidenceRevision, selectedRunId]);

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const activeEvidence = evidence?.runId === selectedRunId ? evidence : null;
  const activeError = evidenceError?.runId === selectedRunId ? evidenceError.message : null;
  const categories = useMemo(
    () => [...new Set(activeEvidence?.results.map((result) => result.category) ?? [])].sort(),
    [activeEvidence],
  );
  const filteredResults = useMemo(() => {
    if (!activeEvidence) return [];
    const needle = query.trim().toLocaleLowerCase();
    return activeEvidence.results.filter((result) => {
      const matchesCategory = category === "all" || result.category === category;
      const matchesOutcome =
        outcome === "all" || (outcome === "passed" ? result.passed : !result.passed);
      const searchable = [
        result.case_id,
        result.category,
        result.prompt,
        result.generation.text,
      ]
        .join(" ")
        .toLocaleLowerCase();
      return matchesCategory && matchesOutcome && (!needle || searchable.includes(needle));
    });
  }, [activeEvidence, category, outcome, query]);

  return (
    <section className="results-workspace" aria-labelledby="results-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Analyze</p>
          <h2 id="results-title">Completed evaluations</h2>
        </div>
        <p>Inspect aggregate performance, category balance, and the evidence behind every grade.</p>
      </div>

      {historyError && (
        <div className="alert alert-error" role="alert">
          <div>
            <strong>Couldn’t load run history.</strong>
            <span>{historyError}</span>
          </div>
          <button
            className="button button-secondary"
            type="button"
            onClick={() => {
              setHistoryLoading(true);
              setHistoryRevision((revision) => revision + 1);
            }}
          >
            Retry history
          </button>
        </div>
      )}

      {historyLoading ? (
        <div className="results-placeholder" aria-live="polite">Loading completed runs…</div>
      ) : runs.length === 0 && !historyError ? (
        <div className="results-placeholder">
          <strong>No completed evaluations yet.</strong>
          <span>Finish a run above and its evidence will appear here automatically.</span>
        </div>
      ) : runs.length > 0 ? (
        <div className="results-layout">
          <aside className="run-history" aria-label="Completed run history">
            <div className="subheading">
              <span>Run history</span>
              <small>{runs.length} completed</small>
            </div>
            <div className="run-list">
              {runs.map((storedRun) => (
                <button
                  className={`run-card${storedRun.id === selectedRunId ? " selected" : ""}`}
                  type="button"
                  key={storedRun.id}
                  aria-pressed={storedRun.id === selectedRunId}
                  onClick={() => setSelectedRunId(storedRun.id)}
                >
                  <span className="run-card-topline">
                    <strong>{storedRun.model}</strong>
                    <b>{percent(storedRun.accuracy)}</b>
                  </span>
                  <span>{storedRun.suite_name} · v{storedRun.suite_version}</span>
                  <small>{formatDate(storedRun.created_at)}</small>
                </button>
              ))}
            </div>
          </aside>

          <div className="evidence-area">
            {selectedRun && (
              <div className="selected-run-heading">
                <div>
                  <p className="eyebrow">Selected run</p>
                  <h3>{selectedRun.model}</h3>
                </div>
                <code title={selectedRun.id}>{selectedRun.id}</code>
              </div>
            )}

            {activeError ? (
              <div className="alert alert-error" role="alert">
                <div>
                  <strong>Couldn’t load run evidence.</strong>
                  <span>{activeError}</span>
                </div>
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() => setEvidenceRevision((revision) => revision + 1)}
                >
                  Retry evidence
                </button>
              </div>
            ) : !activeEvidence ? (
              <div className="results-placeholder" aria-live="polite">Loading metrics and evidence…</div>
            ) : (
              <>
                <div className="metric-grid" aria-label="Run metrics">
                  <article className="metric-card metric-primary">
                    <span>Accuracy</span>
                    <strong>{percent(activeEvidence.metrics.accuracy)}</strong>
                    <small>
                      95% CI {percent(activeEvidence.metrics.ci_low)}–{percent(activeEvidence.metrics.ci_high)}
                    </small>
                  </article>
                  <article className="metric-card">
                    <span>Cases</span>
                    <strong>{activeEvidence.metrics.case_count}</strong>
                    <small>graded prompts</small>
                  </article>
                  <article className="metric-card">
                    <span>Median latency</span>
                    <strong>{decimal(activeEvidence.metrics.latency_ms_p50, "ms")}</strong>
                    <small>p95 {decimal(activeEvidence.metrics.latency_ms_p95, "ms")}</small>
                  </article>
                  <article className="metric-card">
                    <span>Throughput</span>
                    <strong>{decimal(activeEvidence.metrics.tokens_per_second, "tok/s")}</strong>
                    <small>generated tokens</small>
                  </article>
                </div>

                <section className="category-panel" aria-labelledby="category-title">
                  <div className="subheading">
                    <span id="category-title">Category accuracy</span>
                    <small>pass rate</small>
                  </div>
                  <div className="category-chart">
                    {Object.entries(activeEvidence.metrics.category_accuracy)
                      .sort(([left], [right]) => left.localeCompare(right))
                      .map(([name, accuracy]) => (
                        <div className="category-row" key={name}>
                          <span>{name}</span>
                          <div
                            className="category-track"
                            role="img"
                            aria-label={`${name}: ${percent(accuracy)}`}
                          >
                            <i style={{ width: `${accuracy * 100}%` }} />
                          </div>
                          <strong>{percent(accuracy)}</strong>
                        </div>
                      ))}
                  </div>
                </section>

                <section className="case-explorer" aria-labelledby="cases-title">
                  <div className="subheading">
                    <span id="cases-title">Case evidence</span>
                    <small>{filteredResults.length} of {activeEvidence.results.length}</small>
                  </div>
                  <div className="case-filters">
                    <label>
                      <span>Search evidence</span>
                      <input
                        type="search"
                        value={query}
                        placeholder="Case ID, prompt, or output"
                        onChange={(event) => setQuery(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Category</span>
                      <select value={category} onChange={(event) => setCategory(event.target.value)}>
                        <option value="all">All categories</option>
                        {categories.map((name) => <option key={name}>{name}</option>)}
                      </select>
                    </label>
                    <label>
                      <span>Outcome</span>
                      <select value={outcome} onChange={(event) => setOutcome(event.target.value)}>
                        <option value="all">All outcomes</option>
                        <option value="passed">Passed</option>
                        <option value="failed">Failed</option>
                      </select>
                    </label>
                  </div>

                  {filteredResults.length === 0 ? (
                    <div className="filter-empty">No cases match the current filters.</div>
                  ) : (
                    <div className="case-list">
                      {filteredResults.map((result) => (
                        <details className="case-card" key={result.case_id}>
                          <summary>
                            <span className={`case-outcome ${result.passed ? "passed" : "failed"}`}>
                              {result.passed ? "Pass" : "Fail"}
                            </span>
                            <span className="case-summary-copy">
                              <strong>{result.case_id}</strong>
                              <small>{result.category}</small>
                            </span>
                            <span className="case-prompt-preview">{result.prompt}</span>
                          </summary>
                          <div className="case-detail">
                            <div>
                              <span>Prompt</span>
                              <pre>{result.prompt}</pre>
                            </div>
                            <div>
                              <span>Model output</span>
                              <pre>{result.generation.text}</pre>
                            </div>
                            <div>
                              <span>Expected</span>
                              <pre>{formatExpected(result.expected)}</pre>
                            </div>
                            <div className="judgment-list">
                              <span>Judgments</span>
                              {result.judgments.map((judgment, index) => (
                                <article key={`${judgment.grader_type}-${index}`}>
                                  <strong>{judgment.grader_type}</strong>
                                  <b className={judgment.passed ? "passed" : "failed"}>
                                    {judgment.passed ? "Passed" : "Failed"} · {judgment.score.toFixed(2)}
                                  </b>
                                  <p>{judgment.rationale}</p>
                                </article>
                              ))}
                            </div>
                          </div>
                        </details>
                      ))}
                    </div>
                  )}
                </section>
              </>
            )}
          </div>
        </div>
      ) : null}
    </section>
  );
}
