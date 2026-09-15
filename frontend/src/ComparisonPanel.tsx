import { FormEvent, useMemo, useState } from "react";
import { compareRuns, RunComparison, RunSummary } from "./api";

interface ComparisonPanelProps {
  runs: RunSummary[];
  preferredCandidateId: string | null;
}

interface LoadedComparison {
  baselineId: string;
  candidateId: string;
  value: RunComparison;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function percentagePoints(value: number): string {
  const points = value * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pp`;
}

function pValue(value: number): string {
  if (value < 0.0001) return "< 0.0001";
  return value.toFixed(4);
}

function compatible(left: RunSummary, right: RunSummary): boolean {
  return left.suite_name === right.suite_name && left.suite_version === right.suite_version;
}

export default function ComparisonPanel({ runs, preferredCandidateId }: ComparisonPanelProps) {
  const [candidateSelection, setCandidateSelection] = useState("");
  const [baselineSelection, setBaselineSelection] = useState("");
  const [comparison, setComparison] = useState<LoadedComparison | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const candidateRuns = useMemo(
    () => runs.filter((run) => runs.some((other) => other.id !== run.id && compatible(run, other))),
    [runs],
  );
  const candidateId = candidateRuns.some((run) => run.id === candidateSelection)
    ? candidateSelection
    : candidateRuns.some((run) => run.id === preferredCandidateId)
      ? preferredCandidateId!
      : (candidateRuns[0]?.id ?? "");
  const candidate = runs.find((run) => run.id === candidateId) ?? null;
  const baselineRuns = candidate
    ? runs.filter((run) => run.id !== candidate.id && compatible(run, candidate))
    : [];
  const baselineId = baselineRuns.some((run) => run.id === baselineSelection)
    ? baselineSelection
    : (baselineRuns[0]?.id ?? "");
  const baseline = runs.find((run) => run.id === baselineId) ?? null;
  const activeComparison =
    comparison?.baselineId === baselineId && comparison.candidateId === candidateId
      ? comparison.value
      : null;

  async function handleCompare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!baselineId || !candidateId || baselineId === candidateId || loading) return;
    setLoading(true);
    setComparisonError(null);
    try {
      const value = await compareRuns(baselineId, candidateId);
      setComparison({ baselineId, candidateId, value });
    } catch (error) {
      setComparisonError(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="comparison-panel" aria-labelledby="comparison-title">
      <div className="subheading">
        <span id="comparison-title">Paired comparison</span>
        <small>exact McNemar test · α 0.05</small>
      </div>

      {candidateRuns.length === 0 ? (
        <div className="comparison-empty">
          Complete a second run of the same suite version to unlock paired comparison.
        </div>
      ) : (
        <>
          <form className="comparison-form" onSubmit={handleCompare}>
            <label>
              <span>Baseline</span>
              <select
                value={baselineId}
                onChange={(event) => setBaselineSelection(event.target.value)}
                disabled={loading}
              >
                {baselineRuns.map((run) => (
                  <option key={run.id} value={run.id}>{run.model} · {percent(run.accuracy ?? 0)}</option>
                ))}
              </select>
            </label>
            <span className="comparison-arrow" aria-hidden="true">→</span>
            <label>
              <span>Candidate</span>
              <select
                value={candidateId}
                onChange={(event) => {
                  setCandidateSelection(event.target.value);
                  setBaselineSelection("");
                }}
                disabled={loading}
              >
                {candidateRuns.map((run) => (
                  <option key={run.id} value={run.id}>{run.model} · {percent(run.accuracy ?? 0)}</option>
                ))}
              </select>
            </label>
            <button className="button button-secondary" type="submit" disabled={loading}>
              {loading ? "Comparing…" : "Compare runs"}
            </button>
          </form>

          {comparisonError && (
            <p className="inline-error comparison-error" role="alert">{comparisonError}</p>
          )}

          {activeComparison && baseline && candidate && (
            <div className="comparison-result" aria-live="polite">
              <div className="comparison-summary">
                <article>
                  <span>Baseline · {baseline.model}</span>
                  <strong>{percent(activeComparison.baseline_accuracy)}</strong>
                </article>
                <article className={activeComparison.accuracy_delta >= 0 ? "positive" : "negative"}>
                  <span>Accuracy delta</span>
                  <strong>{percentagePoints(activeComparison.accuracy_delta)}</strong>
                </article>
                <article>
                  <span>Candidate · {candidate.model}</span>
                  <strong>{percent(activeComparison.candidate_accuracy)}</strong>
                </article>
              </div>

              <div className={`significance-note${activeComparison.mcnemar.significant ? " significant" : ""}`}>
                <strong>
                  {activeComparison.mcnemar.significant
                    ? "Statistically significant difference"
                    : "No statistically significant difference"}
                </strong>
                <span>
                  Exact p = {pValue(activeComparison.mcnemar.exact_p_value)} across {activeComparison.case_count} paired cases
                </span>
              </div>

              <div className="comparison-grid">
                <div className="contingency" aria-label="Paired outcome counts">
                  <article>
                    <strong>{activeComparison.mcnemar.both_passed}</strong>
                    <span>Both passed</span>
                  </article>
                  <article className="improvement-cell">
                    <strong>{activeComparison.mcnemar.candidate_only_passed}</strong>
                    <span>Candidate only</span>
                  </article>
                  <article className="regression-cell">
                    <strong>{activeComparison.mcnemar.baseline_only_passed}</strong>
                    <span>Baseline only</span>
                  </article>
                  <article>
                    <strong>{activeComparison.mcnemar.both_failed}</strong>
                    <span>Both failed</span>
                  </article>
                </div>
                <div className="change-lists">
                  <div>
                    <span className="improvement-label">Improvements · {activeComparison.improvements.length}</span>
                    {activeComparison.improvements.length > 0 ? (
                      <ul>{activeComparison.improvements.map((id) => <li key={id}>{id}</li>)}</ul>
                    ) : <p>No candidate improvements.</p>}
                  </div>
                  <div>
                    <span className="regression-label">Regressions · {activeComparison.regressions.length}</span>
                    {activeComparison.regressions.length > 0 ? (
                      <ul>{activeComparison.regressions.map((id) => <li key={id}>{id}</li>)}</ul>
                    ) : <p>No candidate regressions.</p>}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
