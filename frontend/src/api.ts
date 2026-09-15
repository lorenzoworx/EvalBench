export type RunState =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface SuiteSummary {
  id: string;
  name: string;
  version: string;
  case_count: number;
  categories: string[];
}

export interface RunAccepted {
  id: string;
  status: "pending";
}

export interface RunStatus {
  id: string;
  status: RunState;
  completed_cases: number;
  total_cases: number;
  error: string | null;
}

export interface RunSummary extends RunStatus {
  suite_name: string;
  suite_version: string;
  model: string;
  provider: string;
  created_at: string;
  completed_at: string | null;
  accuracy: number | null;
  ci_low: number | null;
  ci_high: number | null;
}

export interface RunMetrics {
  case_count: number;
  accuracy: number | null;
  ci_low: number | null;
  ci_high: number | null;
  category_accuracy: Record<string, number>;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  tokens_per_second: number | null;
}

export interface Generation {
  text: string;
  done_reason: string | null;
  prompt_tokens: number | null;
  output_tokens: number | null;
  total_duration_ns: number | null;
  eval_duration_ns: number | null;
  cached: boolean;
}

export interface Judgment {
  grader_type: string;
  passed: boolean;
  score: number;
  rationale: string;
  primary: boolean;
}

export interface CaseResult {
  case_id: string;
  category: string;
  prompt: string;
  expected: unknown | null;
  generation: Generation;
  judgments: Judgment[];
  passed: boolean;
}

interface ErrorPayload {
  detail?: string;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}.`;
    try {
      const payload = (await response.json()) as ErrorPayload;
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based fallback when a proxy or server returns non-JSON.
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export function listModels(): Promise<string[]> {
  return request<string[]>("/api/models");
}

export function listSuites(): Promise<SuiteSummary[]> {
  return request<SuiteSummary[]>("/api/suites");
}

export function startRun(suiteId: string, model: string): Promise<RunAccepted> {
  return request<RunAccepted>("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suite_id: suiteId, model }),
  });
}

export function getRunStatus(runId: string): Promise<RunStatus> {
  return request<RunStatus>(`/api/runs/${encodeURIComponent(runId)}/status`);
}

export function cancelRun(runId: string): Promise<void> {
  return request(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
}

export function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/api/runs");
}

export function getRunMetrics(runId: string): Promise<RunMetrics> {
  return request<RunMetrics>(`/api/runs/${encodeURIComponent(runId)}/metrics`);
}

export function listRunResults(runId: string): Promise<CaseResult[]> {
  return request<CaseResult[]>(`/api/runs/${encodeURIComponent(runId)}/results`);
}
