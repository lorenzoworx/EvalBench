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
