import type { CaseResult, RunComparison, RunMetrics, RunSummary } from "./api";
import { ApiError } from "./api";
export interface ReplayData {
  schema_version: number;
  provenance: string;
  runs: RunSummary[];
  metrics: Record<string, RunMetrics>;
  results: Record<string, CaseResult[]>;
  comparisons: RunComparison[];
}
let pending: Promise<ReplayData> | null = null;
async function load(): Promise<ReplayData> {
  pending ??= fetch(`${import.meta.env.BASE_URL}replay-data.json`).then(async (response) => {
    if (!response.ok) throw new ApiError(response.status, "Stored replay data could not be loaded.");
    const data = await response.json() as ReplayData;
    if (data.schema_version !== 1 || !Array.isArray(data.runs)) throw new Error("Unsupported replay dataset.");
    return data;
  }).catch((error: unknown) => { pending = null; throw error; });
  return pending;
}
export async function replayRequest<T>(path: string, method = "GET"): Promise<T> {
  // Guard before reading data: this client can never launch or cancel inference.
  if (method !== "GET") throw new ApiError(403, "This replay is read-only. Run models in the local workbench.");
  const url = new URL(path, "https://replay.invalid");
  const data = await load();
  let result: unknown;
  if (url.pathname === "/api/runs") result = data.runs;
  const match = /^\/api\/runs\/([^/]+)\/(metrics|results)$/.exec(url.pathname);
  if (match) {
    const id = decodeURIComponent(match[1]);
    result = match[2] === "metrics" ? data.metrics[id] : data.results[id];
  }
  if (url.pathname === "/api/compare") {
    const alpha = Number(url.searchParams.get("alpha") ?? "0.05");
    if (alpha !== 0.05) throw new ApiError(400, "Stored comparisons use alpha 0.05.");
    result = data.comparisons.find((item) => item.baseline_run_id === url.searchParams.get("baseline_run_id") && item.candidate_run_id === url.searchParams.get("candidate_run_id"));
  }
  if (result === undefined) throw new ApiError(404, "This resource is not part of the stored replay.");
  return structuredClone(result) as T;
}
