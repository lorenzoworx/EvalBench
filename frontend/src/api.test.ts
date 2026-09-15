import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  cancelRun,
  compareRuns,
  getRunMetrics,
  getRunStatus,
  listModels,
  listRunResults,
  listRuns,
  listSuites,
  startRun,
} from "./api";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });

  it("loads stored run summaries, metrics, and case evidence", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse([{ id: "run/one", status: "completed" }]))
      .mockResolvedValueOnce(jsonResponse({ case_count: 60, accuracy: 0.8 }))
      .mockResolvedValueOnce(jsonResponse([{ case_id: "reasoning-01", passed: true }]));
    vi.stubGlobal("fetch", fetchMock);

    await listRuns();
    await getRunMetrics("run/one");
    await listRunResults("run/one");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/runs", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/runs/run%2Fone/metrics", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/api/runs/run%2Fone/results", undefined);
  });

  it("encodes paired comparison inputs and the significance threshold", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ case_count: 60 }));
    vi.stubGlobal("fetch", fetchMock);

    await compareRuns("baseline/one", "candidate two", 0.1);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/compare?baseline_run_id=baseline%2Fone&candidate_run_id=candidate+two&alpha=0.1",
      undefined,
    );
  });
}

describe("API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("maps evaluation endpoints and encodes run IDs", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(["qwen3:0.6b"]))
      .mockResolvedValueOnce(jsonResponse([]))
      .mockResolvedValueOnce(jsonResponse({ id: "run/one", status: "pending" }, 202))
      .mockResolvedValueOnce(
        jsonResponse({
          id: "run/one",
          status: "running",
          completed_cases: 1,
          total_cases: 60,
          error: null,
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ id: "run/one", cancellation_requested: true }, 202));
    vi.stubGlobal("fetch", fetchMock);

    await expect(listModels()).resolves.toEqual(["qwen3:0.6b"]);
    await expect(listSuites()).resolves.toEqual([]);
    await expect(startRun("core", "qwen3:0.6b")).resolves.toEqual({
      id: "run/one",
      status: "pending",
    });
    await getRunStatus("run/one");
    await cancelRun("run/one");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/models", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/suites", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/runs",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ suite_id: "core", model: "qwen3:0.6b" }) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(4, "/api/runs/run%2Fone/status", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(5, "/api/runs/run%2Fone/cancel", { method: "POST" });
  });

  it("surfaces structured API errors with their HTTP status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Run already active." }, 409)),
    );

    const failure = listModels();

    await expect(failure).rejects.toEqual(new ApiError(409, "Run already active."));
  });
});
