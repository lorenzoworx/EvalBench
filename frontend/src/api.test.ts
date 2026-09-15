import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, cancelRun, getRunStatus, listModels, listSuites, startRun } from "./api";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
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
