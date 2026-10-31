import { afterEach, expect, test, vi } from "vitest";
import { replayRequest } from "./replay";
afterEach(() => vi.unstubAllGlobals());
test("read-only replay refuses inference writes without contacting any endpoint", async () => {
  const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher);
  await expect(replayRequest("/api/runs", "POST")).rejects.toMatchObject({ status: 403 });
  await expect(replayRequest("/api/runs/example/cancel", "POST")).rejects.toMatchObject({ status: 403 });
  expect(fetcher).not.toHaveBeenCalled();
});
