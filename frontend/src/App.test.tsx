import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  cancelRun: vi.fn(),
  getRunStatus: vi.fn(),
  listModels: vi.fn(),
  listSuites: vi.fn(),
  startRun: vi.fn(),
}));

vi.mock("./api", () => api);

import App from "./App";

const suite = {
  id: "core",
  name: "Core Assistant",
  version: "1.0",
  case_count: 60,
  categories: ["reasoning", "safety"],
};

describe("evaluation control", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    api.listModels.mockResolvedValue(["qwen3:0.6b", "qwen3:4b"]);
    api.listSuites.mockResolvedValue([suite]);
    api.startRun.mockResolvedValue({ id: "run-123", status: "pending" });
    api.cancelRun.mockResolvedValue({ id: "run-123", cancellation_requested: true });
  });

  it("loads local models and suites into the run form", async () => {
    render(<App />);

    expect(await screen.findByRole("option", { name: /Core Assistant/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "qwen3:0.6b" })).toBeInTheDocument();
    expect(screen.getByText("2 categories")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run evaluation" })).toBeEnabled();
  });

  it("starts a run and polls its progress once per second", async () => {
    vi.useFakeTimers();
    api.getRunStatus.mockResolvedValue({
      id: "run-123",
      status: "completed",
      completed_cases: 60,
      total_cases: 60,
      error: null,
    });
    render(<App />);
    await act(async () => Promise.resolve());

    fireEvent.click(screen.getByRole("button", { name: "Run evaluation" }));
    await act(async () => Promise.resolve());

    expect(api.startRun).toHaveBeenCalledWith("core", "qwen3:0.6b");
    expect(screen.getByText("PENDING")).toBeInTheDocument();
    expect(api.getRunStatus).not.toHaveBeenCalled();

    await act(async () => vi.advanceTimersByTimeAsync(1000));

    expect(api.getRunStatus).toHaveBeenCalledTimes(1);
    expect(api.getRunStatus).toHaveBeenCalledWith("run-123");
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.getByText("60 / 60")).toBeInTheDocument();
  });

  it("requests cooperative cancellation and explains the delay", async () => {
    render(<App />);
    await screen.findByRole("button", { name: "Run evaluation" });
    fireEvent.click(screen.getByRole("button", { name: "Run evaluation" }));

    const cancel = await screen.findByRole("button", { name: "Cancel after current case" });
    fireEvent.click(cancel);

    await waitFor(() => expect(api.cancelRun).toHaveBeenCalledWith("run-123"));
    expect(screen.getByText(/current case will finish first/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancellation pending…" })).toBeDisabled();
  });

  it("shows an actionable loading failure and retries", async () => {
    api.listModels.mockRejectedValueOnce(new Error("Ollama is not reachable"));
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Ollama is not reachable");
    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }));

    expect(await screen.findByRole("option", { name: "qwen3:0.6b" })).toBeInTheDocument();
    expect(api.listModels).toHaveBeenCalledTimes(2);
  });

  it("shows a launch failure before a run exists", async () => {
    api.startRun.mockRejectedValueOnce(new Error("Another evaluation is already active"));
    render(<App />);
    await screen.findByRole("button", { name: "Run evaluation" });

    fireEvent.click(screen.getByRole("button", { name: "Run evaluation" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Another evaluation is already active",
    );
  });
});
