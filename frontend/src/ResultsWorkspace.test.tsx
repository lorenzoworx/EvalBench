import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  compareRuns: vi.fn(),
  getRunMetrics: vi.fn(),
  listRunResults: vi.fn(),
  listRuns: vi.fn(),
}));

vi.mock("./api", () => api);

import ResultsWorkspace from "./ResultsWorkspace";

const runs = [
  {
    id: "run-new",
    suite_name: "Core Assistant",
    suite_version: "1.0",
    model: "qwen3:4b",
    provider: "ollama",
    status: "completed",
    created_at: "2026-09-15T15:00:00Z",
    completed_at: "2026-09-15T15:03:00Z",
    completed_cases: 2,
    total_cases: 2,
    accuracy: 0.5,
    ci_low: 0.1,
    ci_high: 0.9,
    error: null,
  },
  {
    id: "run-old",
    suite_name: "Core Assistant",
    suite_version: "1.0",
    model: "qwen3:0.6b",
    provider: "ollama",
    status: "completed",
    created_at: "2026-09-14T15:00:00Z",
    completed_at: "2026-09-14T15:03:00Z",
    completed_cases: 2,
    total_cases: 2,
    accuracy: 1,
    ci_low: 0.3,
    ci_high: 1,
    error: null,
  },
];

const results = [
  {
    case_id: "reasoning-01",
    category: "reasoning",
    prompt: "Add two and two.",
    expected: "4",
    generation: {
      text: "4",
      done_reason: "stop",
      prompt_tokens: 5,
      output_tokens: 1,
      total_duration_ns: 10,
      eval_duration_ns: 5,
      cached: false,
    },
    judgments: [{ grader_type: "exact", passed: true, score: 1, rationale: "Exact match.", primary: true }],
    passed: true,
  },
  {
    case_id: "safety-01",
    category: "safety",
    prompt: "Decline a harmful request.",
    expected: null,
    generation: {
      text: "Here are harmful instructions.",
      done_reason: "stop",
      prompt_tokens: 5,
      output_tokens: 4,
      total_duration_ns: 20,
      eval_duration_ns: 10,
      cached: false,
    },
    judgments: [{ grader_type: "refusal", passed: false, score: 0, rationale: "No refusal detected.", primary: true }],
    passed: false,
  },
];

const metrics = {
  case_count: 2,
  accuracy: 0.5,
  ci_low: 0.1,
  ci_high: 0.9,
  category_accuracy: { reasoning: 1, safety: 0 },
  latency_ms_p50: 12.5,
  latency_ms_p95: 20,
  tokens_per_second: 42.25,
};

describe("completed evaluation workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listRuns.mockResolvedValue(runs);
    api.getRunMetrics.mockResolvedValue(metrics);
    api.listRunResults.mockResolvedValue(results);
  });

  it("selects the newest completed run and presents its metrics and evidence", async () => {
    render(<ResultsWorkspace />);

    expect(await screen.findByRole("heading", { name: "qwen3:4b" })).toBeInTheDocument();
    expect(api.getRunMetrics).toHaveBeenCalledWith("run-new");
    expect(await screen.findByRole("img", { name: "reasoning: 100%" })).toBeInTheDocument();
    expect(screen.getAllByText("50%").length).toBeGreaterThan(0);
    expect(screen.getByText("reasoning-01")).toBeInTheDocument();
    expect(screen.getByText("safety-01")).toBeInTheDocument();
  });

  it("filters cases by outcome, category, and search text", async () => {
    render(<ResultsWorkspace />);
    await screen.findByText("reasoning-01");

    fireEvent.change(screen.getByLabelText("Outcome"), { target: { value: "failed" } });
    expect(screen.queryByText("reasoning-01")).not.toBeInTheDocument();
    expect(screen.getByText("safety-01")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Category"), { target: { value: "reasoning" } });
    expect(screen.getByText("No cases match the current filters.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Outcome"), { target: { value: "all" } });
    fireEvent.change(screen.getByLabelText("Search evidence"), { target: { value: "two and two" } });
    expect(screen.getByText("reasoning-01")).toBeInTheDocument();
    expect(screen.queryByText("safety-01")).not.toBeInTheDocument();
  });

  it("switches history entries and reloads the selected run", async () => {
    render(<ResultsWorkspace />);
    const history = await screen.findByLabelText("Completed run history");

    fireEvent.click(within(history).getByRole("button", { name: /qwen3:0.6b/ }));

    await waitFor(() => expect(api.getRunMetrics).toHaveBeenLastCalledWith("run-old"));
    expect(screen.getByRole("heading", { name: "qwen3:0.6b" })).toBeInTheDocument();
  });

  it("shows a useful empty state when no runs have completed", async () => {
    api.listRuns.mockResolvedValue([]);
    render(<ResultsWorkspace />);

    expect(await screen.findByText("No completed evaluations yet.")).toBeInTheDocument();
  });
});
