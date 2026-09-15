import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({ compareRuns: vi.fn() }));
vi.mock("./api", () => api);

import ComparisonPanel from "./ComparisonPanel";

const baseRun = {
  suite_name: "Core Assistant",
  suite_version: "1.0",
  provider: "ollama",
  status: "completed" as const,
  created_at: "2026-09-15T15:00:00Z",
  completed_at: "2026-09-15T15:03:00Z",
  completed_cases: 60,
  total_cases: 60,
  ci_low: 0.5,
  ci_high: 0.8,
  error: null,
};
const runs = [
  { ...baseRun, id: "candidate", model: "qwen3:4b", accuracy: 0.75 },
  { ...baseRun, id: "baseline", model: "qwen3:0.6b", accuracy: 0.65 },
];
const comparison = {
  baseline_run_id: "baseline",
  candidate_run_id: "candidate",
  case_count: 60,
  baseline_accuracy: 0.65,
  candidate_accuracy: 0.75,
  accuracy_delta: 0.1,
  regressions: ["safety-03"],
  improvements: ["reasoning-02", "json-04"],
  mcnemar: {
    both_passed: 36,
    baseline_only_passed: 3,
    candidate_only_passed: 9,
    both_failed: 12,
    exact_p_value: 0.0386,
    alpha: 0.05,
    significant: true,
  },
};

describe("paired run comparison", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.compareRuns.mockResolvedValue(comparison);
  });

  it("compares compatible runs and explains the paired result", async () => {
    render(<ComparisonPanel runs={runs} preferredCandidateId="candidate" />);

    expect(screen.getByLabelText("Baseline")).toHaveValue("baseline");
    expect(screen.getByLabelText("Candidate")).toHaveValue("candidate");
    fireEvent.click(screen.getByRole("button", { name: "Compare runs" }));

    await waitFor(() => expect(api.compareRuns).toHaveBeenCalledWith("baseline", "candidate"));
    expect(await screen.findByText("Statistically significant difference")).toBeInTheDocument();
    expect(screen.getByText("+10.0 pp")).toBeInTheDocument();
    expect(screen.getByText("Exact p = 0.0386 across 60 paired cases")).toBeInTheDocument();
    expect(screen.getByText("reasoning-02")).toBeInTheDocument();
    expect(screen.getByText("safety-03")).toBeInTheDocument();
  });

  it("offers only runs with matching suite versions", () => {
    const incompatible = {
      ...baseRun,
      id: "other-suite",
      model: "gemma3:4b",
      suite_version: "2.0",
      accuracy: 0.9,
    };
    render(<ComparisonPanel runs={[runs[0], incompatible]} preferredCandidateId="candidate" />);

    expect(screen.getByText(/complete a second run of the same suite version/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Compare runs" })).not.toBeInTheDocument();
  });

  it("surfaces API validation failures", async () => {
    api.compareRuns.mockRejectedValue(new Error("Paired runs must contain identical case IDs."));
    render(<ComparisonPanel runs={runs} preferredCandidateId="candidate" />);

    fireEvent.click(screen.getByRole("button", { name: "Compare runs" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Paired runs must contain identical case IDs.",
    );
  });
});
