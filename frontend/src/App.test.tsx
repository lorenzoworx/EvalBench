import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("application scaffold", () => {
  it("identifies the current project stage", () => {
    render(<App />);
    expect(screen.getByText("Explainable evaluation for local language models.")).toBeInTheDocument();
    expect(screen.getByText(/Run workflows and results arrive/)).toBeInTheDocument();
  });
});
