// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import AskPanel from "../components/AskPanel";
import { useWorkspace } from "../components/WorkspaceProvider";
import { ApiError, askQuestion, verifyRagResult } from "../lib/api";
import type { RagQueryResponse } from "../lib/types";

import { latestRagResult } from "./ragResultFixture";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    askQuestion: vi.fn(),
    verifyRagResult: vi.fn(),
  };
});

vi.mock("../components/WorkspaceProvider", () => ({
  useWorkspace: vi.fn(),
}));

const mockedAsk = vi.mocked(askQuestion);
const mockedVerify = vi.mocked(verifyRagResult);
const mockedWorkspace = vi.mocked(useWorkspace);

const pendingResult: RagQueryResponse = {
  ...latestRagResult,
  result_id: "result-2",
  answer_mode: "verification_pending",
  answer_status: "verification_pending",
  verification: {
    claim_level_verification: false,
    status: "pending",
    method: "none",
    note: "Claim-level verification has not been run.",
  },
  timings: {
    ...latestRagResult.timings,
    verification_ms: 0,
  },
};

function workspaceState(
  result: RagQueryResponse | null,
  setLatestRagResult = vi.fn(),
) {
  return {
    health: null,
    healthLoading: false,
    healthError: null,
    refreshHealth: vi.fn(),
    documents: [],
    documentsLoading: false,
    documentsError: null,
    refreshDocuments: vi.fn(),
    latestRagResult: result,
    latestRagResultLoading: false,
    latestRagResultError: null,
    setLatestRagResult,
    clearLatestRagResult: vi.fn(),
    refreshAll: vi.fn(),
    refreshing: false,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedWorkspace.mockReturnValue(workspaceState(latestRagResult));
});

afterEach(() => {
  cleanup();
});

describe("AskPanel shared result rendering", () => {
  test("renders the shared result, evidence, and NLI results after navigation", () => {
    render(<AskPanel />);

    expect(screen.getByText("Question: When was Acme founded?")).toBeTruthy();
    expect(
      screen.getAllByText("Acme was founded in 2015.").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("Retrieved evidence (1)")).toBeTruthy();
    expect(screen.getAllByText("iso-27001.txt").length).toBeGreaterThan(0);
    expect(screen.getAllByText("page 2").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText("Acme was founded in 2015 in Paris.").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/1\/1 fully evaluated/)).toBeTruthy();
    expect(
      screen.getByText(
        "NLI scores: entailment 0.9400 / contradiction 0.0100 / neutral 0.0500",
      ),
    ).toBeTruthy();
  });

  test("requests exactly five evidence chunks and clears the previous result", async () => {
    const setLatestRagResult = vi.fn();
    const clearLatestRagResult = vi.fn();
    const state = workspaceState(latestRagResult, setLatestRagResult);
    state.clearLatestRagResult = clearLatestRagResult;
    mockedWorkspace.mockReturnValue(state);
    mockedAsk.mockResolvedValue(pendingResult);

    render(<AskPanel />);
    expect(screen.queryByLabelText(/top_k/i)).toBeNull();
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "What changed?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask documents" }));

    await waitFor(() => {
      expect(setLatestRagResult).toHaveBeenCalledWith(pendingResult);
    });
    expect(mockedAsk).toHaveBeenCalledWith(
      {
        question: "What changed?",
        top_k: 5,
        document_ids: undefined,
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(clearLatestRagResult).toHaveBeenCalledTimes(1);
  });

  test("does not run NLI until Verify Answer Claims is pressed", async () => {
    const setLatestRagResult = vi.fn();
    mockedWorkspace.mockReturnValue(
      workspaceState(pendingResult, setLatestRagResult),
    );

    render(<AskPanel />);

    expect(screen.getByText("Question: When was Acme founded?")).toBeTruthy();
    expect(
      screen.getAllByText("Acme was founded in 2015.").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("Retrieved evidence (1)")).toBeTruthy();
    expect(screen.queryByText(/NLI scores:/)).toBeNull();
    expect(screen.queryByTestId("verification-claim")).toBeNull();

    let resolveVerification!: (value: RagQueryResponse) => void;
    const pendingVerification = new Promise<RagQueryResponse>((resolve) => {
      resolveVerification = resolve;
    });
    mockedVerify.mockReturnValue(pendingVerification);

    const button = screen.getByRole("button", {
      name: "Verify Answer Claims",
    });
    fireEvent.click(button);
    fireEvent.click(button);

    expect(mockedVerify).toHaveBeenCalledTimes(1);
    expect(mockedVerify).toHaveBeenCalledWith(
      "result-2",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(
      (screen.getByRole("button", {
        name: "Verifying claims...",
      }) as HTMLButtonElement).disabled,
    ).toBe(true);

    resolveVerification(latestRagResult);

    await waitFor(() => {
      expect(setLatestRagResult).toHaveBeenCalledWith(latestRagResult);
    });
  });

  test("shows a verification error and allows a retry", async () => {
    mockedWorkspace.mockReturnValue(workspaceState(pendingResult));
    mockedVerify.mockRejectedValue(new ApiError("NLI service is busy.", 503));

    render(<AskPanel />);
    fireEvent.click(
      screen.getByRole("button", { name: "Verify Answer Claims" }),
    );

    expect(await screen.findByText("NLI service is busy.")).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Verify Answer Claims",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });
});
