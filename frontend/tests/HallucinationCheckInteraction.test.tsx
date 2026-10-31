// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import HallucinationCheckPanel, {
  HallucinationResultView,
} from "../components/HallucinationCheckPanel";
import { useWorkspace } from "../components/WorkspaceProvider";
import {
  ApiError,
  checkHallucinations,
  getHallucinationResult,
} from "../lib/api";
import type {
  HallucinationCheckResponse,
  RagQueryResponse,
} from "../lib/types";

import { latestRagResult } from "./ragResultFixture";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    checkHallucinations: vi.fn(),
    getHallucinationResult: vi.fn(),
  };
});

vi.mock("../components/WorkspaceProvider", () => ({
  useWorkspace: vi.fn(),
}));

const detectionResult: HallucinationCheckResponse = {
  check_id: "check-1",
  result_id: "result-1",
  question: "When was Acme founded?",
  model_name: "Horizon-Labs/hallucination-guard-base",
  model_input_order: "source_then_claim",
  support_threshold: 0.5,
  review_margin: 0.05,
  claims: [
    {
      claim_id: "claim-1",
      sentence_index: 0,
      sentence: "Acme was founded in 2015.",
      claim: "Acme was founded in 2015.",
      claim_start: 0,
      claim_end: 25,
      verdict: "potentially_hallucinated",
      raw_model_label: "UNSUPPORTED",
      support_probability: 0.12,
      explanation: "The supplied evidence did not support the claim.",
      evidence: [
        {
          evidence_id: "evidence-1",
          document_id: "doc-1",
          filename: "iso-27001.txt",
          chunk_id: "chunk-1",
          chunk_index: 4,
          page_number: 2,
          retrieval_score: 0.12,
          retrieval_similarity: 0.88,
          support_probability: 0.12,
          raw_model_label: "UNSUPPORTED",
        },
      ],
      nli_status: "supported",
      nli_confidence: 0.94,
      nli_disagrees: true,
    },
  ],
  summary: {
    claims_total: 1,
    supported: 0,
    potentially_hallucinated: 1,
    nli_disagreements: 1,
    evidence_total: 1,
    pairs_evaluated: 1,
  },
  timings: {
    claim_extraction_ms: 1,
    inference_ms: 20,
    total_ms: 21,
  },
  warnings: [],
};

function workspaceState(
  result: RagQueryResponse | null,
  clearLatestRagResult = vi.fn(),
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
    setLatestRagResult: vi.fn(),
    clearLatestRagResult,
    refreshAll: vi.fn(),
    refreshing: false,
  };
}

const mockedCheck = vi.mocked(checkHallucinations);
const mockedGetResult = vi.mocked(getHallucinationResult);
const mockedWorkspace = vi.mocked(useWorkspace);

beforeEach(() => {
  vi.clearAllMocks();
  mockedGetResult.mockRejectedValue(new ApiError("not found", 404));
  mockedWorkspace.mockReturnValue(workspaceState(latestRagResult));
});

afterEach(() => {
  cleanup();
});

describe("HallucinationCheckPanel interaction", () => {
  test("restores the saved result without running another check", async () => {
    mockedGetResult.mockResolvedValue(detectionResult);

    render(<HallucinationCheckPanel />);

    expect(await screen.findByText("Groundedness summary")).toBeTruthy();
    expect(mockedGetResult).toHaveBeenCalledWith(
      "result-1",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(mockedCheck).not.toHaveBeenCalled();
  });

  test("renders only the compact summary and original answer", () => {
    render(<HallucinationCheckPanel />);

    expect(screen.getByText("When was Acme founded?")).toBeTruthy();
    expect(screen.getByText("openai/gpt-oss-120b")).toBeTruthy();
    expect(screen.getByText("Evidence chunks")).toBeTruthy();
    expect(screen.getByText("1")).toBeTruthy();
    expect(screen.getByText("Existing NLI")).toBeTruthy();
    expect(screen.getByText(/1 supported · 0 contradicted · 0 neutral/)).toBeTruthy();
    expect(screen.getByText("Ask result ID")).toBeTruthy();
    expect(screen.getByText("result-1")).toBeTruthy();
    expect(
      screen.getByText("Acme was founded in 2015.", {
        selector: ".answer-text",
      }),
    ).toBeTruthy();

    expect(screen.queryByText("Reuse latest Ask result")).toBeNull();
    expect(
      screen.queryByText("This page never reruns retrieval or answer generation."),
    ).toBeNull();
    expect(
      screen.queryByRole("heading", { name: "Retrieved evidence" }),
    ).toBeNull();
    expect(screen.queryByText("iso-27001.txt")).toBeNull();
    expect(screen.queryByText(/Acme was founded in 2015 in Paris\./)).toBeNull();
    expect(
      screen.queryByRole("heading", {
        name: "Existing NLI claim verification",
      }),
    ).toBeNull();
    expect(screen.queryByText(/NLI scores:/)).toBeNull();
  });

  test("sends one request for rapid clicks, shows loading, and renders the response", async () => {
    let resolveRequest!: (value: HallucinationCheckResponse) => void;
    const pending = new Promise<HallucinationCheckResponse>((resolve) => {
      resolveRequest = resolve;
    });
    mockedCheck.mockReturnValue(pending);

    render(<HallucinationCheckPanel />);
    const button = screen.getByRole("button", {
      name: "Check for Hallucinations",
    });

    fireEvent.click(button);
    fireEvent.click(button);

    expect(mockedCheck).toHaveBeenCalledTimes(1);
    expect(mockedCheck).toHaveBeenCalledWith(
      { result_id: "result-1" },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(
      (screen.getByRole("button", {
        name: "Checking...",
      }) as HTMLButtonElement).disabled,
    ).toBe(true);

    resolveRequest(detectionResult);

    expect(await screen.findByText("Groundedness summary")).toBeTruthy();
    expect(
      screen.getAllByText("Potentially hallucinated").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("UNSUPPORTED").length).toBeGreaterThan(0);
    expect(
      (screen.getByRole("button", {
        name: "Check for Hallucinations",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  test("keeps evidence references available through Inspect", () => {
    render(
      <HallucinationResultView
        answer="Acme was founded in 2015."
        check={detectionResult}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Inspect" }));

    expect(
      screen.getByText(
        "The dedicated detector and independent NLI result disagree.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("support 12.0%")).toBeTruthy();
    expect(
      screen.getByText("iso-27001.txt", { selector: ".result-file" }),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(
      screen.queryByText(
        "The dedicated detector and independent NLI result disagree.",
      ),
    ).toBeNull();
  });

  test("shows an actionable error and re-enables the button", async () => {
    mockedCheck.mockRejectedValue(
      new ApiError("The local hallucination model failed.", 503),
    );

    render(<HallucinationCheckPanel />);
    const button = screen.getByRole("button", {
      name: "Check for Hallucinations",
    });
    fireEvent.click(button);

    expect(
      await screen.findByText("The local hallucination model failed."),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Check for Hallucinations",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  test("does not offer a check without a completed Ask result", () => {
    mockedWorkspace.mockReturnValue(workspaceState(null));

    render(<HallucinationCheckPanel />);

    expect(screen.getByText("No Ask result to check")).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Check for Hallucinations" }),
    ).toBeNull();
  });

  test("clears the shared result through the existing action", () => {
    const clearLatestRagResult = vi.fn();
    mockedWorkspace.mockReturnValue(
      workspaceState(latestRagResult, clearLatestRagResult),
    );

    render(<HallucinationCheckPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Clear shared result" }));

    expect(clearLatestRagResult).toHaveBeenCalledTimes(1);
  });
});
