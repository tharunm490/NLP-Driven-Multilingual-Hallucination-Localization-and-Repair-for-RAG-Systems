// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import HallucinationRepairPanel from "../components/HallucinationRepairPanel";
import { useWorkspace } from "../components/WorkspaceProvider";
import { getRepairSource, runRepair } from "../lib/api";
import type { RagQueryResponse } from "../lib/types";

import { latestRagResult } from "./ragResultFixture";
import {
  repairResult,
  repairSource,
  repairSourceWithResult,
} from "./repairFixture";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    getRepairSource: vi.fn(),
    runRepair: vi.fn(),
  };
});

vi.mock("../components/WorkspaceProvider", () => ({
  useWorkspace: vi.fn(),
}));

function workspaceState(result: RagQueryResponse | null) {
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
    clearLatestRagResult: vi.fn(),
    refreshAll: vi.fn(),
    refreshing: false,
  };
}

const mockedGetRepairSource = vi.mocked(getRepairSource);
const mockedRunRepair = vi.mocked(runRepair);
const mockedWorkspace = vi.mocked(useWorkspace);

beforeEach(() => {
  vi.clearAllMocks();
  mockedWorkspace.mockReturnValue(workspaceState(latestRagResult));
});

afterEach(() => {
  cleanup();
});

describe("HallucinationRepairPanel interaction", () => {
  test("shows an empty state when there is no Ask result", () => {
    mockedWorkspace.mockReturnValue(workspaceState(null));

    render(<HallucinationRepairPanel />);

    expect(screen.getByText("No Ask result to repair")).toBeTruthy();
    expect(mockedGetRepairSource).not.toHaveBeenCalled();
  });

  test("loads the repair source and runs repair when requested", async () => {
    mockedGetRepairSource.mockResolvedValue(repairSource);
    mockedRunRepair.mockResolvedValue(repairResult);

    render(<HallucinationRepairPanel />);

    expect(await screen.findByText("Repair overview")).toBeTruthy();
    expect(
      screen.getAllByText("Acme was founded in 2015.").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("1 repair candidate(s)")).toBeTruthy();
    expect(mockedGetRepairSource).toHaveBeenCalledWith(
      "result-1",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );

    fireEvent.click(screen.getByRole("button", { name: /repair unsupported claims/i }));

    expect(await screen.findByText("Acme was founded in 2016.")).toBeTruthy();
    expect(mockedRunRepair).toHaveBeenCalledWith(
      { result_id: "result-1", max_attempts: 3 },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(screen.getAllByText("Fully verified").length).toBeGreaterThan(0);
    expect(screen.getByText("Claim-by-claim repair trail")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /inspect/i }));

    expect(await screen.findByText("Attempt 1")).toBeTruthy();
    expect(screen.getByText("Both supported")).toBeTruthy();
  });

  test("shows an existing repair without running it again", async () => {
    mockedGetRepairSource.mockResolvedValue(repairSourceWithResult);

    render(<HallucinationRepairPanel />);

    expect(await screen.findByText("Acme was founded in 2016.")).toBeTruthy();
    expect(screen.getByText("Repair completed")).toBeTruthy();
    expect(screen.getByRole("button", { name: /repair completed/i })).toHaveProperty(
      "disabled",
      true,
    );
    expect(mockedRunRepair).not.toHaveBeenCalled();
  });

  test("does not prepare a repair for a result without a generated answer", () => {
    mockedWorkspace.mockReturnValue(
      workspaceState({
        ...latestRagResult,
        generation: { ...latestRagResult.generation, used_llm: false },
      }),
    );

    render(<HallucinationRepairPanel />);

    expect(screen.getByText("This result has no generated answer.")).toBeTruthy();
    expect(mockedGetRepairSource).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /repair unsupported claims/i }),
    ).toHaveProperty("disabled", true);
  });

  test("surfaces a source preparation error", async () => {
    mockedGetRepairSource.mockRejectedValue(new Error("source failed"));

    render(<HallucinationRepairPanel />);

    expect(
      await screen.findByText("Could not prepare the repair source."),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: /retry preparation/i })).toBeTruthy();
  });
});
