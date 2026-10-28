import { describe, expect, it } from "vitest";

import { answerStatusLabel, answerStatusShowsAbstention, answerStatusTone } from "../lib/format";

describe("answer status helpers", () => {
  it("does not label an evidence-supported generated answer as insufficient", () => {
    expect(answerStatusLabel("verification_pending")).toBe("Verification pending");
    expect(answerStatusShowsAbstention("verification_pending")).toBe(false);
    expect(answerStatusTone("verification_pending")).not.toBe("warning");
  });

  it("distinguishes partial and irrelevant outcomes", () => {
    expect(answerStatusLabel("partially_grounded")).toBe("Partially supported");
    expect(answerStatusShowsAbstention("partially_grounded")).toBe(false);
    expect(answerStatusLabel("irrelevant_retrieval")).toBe("No relevant evidence found");
    expect(answerStatusShowsAbstention("irrelevant_retrieval")).toBe(true);
  });

  it("shows abstention only for genuinely unanswerable outcomes", () => {
    expect(answerStatusLabel("insufficient_evidence")).toBe("Insufficient evidence");
    expect(answerStatusShowsAbstention("insufficient_evidence")).toBe(true);
    expect(answerStatusShowsAbstention("grounded")).toBe(false);
  });

  it("reports contradicted answers distinctly", () => {
    expect(answerStatusLabel("contradicted")).toBe("Contradicted by evidence");
    expect(answerStatusTone("contradicted")).toBe("error");
    expect(answerStatusShowsAbstention("contradicted")).toBe(false);
  });
});
