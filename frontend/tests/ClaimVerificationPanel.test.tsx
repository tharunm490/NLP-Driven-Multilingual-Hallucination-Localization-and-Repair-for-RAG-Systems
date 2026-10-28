import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";

import ClaimVerificationPanel from "../components/ClaimVerificationPanel";
import type {
  ClaimVerificationResult,
  RagVerificationInfo,
  VerificationResult,
  VerifiedEvidence,
} from "../lib/types";

function evidence(
  id: string,
  rank: number,
  overrides: Partial<VerifiedEvidence> = {},
): VerifiedEvidence {
  return {
    evidence: {
      evidence_id: id,
      filename: "iso-27001.txt",
      page_number: 1,
      text_preview: `Preview for ${id}`,
    },
    rank,
    similarity: 0.5,
    retrieval_similarity: 0.8,
    retrieval_score: 0.2,
    evaluated: true,
    label: "entailment",
    confidence: 0.9,
    relevance: "relevant",
    relevance_score: 0.8,
    claim_term_coverage: 0.8,
    counts_for_verdict: true,
    premise_text: `Premise for ${id}`,
    premise_strategy: "full_evidence",
    evidence_char_length: 40,
    premise_char_length: 40,
    truncated: false,
    window_start: 0,
    window_end: 40,
    explanation: `Explanation for ${id}`,
    ...overrides,
  };
}

function claim(
  id: string,
  overrides: Partial<ClaimVerificationResult> = {},
): ClaimVerificationResult {
  return {
    claim: {
      claim_id: id,
      sentence_index: 1,
      sentence: `Claim ${id}.`,
      claim: `Claim ${id}.`,
      claim_start: 0,
      claim_end: 9,
      extraction_method: "heuristic",
    },
    status: "supported",
    label: "entailment",
    confidence: 0.9,
    pairs_expected: 3,
    pairs_evaluated: 3,
    complete: true,
    matched_evidence: [],
    ...overrides,
  };
}

const verification: RagVerificationInfo = {
  claim_level_verification: true,
  status: "partial",
  method: "local_nli",
  note: "Every claim was compared against every retrieved chunk that fit the budget.",
  claims_total: 2,
  claims_checked: 2,
  claims_complete: 1,
  claims_incomplete: 1,
  supported: 1,
  contradicted: 0,
  neutral: 1,
  verification_error: 0,
  evidence_total: 3,
  pairs_expected: 6,
  pairs_evaluated: 5,
  pairs_missing: 1,
  details: {
    status: "partial",
    enabled: true,
    claim_extraction_method: "heuristic",
    nli_model: "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    ner_model: null,
    claims: [
      claim("c1", {
        matched_evidence: [
          evidence("e1", 1, {
            retrieval_similarity: 0.91,
            retrieval_score: 0.09,
            similarity: 0.88,
          }),
          evidence("e2", 2, {
            retrieval_similarity: 0.72,
            retrieval_score: 0.28,
            similarity: 0.55,
          }),
          evidence("e3", 3, {
            retrieval_similarity: null,
            retrieval_score: null,
            similarity: 0.11,
            label: "neutral",
            evaluated: true,
          }),
        ],
      }),
      claim("c2", {
        status: "neutral",
        label: "neutral",
        confidence: 0.7,
        pairs_evaluated: 2,
        complete: false,
        incomplete_reason: "The evidence budget was exhausted.",
        matched_evidence: [
          evidence("e4", 1, {
            retrieval_similarity: 0.66,
            retrieval_score: 0.34,
            similarity: 0.44,
            label: "neutral",
          }),
          evidence("e5", 2, {
            retrieval_similarity: 0.61,
            retrieval_score: 0.39,
            similarity: 0.4,
            label: "neutral",
          }),
          evidence("e6", 3, {
            retrieval_similarity: 0.58,
            retrieval_score: 0.42,
            similarity: 0.3,
            evaluated: false,
            label: null,
            confidence: null,
            counts_for_verdict: false,
            premise_text: "",
            premise_strategy: "unavailable",
            explanation: "Not compared: budget reached.",
          }),
        ],
      }),
    ],
    summary: {
      claims_total: 2,
      claims_checked: 2,
      claims_complete: 1,
      claims_incomplete: 1,
      supported: 1,
      contradicted: 0,
      neutral: 1,
      verification_error: 0,
      evidence_passed: 3,
      evidence_total: 3,
      pairs_expected: 6,
      pairs_evaluated: 5,
      pairs_missing: 1,
    },
    warnings: [],
    error: null,
    verification_ms: 12,
  } as VerificationResult,
};

describe("ClaimVerificationPanel", () => {
  test("renders every claim and every expected evidence pair", () => {
    const html = renderToStaticMarkup(
      <ClaimVerificationPanel verification={verification} />,
    );

    expect(html.match(/data-testid="verification-pair-row"/g)).toHaveLength(6);
    for (const id of ["e1", "e2", "e3", "e4", "e5", "e6"]) {
      expect(html).toContain(`data-evidence-id="${id}"`);
    }
    for (let index = 1; index <= 5; index += 1) {
      expect(html).toContain(`Premise for e${index}`);
    }
    expect(html).toContain("Preview for e6");
  });

  test("shows original retrieval similarity separately from matcher similarity", () => {
    const html = renderToStaticMarkup(
      <ClaimVerificationPanel verification={verification} />,
    );

    expect(html).toContain("Retrieval similarity");
    expect(html).toContain("Matcher similarity");
    expect(html).toContain("0.9100");
    expect(html).toContain("distance 0.0900");
    expect(html).toContain("0.8800");
    expect(html).toContain("0.7200");
    expect(html).toContain("0.5500");
    expect(html).toContain("n/a");
    expect(html).toContain("0.1100");
  });

  test("exposes full pair inputs, token accounting, quality, and model scores", () => {
    const withDetails = structuredClone(verification);
    const details = withDetails.details?.claims[0].matched_evidence[0];
    if (!details) {
      throw new Error("missing test pair");
    }
    details.claim_id = "c1";
    details.hypothesis_text = "Exact hypothesis for c1";
    details.evidence_text = "Full evidence text for e1";
    details.evidence_quality = "substantive";
    details.evidence_quality_reason = "Complete prose sentence.";
    details.full_pair_token_count = 120;
    details.input_pair_token_count = 96;
    details.max_pair_tokens = 256;
    details.nli_input_order = "premise_evidence,hypothesis_claim";
    details.probabilities = {
      entailment: 0.91,
      contradiction: 0.02,
      neutral: 0.07,
    };

    const html = renderToStaticMarkup(
      <ClaimVerificationPanel verification={withDetails} />,
    );

    expect(html).toContain("Exact hypothesis for c1");
    expect(html).toContain("Full evidence text for e1");
    expect(html).toContain("quality: substantive");
    expect(html).toContain("Complete prose sentence.");
    expect(html).toContain("input 96 / full 120 / max 256");
    expect(html).toContain("entailment 0.9100");
    expect(html).toContain("premise_evidence,hypothesis_claim");
  });

  test("renders a missing verdict as neutral without exclusion copy", () => {
    const withUnevaluable = structuredClone(verification);
    withUnevaluable.claims_unevaluable = 1;
    withUnevaluable.unevaluable = 1;
    const original = withUnevaluable.details?.claims[1];
    if (!original || !withUnevaluable.details) {
      throw new Error("missing test claim");
    }
    withUnevaluable.details.claims[1] = {
      ...original,
      status: null,
      diagnostic_status: "unevaluable",
      label: null,
      confidence: null,
      pairs_expected: 0,
      pairs_evaluated: 0,
      complete: false,
      incomplete_reason: "The claim starts with a pronoun.",
      matched_evidence: [],
      claim: {
        ...original.claim,
        evaluable: false,
        complete: false,
        incomplete_reason: "The claim starts with a pronoun.",
      },
    };

    const html = renderToStaticMarkup(
      <ClaimVerificationPanel verification={withUnevaluable} />,
    );

    expect(html).toContain("Neutral");
    expect(html).not.toContain("Not evaluated");
    expect(html).not.toContain("excluded from NLI evaluation");
  });

  test("renders coverage counts and aggregate claim statuses", () => {
    const html = renderToStaticMarkup(
      <ClaimVerificationPanel verification={verification} />,
    );

    expect(html).toContain("1/2 fully evaluated");
    expect(html).toContain("5/6 evidence pairs");
    expect(html).toContain("1 supported");
    expect(html).toContain("0 contradicted");
    expect(html).toContain("1 neutral");
    expect(html).toContain("1 pairs missing");
    expect(html).toContain("Supported");
    expect(html).toContain("Neutral");
    expect(html).toContain("1 evidence comparison(s) were unavailable.");
    expect(html).toContain(
      "1 claim(s) have partial evidence-pair coverage.",
    );
  });
});
