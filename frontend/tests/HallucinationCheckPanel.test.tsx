import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";

import {
  buildHighlightSegments,
  buildVerdictLegend,
  HallucinationResultView,
} from "../components/HallucinationCheckPanel";
import {
  hallucinationVerdictLabel,
  hallucinationVerdictTone,
} from "../lib/format";
import type {
  HallucinationCheckResponse,
  HallucinationClaimResult,
} from "../lib/types";

const ANSWER =
  "Acme was founded in 2015. Beta was founded in 2020. Gamma has 12 staff.";
const CLAIMS = [
  "Acme was founded in 2015.",
  "Beta was founded in 2020.",
  "Gamma has 12 staff.",
];

function claim(
  index: number,
  verdict: HallucinationClaimResult["verdict"],
  probability: number,
  overrides: Partial<HallucinationClaimResult> = {},
): HallucinationClaimResult {
  const text = CLAIMS[index];
  const start = ANSWER.indexOf(text);
  return {
    claim_id: `c${index + 1}`,
    sentence_index: index,
    sentence: text,
    claim: text,
    claim_start: start,
    claim_end: start + text.length,
    verdict,
    raw_model_label: verdict === "supported" ? "SUPPORTED" : "UNSUPPORTED",
    support_probability: probability,
    explanation: `Evidence support for ${text}`,
    evidence: [
      {
        evidence_id: `e${index + 1}`,
        document_id: "doc-1",
        filename: "iso-27001.txt",
        chunk_id: `e${index + 1}`,
        chunk_index: index,
        page_number: index + 3,
        retrieval_score: 0.12,
        retrieval_similarity: 0.88,
        support_probability: probability,
        raw_model_label:
          verdict === "supported" ? "SUPPORTED" : "UNSUPPORTED",
      },
    ],
    nli_status: "supported",
    nli_confidence: 0.9,
    nli_disagrees: index === 1,
    ...overrides,
  };
}

const check: HallucinationCheckResponse = {
  check_id: "check-1",
  result_id: "result-1",
  question: "When were the companies founded?",
  model_name: "Horizon-Labs/hallucination-guard-base",
  model_input_order: "source_then_claim",
  support_threshold: 0.5,
  review_margin: 0.05,
  claims: [
    claim(0, "supported", 0.93),
    claim(1, "potentially_hallucinated", 0.12),
    claim(2, "supported", 0.51),
  ],
  summary: {
    claims_total: 3,
    supported: 2,
    potentially_hallucinated: 1,
    nli_disagreements: 1,
    evidence_total: 3,
    pairs_evaluated: 9,
  },
  timings: {
    claim_extraction_ms: 1.2,
    inference_ms: 42.5,
    total_ms: 45.1,
  },
  warnings: ["Example warning"],
};

describe("hallucination verdict presentation", () => {
  test("maps exactly two labels and tones without using contradiction", () => {
    expect(hallucinationVerdictLabel("supported")).toBe("Supported");
    expect(hallucinationVerdictLabel("potentially_hallucinated")).toBe(
      "Potentially hallucinated",
    );
    expect(hallucinationVerdictTone("supported")).toBe("success");
    expect(hallucinationVerdictTone("potentially_hallucinated")).toBe("error");
    expect(hallucinationVerdictLabel("potentially_hallucinated")).not.toContain(
      "contradict",
    );
  });
});

describe("answer highlighting", () => {
  test("uses original answer offsets without inventing word spans", () => {
    const segments = buildHighlightSegments(ANSWER, check.claims);
    const highlighted = segments.filter((segment) => segment.claim);

    expect(highlighted).toHaveLength(3);
    expect(highlighted.map((segment) => segment.text)).toEqual(CLAIMS);
    expect(segments.map((segment) => segment.text).join("")).toBe(ANSWER);
  });

  test("ignores invalid or overlapping spans", () => {
    const first = claim(0, "supported", 0.9);
    const overlapping = claim(1, "potentially_hallucinated", 0.5, {
      claim_start: 2,
      claim_end: 10,
    });
    const segments = buildHighlightSegments(ANSWER, [first, overlapping]);
    expect(segments.map((segment) => segment.text).join("")).toBe(ANSWER);
    expect(segments.filter((segment) => segment.claim)).toHaveLength(1);
  });
});

describe("HallucinationResultView", () => {
  test("shows summary, evidence, raw scores, and NLI disagreement", () => {
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={check} />,
    );

    expect(html).toContain("Horizon-Labs/hallucination-guard-base");
    expect(html).toContain("Potentially hallucinated");
    expect(html).toContain("NLI disagreements");
    expect(html).toContain("12.0%");
    expect(html).toContain("SUPPORTED");
    expect(html).not.toMatch(
      /hallucination-support-label">\s*UNSUPPORTED/,
    );
    expect(html).toContain("iso-27001.txt");
    expect(html).toContain("p.4");
    expect(html).toContain("The dedicated model is binary");
    expect(html).not.toContain("Needs review");
    expect(html).not.toContain("Contradicted by evidence");
  });

  test("renders no-claim results without a fake verdict", () => {
    const empty = {
      ...check,
      claims: [],
      summary: {
        ...check.summary,
        claims_total: 0,
        supported: 0,
        potentially_hallucinated: 0,
        nli_disagreements: 0,
      },
    };
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={empty} />,
    );
    expect(html).toContain("No factual claims were extracted");
  });
});

describe("support score presentation and two-class verdicts", () => {
  const supportCheck: HallucinationCheckResponse = {
    ...check,
    claims: [
      claim(0, "supported", 0.939),
      claim(1, "potentially_hallucinated", 0.159),
      claim(2, "supported", 0.51),
    ],
  };

  test("labels every Support column row SUPPORTED without changing scores", () => {
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={supportCheck} />,
    );

    const supportLabels = html.match(
      /hallucination-support-label">\s*SUPPORTED\s*<\/div>/g,
    );
    expect(supportLabels).toHaveLength(3);
    expect(html).not.toMatch(
      /hallucination-support-label">\s*UNSUPPORTED/,
    );
    expect(html).toContain("93.9%");
    expect(html).toContain("15.9%");
    expect(html).not.toContain("84.1%");
    expect(html).not.toContain("6.1%");
  });

  test("keeps the Potentially hallucinated verdict for UNSUPPORTED claims", () => {
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={supportCheck} />,
    );

    expect(html).toContain("Potentially hallucinated");
    expect(html).toContain("15.9%");
  });

  test("keeps the Supported verdict for SUPPORTED claims", () => {
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={supportCheck} />,
    );

    expect(html).toContain(">Supported<");
    expect(html).toContain("93.9%");
  });

  test("keeps the independent three-class NLI predictions", () => {
    const nliCheck: HallucinationCheckResponse = {
      ...check,
      claims: [
        claim(0, "supported", 0.93, { nli_status: "supported" }),
        claim(1, "potentially_hallucinated", 0.12, {
          nli_status: "contradicted",
        }),
        claim(2, "supported", 0.51, { nli_status: "neutral" }),
      ],
    };
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={nliCheck} />,
    );

    expect(html).toContain(">Supported<");
    expect(html).toContain(">Contradicted<");
    expect(html).toContain(">Neutral<");
    expect(html).toContain("Disagrees");
  });

  test("renders only the two model-class categories in the legend", () => {
    expect(buildVerdictLegend()).toEqual([
      {
        verdict: "supported",
        label: "Supported",
        modelLabel: "SUPPORTED",
      },
      {
        verdict: "potentially_hallucinated",
        label: "Potentially hallucinated",
        modelLabel: "UNSUPPORTED",
      },
    ]);

    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={supportCheck} />,
    );
    expect(html).toContain("Hallucination Guard");
    expect(html).toContain("SUPPORTED");
    expect(html).toContain("UNSUPPORTED");
    expect(html).not.toContain("Needs review");
    expect(html).not.toContain("NEEDS_REVIEW");
  });

  test("does not render a review verdict or review summary card", () => {
    const html = renderToStaticMarkup(
      <HallucinationResultView answer={ANSWER} check={supportCheck} />,
    );

    expect(html).not.toContain("Needs review");
    expect(html).not.toContain("needs_review");
    expect(html).toContain("NLI disagreements");
  });
});
