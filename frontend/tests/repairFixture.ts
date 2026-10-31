import type {
  HallucinationCheckResponse,
  RepairClaimResult,
  RepairResult,
  RepairSourceResponse,
} from "../lib/types";

import { latestRagResult } from "./ragResultFixture";

const originalAnswer = latestRagResult.answer;
const finalAnswer = "Acme was founded in 2016.";

const evidence = {
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
};

export const repairCheck: HallucinationCheckResponse = {
  check_id: "check-1",
  result_id: "result-1",
  question: latestRagResult.question,
  model_name: "Horizon-Labs/hallucination-guard-base",
  model_input_order: "source_then_claim",
  support_threshold: 0.5,
  review_margin: 0.05,
  claims: [
    {
      claim_id: "claim-1",
      sentence_index: 0,
      sentence: originalAnswer,
      claim: originalAnswer,
      claim_start: 0,
      claim_end: originalAnswer.length,
      verdict: "potentially_hallucinated",
      raw_model_label: "UNSUPPORTED",
      support_probability: 0.12,
      explanation: "The supplied evidence did not support the claim.",
      evidence: [evidence],
      nli_status: "neutral",
      nli_confidence: 0.62,
      nli_disagrees: false,
    },
  ],
  summary: {
    claims_total: 1,
    supported: 0,
    potentially_hallucinated: 1,
    nli_disagreements: 0,
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

const verifiedClaim: RepairClaimResult = {
  claim_id: "claim-1",
  requires_repair: true,
  original_claim: originalAnswer,
  original_sentence: originalAnswer,
  original_context: null,
  original_claim_start: 0,
  original_claim_end: originalAnswer.length,
  original_verdict: "potentially_hallucinated",
  original_support_probability: 0.12,
  original_raw_model_label: "UNSUPPORTED",
  original_nli_status: "neutral",
  original_nli_confidence: 0.62,
  original_nli_disagrees: false,
  status: "VERIFIED",
  attempts: [
    {
      attempt_number: 1,
      status: "VERIFIED",
      decision: "REPAIRED",
      repaired_claim: finalAnswer,
      reason: "The evidence states the corrected founding year.",
      supporting_evidence: [
        {
          source: "iso-27001.txt",
          page: 2,
          evidence_quote: "Acme was founded in 2016 in Paris.",
        },
      ],
      verification: {
        nli_status: "supported",
        nli_confidence: 0.94,
        nli_explanation: "The NLI verifier supported the repaired claim.",
        guard_verdict: "supported",
        guard_support_probability: 0.93,
        guard_raw_label: "SUPPORTED",
        both_supported: true,
      },
      failure_reason: null,
      error: null,
      duration_ms: 220,
    },
  ],
  selected_evidence: [
    {
      evidence_id: "evidence-1",
      document_id: "doc-1",
      filename: "iso-27001.txt",
      chunk_id: "chunk-1",
      chunk_index: 4,
      page_number: 2,
      text: "Acme was founded in 2016 in Paris.",
      retrieval_score: 0.12,
      retrieval_similarity: 0.88,
      guard_support_probability: 0.12,
      guard_raw_label: "UNSUPPORTED",
    },
  ],
  final_claim: finalAnswer,
  final_decision: "REPAIRED",
  included_in_final: true,
};

export const repairResult: RepairResult = {
  repair_id: "repair-1",
  result_id: "result-1",
  question: latestRagResult.question,
  original_answer: originalAnswer,
  original_verification: latestRagResult.verification!,
  hallucination_check: repairCheck,
  overview: {
    claims_total: 1,
    supported: 0,
    potentially_hallucinated: 1,
    neutral: 0,
    nli_disagreements: 0,
    claims_unevaluable: 0,
    invalid_fragments_skipped: 0,
    evidence_total: 1,
  },
  claims: [verifiedClaim],
  final_answer: finalAnswer,
  final_claims: [
    {
      claim_id: "claim-1",
      text: finalAnswer,
      start: 0,
      end: finalAnswer.length,
      verdict: "verified",
    },
  ],
  omissions: [],
  iterations: [
    {
      iteration_number: 1,
      submitted_claim_ids: ["claim-1"],
      verified_claim_ids: ["claim-1"],
      remaining_claim_ids: [],
      status: "COMPLETED",
    },
  ],
  status: "FULLY_VERIFIED",
  max_attempts: 3,
  model_provider: "groq",
  model_name: "openai/gpt-oss-120b",
  created_at: "2026-10-31T10:00:00Z",
  timings: {
    preparation_ms: 12,
    repair_ms: 230,
    total_ms: 242,
  },
  warnings: [],
};

export const repairSource: RepairSourceResponse = {
  result_id: "result-1",
  original: latestRagResult,
  hallucination: repairCheck,
  overview: repairResult.overview,
  repair: null,
  max_attempts: 3,
};

export const repairSourceWithResult: RepairSourceResponse = {
  ...repairSource,
  repair: repairResult,
};
