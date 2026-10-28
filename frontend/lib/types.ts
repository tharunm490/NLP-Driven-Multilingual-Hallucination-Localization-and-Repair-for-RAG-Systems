export type DocumentStatus = "processing" | "completed" | "failed";

export type UploadItemStatus = "completed" | "failed" | "rejected";

export interface ComponentStatus {
  status: "ok" | "error";
  detail: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  embedding_model: string;
  embedding_device?: string | null;
  embedding_loaded?: boolean | null;
  max_upload_size_mb?: number;
  max_files_per_request?: number;
  components: Record<string, ComponentStatus>;
}

export interface UploadItemResult {
  document_id: string | null;
  original_filename: string;
  status: UploadItemStatus;
  size_bytes: number;
  content_type: string | null;
  page_count: number | null;
  character_count: number | null;
  token_count: number | null;
  chunk_count: number;
  chunks_stored: number;
  language: string | null;
  duplicate: boolean;
  error: string | null;
  message: string | null;
}

export interface UploadResponse {
  results: UploadItemResult[];
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  size_bytes: number;
  content_type: string | null;
  page_count: number | null;
  character_count: number | null;
  token_count: number | null;
  chunk_count: number;
  chunks_stored: number;
  language: string | null;
  embedding_model: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface DeleteDocumentResponse {
  document_id: string;
  deleted: boolean;
  chunks_deleted: number | null;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  document_ids?: string[];
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  page_number: number | null;
  language: string | null;
  text: string;
  distance: number;
  similarity: number;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  metric: string;
  embedding_model: string;
  results: RetrievedChunk[];
  total: number;
}

export type AnswerStatus =
  | "grounded"
  | "contradicted"
  | "insufficient_evidence"
  | "irrelevant_retrieval"
  | "partially_grounded"
  | "verification_pending";

export type AnswerMode = AnswerStatus;

export type EvidenceState =
  | "sufficient_evidence"
  | "partial_evidence"
  | "insufficient_evidence"
  | "irrelevant_evidence";

export interface RagQueryRequest {
  question: string;
  top_k?: number;
  document_ids?: string[];
}

export interface RagSource {
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  language: string | null;
  text: string;
  retrieval_score: number;
  similarity: number;
  passed_relevance_filter?: boolean;
  relevance_reason?: string | null;
}

export interface RagRejectedChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  page_number: number | null;
  retrieval_score: number;
  similarity: number;
  reason: string;
}

export interface RagRetrievalInfo {
  top_k: number;
  candidate_chunks?: number;
  filtered_chunks?: number;
  returned_chunks: number;
  sufficient_evidence: boolean;
  metric: string;
  score_interpretation: string;
  min_evidence_chunks: number;
  max_distance: number | null;
  distance_filter_applied: boolean;
  rejected_chunks?: RagRejectedChunk[];
}

export interface RagAssessmentInfo {
  state: EvidenceState;
  method: string;
  used_llm: boolean;
  reason: string | null;
  supported_aspects: string[];
  missing_aspects: string[];
  premise_contradicted: boolean;
  fallback_reason: string | null;
}

export type NliLabel = "entailment" | "contradiction" | "neutral";
export type ClaimVerdict = "supported" | "contradicted" | "neutral";
export type ClaimVerificationStatus = ClaimVerdict;
export type ClaimDiagnosticStatus = "unevaluable" | "verification_error";

export interface VerificationNamedEntity {
  text: string;
  label: string;
  start: number;
  end: number;
  score: number | null;
}

export interface VerifiedClaim {
  claim_id: string;
  sentence_index: number;
  sentence: string;
  claim: string;
  claim_start: number;
  claim_end: number;
  hypothesis?: string;
  context?: string | null;
  complete?: boolean;
  evaluable?: boolean;
  incomplete_reason?: string | null;
  entities?: VerificationNamedEntity[];
  extraction_method: string;
  warnings?: string[];
}

export interface NliWindowPrediction {
  window_index: number;
  start: number;
  end: number;
  text: string;
  label: NliLabel;
  confidence: number;
  probabilities: Record<string, number>;
}

export type ComparisonRelevance =
  | "relevant"
  | "weakly_relevant"
  | "irrelevant"
  | "non_substantive"
  | "unknown";
export type EvidenceQuality =
  | "substantive"
  | "table_of_contents"
  | "heading_or_index"
  | "boilerplate"
  | "short"
  | "empty";
export type PremiseStrategy =
  | "full_evidence"
  | "focused_window"
  | "unavailable";

export interface VerifiedEvidence {
  claim_id?: string;
  hypothesis_text?: string;
  evidence: {
    evidence_id: string;
    document_id?: string | null;
    filename: string | null;
    chunk_id?: string | null;
    chunk_index?: number | null;
    page_number: number | null;
    text_preview: string;
  };
  evidence_text?: string;
  rank: number;
  similarity: number;
  retrieval_similarity?: number | null;
  retrieval_score?: number | null;
  evaluated: boolean;
  label: NliLabel | null;
  confidence: number | null;
  probabilities?: Record<string, number>;
  relevance: ComparisonRelevance;
  relevance_score: number;
  claim_term_coverage: number;
  counts_for_verdict: boolean;
  premise_text: string;
  premise_strategy: PremiseStrategy;
  evidence_char_length: number;
  premise_char_length: number;
  truncated: boolean;
  window_start: number | null;
  window_end: number | null;
  evidence_quality?: EvidenceQuality | null;
  evidence_quality_reason?: string | null;
  full_pair_token_count?: number | null;
  input_pair_token_count?: number | null;
  max_pair_tokens?: number | null;
  nli_input_order?: string;
  conflict?: boolean;
  explanation?: string;
  windows?: NliWindowPrediction[];
  error?: string | null;
  warnings?: string[];
}

export interface ClaimVerificationResult {
  claim: VerifiedClaim;
  status: ClaimVerdict | null;
  diagnostic_status?: ClaimDiagnosticStatus | null;
  label: NliLabel | null;
  confidence: number | null;
  probabilities?: Record<string, number>;
  logits?: number[] | null;
  support_confidence?: number | null;
  contradiction_confidence?: number | null;
  conflict?: boolean;
  composite_premise?: string | null;
  explanation?: string;
  pairs_expected: number;
  pairs_evaluated: number;
  pairs_retried?: number;
  pairs_failed?: number;
  complete: boolean;
  incomplete_reason?: string | null;
  matched_evidence: VerifiedEvidence[];
  aggregation?: string;
  warnings?: string[];
  error?: string | null;
}

export interface VerificationSummary {
  claims_total: number;
  claims_evaluable?: number;
  claims_unevaluable?: number;
  claims_checked?: number;
  claims_complete?: number;
  claims_incomplete?: number;
  supported: number;
  contradicted: number;
  neutral: number;
  unevaluable?: number;
  verification_error?: number;
  evidence_passed: number;
  evidence_total?: number;
  pairs_expected?: number;
  pairs_evaluated?: number;
  pairs_missing?: number;
  pairs_retried?: number;
  pairs_failed?: number;
}

export interface VerificationTimings {
  claim_extraction_ms: number;
  evidence_selection_ms: number;
  nli_ms: number;
  aggregation_ms: number;
}

export interface VerificationResult {
  status: "disabled" | "completed" | "partial" | "error";
  enabled: boolean;
  claim_extraction_method: string;
  nli_model: string | null;
  ner_model: string | null;
  claims: ClaimVerificationResult[];
  summary: VerificationSummary;
  timings?: VerificationTimings;
  warnings: string[];
  error: string | null;
  verification_ms: number;
}

export interface RagVerificationInfo {
  claim_level_verification: boolean;
  status: "not_performed" | "pending" | "verified" | "partial" | "error";
  method: string;
  note: string;
  claims_total?: number;
  claims_evaluable?: number;
  claims_unevaluable?: number;
  claims_checked?: number;
  claims_complete?: number;
  claims_incomplete?: number;
  supported?: number;
  contradicted?: number;
  neutral?: number;
  unevaluable?: number;
  verification_error?: number;
  evidence_total?: number;
  pairs_expected?: number;
  pairs_evaluated?: number;
  pairs_missing?: number;
  pairs_retried?: number;
  pairs_failed?: number;
  details?: VerificationResult | null;
}

export interface RagDiagnostics {
  question_preview: string | null;
  candidates_retrieved: number;
  candidates_after_filter: number;
  rejected_chunk_ids: string[];
  assessment_method: string;
  assessment_fallback_reason: string | null;
  generation_skipped: boolean;
  generation_skip_reason: string | null;
  verification_error?: string | null;
  errors: string[];
}

export interface RagGenerationInfo {
  used_llm: boolean;
  provider: string | null;
  model: string | null;
  prompt_version: string;
  finish_reason: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

export interface RagTimings {
  retrieval_ms: number;
  assessment_ms?: number;
  generation_ms: number;
  verification_ms?: number;
  total_ms: number;
}

export interface RagQueryResponse {
  result_id?: string | null;
  question: string;
  normalized_question: string;
  detected_language: string | null;
  language_confidence: number | null;
  language_reliable: boolean;
  answer: string;
  answer_mode: AnswerMode;
  answer_status?: AnswerStatus;
  evidence_state?: EvidenceState;
  sources: RagSource[];
  retrieval: RagRetrievalInfo;
  assessment?: RagAssessmentInfo;
  verification?: RagVerificationInfo;
  diagnostics?: RagDiagnostics;
  generation: RagGenerationInfo;
  timings: RagTimings;
}

export type HallucinationVerdict =
  | "supported"
  | "potentially_hallucinated"
  | "needs_review";

export interface HallucinationCheckRequest {
  result_id: string;
}

export interface HallucinationEvidenceResult {
  evidence_id: string;
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  retrieval_score: number;
  retrieval_similarity: number;
  support_probability: number;
  raw_model_label: string;
}

export interface HallucinationClaimResult {
  claim_id: string;
  sentence_index: number;
  sentence: string;
  claim: string;
  claim_start: number;
  claim_end: number;
  verdict: HallucinationVerdict;
  raw_model_label: string;
  support_probability: number;
  explanation: string;
  evidence: HallucinationEvidenceResult[];
  nli_status: ClaimVerdict | null;
  nli_confidence: number | null;
  nli_disagrees: boolean;
}

export interface HallucinationSummary {
  claims_total: number;
  supported: number;
  potentially_hallucinated: number;
  needs_review: number;
  nli_disagreements: number;
  evidence_total: number;
  pairs_evaluated: number;
}

export interface HallucinationCheckResponse {
  check_id: string;
  result_id: string;
  question: string;
  model_name: string;
  model_input_order: "source_then_claim";
  support_threshold: number;
  review_margin: number;
  claims: HallucinationClaimResult[];
  summary: HallucinationSummary;
  timings: {
    claim_extraction_ms: number;
    inference_ms: number;
    total_ms: number;
  };
  warnings: string[];
}
