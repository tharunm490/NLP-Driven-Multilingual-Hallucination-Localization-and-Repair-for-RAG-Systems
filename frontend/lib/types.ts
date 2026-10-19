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

export type AnswerMode = "grounded" | "insufficient_evidence";

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
}

export interface RagRetrievalInfo {
  top_k: number;
  returned_chunks: number;
  sufficient_evidence: boolean;
  metric: string;
  score_interpretation: string;
  min_evidence_chunks: number;
  max_distance: number | null;
  distance_filter_applied: boolean;
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
  generation_ms: number;
  total_ms: number;
}

export interface RagQueryResponse {
  question: string;
  normalized_question: string;
  detected_language: string | null;
  language_confidence: number | null;
  language_reliable: boolean;
  answer: string;
  answer_mode: AnswerMode;
  sources: RagSource[];
  retrieval: RagRetrievalInfo;
  generation: RagGenerationInfo;
  timings: RagTimings;
}
