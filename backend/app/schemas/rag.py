"""Pydantic schemas for the RAG QA endpoint (Phase 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.verification import VerificationResult

AnswerStatus = Literal[
    "grounded",
    "contradicted",
    "insufficient_evidence",
    "irrelevant_retrieval",
    "partially_grounded",
    "verification_pending",
]

# Backward-compatible field name used by Phase 3 clients. It carries the same
# explicit status as ``answer_status``.
AnswerMode = AnswerStatus

EvidenceState = Literal[
    "sufficient_evidence",
    "partial_evidence",
    "insufficient_evidence",
    "irrelevant_evidence",
]

VerificationStatus = Literal[
    "not_performed", "pending", "verified", "partial", "error"
]


class RagQueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=8000,
        description="The user's question, in any supported language.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of chunks to retrieve. Defaults to TOP_K from settings.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document IDs to restrict retrieval to.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class RagSource(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    language: str | None = None
    text: str
    retrieval_score: float = Field(
        description=(
            "ChromaDB cosine distance for this chunk. Lower is more similar. "
            "It is NOT a probability, confidence, or proof that the chunk is correct. "
            "Equals the 'distance' field returned by /api/retrieval/search."
        )
    )
    similarity: float = Field(
        description="1 - retrieval_score (cosine similarity). Not a confidence value."
    )
    passed_relevance_filter: bool = True
    relevance_reason: str | None = None


class RagRejectedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    retrieval_score: float
    similarity: float
    reason: str = Field(
        description="Why the chunk was not passed as evidence (e.g. distance_above_threshold)."
    )


class RagRetrievalInfo(BaseModel):
    top_k: int
    candidate_chunks: int = 0
    filtered_chunks: int = 0
    returned_chunks: int
    sufficient_evidence: bool
    metric: str = "cosine_distance"
    score_interpretation: str = (
        "lower cosine distance = more similar; not a probability or confidence"
    )
    min_evidence_chunks: int
    max_distance: float | None = None
    distance_filter_applied: bool = False
    rejected_chunks: list[RagRejectedChunk] = Field(default_factory=list)


class RagAssessmentInfo(BaseModel):
    state: EvidenceState
    method: str = Field(
        description="How the sufficiency decision was made: llm, lexical, retrieval_only, or skipped."
    )
    used_llm: bool = False
    reason: str | None = None
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    premise_contradicted: bool = False
    fallback_reason: str | None = None


class RagVerificationInfo(BaseModel):
    claim_level_verification: bool = False
    status: VerificationStatus = "not_performed"
    method: str = "none"
    note: str = (
        "Claim-level verification was not performed for this answer."
    )
    claims_total: int = 0
    claims_evaluable: int = 0
    claims_unevaluable: int = 0
    claims_checked: int = 0
    claims_complete: int = 0
    claims_incomplete: int = 0
    supported: int = 0
    contradicted: int = 0
    neutral: int = 0
    unevaluable: int = 0
    verification_error: int = 0
    evidence_total: int = 0
    pairs_expected: int = 0
    pairs_evaluated: int = 0
    pairs_missing: int = 0
    details: VerificationResult | None = None


class RagDiagnostics(BaseModel):
    question_preview: str | None = Field(
        default=None,
        description="Short, safely-truncated copy of the question for debugging.",
    )
    candidates_retrieved: int = 0
    candidates_after_filter: int = 0
    rejected_chunk_ids: list[str] = Field(default_factory=list)
    assessment_method: str = "skipped"
    assessment_fallback_reason: str | None = None
    generation_skipped: bool = False
    generation_skip_reason: str | None = None
    verification_error: str | None = None
    errors: list[str] = Field(default_factory=list)


class RagGenerationInfo(BaseModel):
    used_llm: bool
    provider: str | None = None
    model: str | None = None
    prompt_version: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class RagTimings(BaseModel):
    retrieval_ms: float
    assessment_ms: float = 0.0
    generation_ms: float
    verification_ms: float = 0.0
    total_ms: float


class RagQueryResponse(BaseModel):
    # Opaque identifier used by /api/rag/results/{result_id}; the full answer
    # stays server-side so document text is not persisted in browser storage.
    result_id: str | None = None
    question: str
    normalized_question: str
    detected_language: str | None = None
    language_confidence: float | None = None
    language_reliable: bool = False
    answer: str
    answer_mode: AnswerMode
    answer_status: AnswerStatus
    evidence_state: EvidenceState
    sources: list[RagSource] = Field(default_factory=list)
    retrieval: RagRetrievalInfo
    assessment: RagAssessmentInfo
    verification: RagVerificationInfo
    generation: RagGenerationInfo
    diagnostics: RagDiagnostics
    timings: RagTimings
