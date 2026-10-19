"""Pydantic schemas for the RAG QA endpoint (Phase 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AnswerMode = Literal["grounded", "insufficient_evidence"]


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


class RagRetrievalInfo(BaseModel):
    top_k: int
    returned_chunks: int
    sufficient_evidence: bool
    metric: str = "cosine_distance"
    score_interpretation: str = "lower distance = more similar; not a probability"
    min_evidence_chunks: int
    max_distance: float | None = None
    distance_filter_applied: bool = False


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
    generation_ms: float
    total_ms: float


class RagQueryResponse(BaseModel):
    question: str
    normalized_question: str
    detected_language: str | None = None
    language_confidence: float | None = None
    language_reliable: bool = False
    answer: str
    answer_mode: AnswerMode
    sources: list[RagSource] = Field(default_factory=list)
    retrieval: RagRetrievalInfo
    generation: RagGenerationInfo
    timings: RagTimings
