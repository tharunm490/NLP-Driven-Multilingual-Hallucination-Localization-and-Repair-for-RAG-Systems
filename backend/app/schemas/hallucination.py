"""Schemas for the dedicated hallucination groundedness check."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HallucinationVerdict = Literal[
    "supported",
    "potentially_hallucinated",
]


class HallucinationCheckRequest(BaseModel):
    result_id: str = Field(
        min_length=1,
        max_length=128,
        description="Result identifier returned by POST /api/rag/query.",
    )


class HallucinationEvidenceResult(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    retrieval_score: float
    retrieval_similarity: float
    support_probability: float = Field(ge=0.0, le=1.0)
    raw_model_label: str


class HallucinationClaimResult(BaseModel):
    claim_id: str
    sentence_index: int
    sentence: str
    claim: str
    claim_start: int = Field(ge=0)
    claim_end: int = Field(ge=0)
    verdict: HallucinationVerdict
    raw_model_label: str
    support_probability: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: list[HallucinationEvidenceResult] = Field(default_factory=list)
    nli_status: Literal["supported", "contradicted", "neutral"] | None = None
    nli_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    nli_disagrees: bool = False


class HallucinationSummary(BaseModel):
    claims_total: int = 0
    supported: int = 0
    potentially_hallucinated: int = 0
    nli_disagreements: int = 0
    evidence_total: int = 0
    pairs_evaluated: int = 0


class HallucinationTimings(BaseModel):
    claim_extraction_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


class HallucinationCheckResponse(BaseModel):
    check_id: str
    result_id: str
    question: str
    model_name: str
    model_input_order: Literal["source_then_claim"] = "source_then_claim"
    support_threshold: float
    review_margin: float
    claims: list[HallucinationClaimResult] = Field(default_factory=list)
    summary: HallucinationSummary
    timings: HallucinationTimings
    warnings: list[str] = Field(default_factory=list)
