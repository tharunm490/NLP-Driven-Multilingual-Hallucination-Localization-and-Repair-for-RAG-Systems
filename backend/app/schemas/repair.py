"""Schemas for evidence-grounded, iterative hallucination repair.

The repair layer never changes the original Ask result, NLI verdicts, or
Hallucination Guard verdicts. Every model-proposed correction is re-verified by
both existing systems before it can be included as a factual statement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.hallucination import (
    HallucinationCheckResponse,
    HallucinationVerdict,
)
from app.schemas.rag import RagQueryResponse, RagVerificationInfo
from app.schemas.verification import ClaimVerdict

RepairDecision = Literal[
    "REPAIRED",
    "QUALIFIED",
    "REMOVED",
    "INSUFFICIENT_EVIDENCE",
]

RepairClaimStatus = Literal[
    "ORIGINAL",
    "REPAIRING",
    "REPAIRED_PENDING_VERIFICATION",
    "VERIFIED",
    "RETRY_REQUIRED",
    "UNRESOLVED",
    "REMOVED",
    "INSUFFICIENT_EVIDENCE",
    "REPAIR_ERROR",
]

RepairOverallStatus = Literal[
    "FULLY_VERIFIED",
    "PARTIALLY_VERIFIED",
    "REPAIR_INCOMPLETE",
    "NO_REPAIR_NEEDED",
]

FinalClaimVerdict = Literal["verified", "unresolved", "insufficient_evidence"]


class RepairEvidence(BaseModel):
    """One claim-specific evidence passage reused from the original verification."""

    evidence_id: str
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    text: str
    retrieval_score: float
    retrieval_similarity: float
    guard_support_probability: float = Field(ge=0.0, le=1.0)
    guard_raw_label: str


class RepairSupportingEvidence(BaseModel):
    """Evidence reference returned by the repair model, validated server-side."""

    source: str
    page: int | None = None
    evidence_quote: str


class RepairLLMOutput(BaseModel):
    """Strict schema for the repair model's JSON response."""

    status: RepairDecision
    original_claim: str
    repaired_claim: str | None = None
    reason: str = ""
    supporting_evidence: list[RepairSupportingEvidence] = Field(
        default_factory=list
    )

    @field_validator("status", mode="before")
    @classmethod
    def status_must_be_a_decision(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value


class RepairVerification(BaseModel):
    nli_status: ClaimVerdict | None = None
    nli_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    nli_explanation: str | None = None
    guard_verdict: HallucinationVerdict | None = None
    guard_support_probability: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    guard_raw_label: str | None = None
    both_supported: bool = False


class RepairAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    status: RepairClaimStatus
    decision: RepairDecision | None = None
    repaired_claim: str | None = None
    reason: str | None = None
    supporting_evidence: list[RepairSupportingEvidence] = Field(
        default_factory=list
    )
    verification: RepairVerification | None = None
    failure_reason: str | None = None
    error: str | None = None
    duration_ms: float = 0.0


class RepairIteration(BaseModel):
    iteration_number: int = Field(ge=1, le=3)
    submitted_claim_ids: list[str] = Field(default_factory=list)
    verified_claim_ids: list[str] = Field(default_factory=list)
    remaining_claim_ids: list[str] = Field(default_factory=list)
    status: Literal["COMPLETED", "PARTIALLY_COMPLETED", "FAILED"] = "COMPLETED"


class RepairClaimResult(BaseModel):
    claim_id: str
    requires_repair: bool
    original_claim: str
    original_sentence: str
    original_context: str | None = None
    original_claim_start: int = Field(ge=0)
    original_claim_end: int = Field(ge=0)
    original_verdict: HallucinationVerdict
    original_support_probability: float = Field(ge=0.0, le=1.0)
    original_raw_model_label: str
    original_nli_status: ClaimVerdict | None = None
    original_nli_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    original_nli_disagrees: bool = False
    status: RepairClaimStatus
    attempts: list[RepairAttempt] = Field(default_factory=list)
    selected_evidence: list[RepairEvidence] = Field(default_factory=list)
    final_claim: str | None = None
    final_decision: RepairDecision | None = None
    included_in_final: bool = False


class FinalAnswerClaim(BaseModel):
    claim_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    verdict: FinalClaimVerdict


class RepairOmission(BaseModel):
    claim_id: str
    original_claim: str
    reason: str


class RepairOverview(BaseModel):
    claims_total: int = 0
    supported: int = 0
    potentially_hallucinated: int = 0
    neutral: int = 0
    nli_disagreements: int = 0
    claims_unevaluable: int = 0
    invalid_fragments_skipped: int = 0
    evidence_total: int = 0


class RepairTimings(BaseModel):
    preparation_ms: float = 0.0
    repair_ms: float = 0.0
    total_ms: float = 0.0


class RepairResult(BaseModel):
    repair_id: str
    result_id: str
    question: str
    original_answer: str
    original_verification: RagVerificationInfo
    hallucination_check: HallucinationCheckResponse
    overview: RepairOverview
    claims: list[RepairClaimResult] = Field(default_factory=list)
    final_answer: str
    final_claims: list[FinalAnswerClaim] = Field(default_factory=list)
    omissions: list[RepairOmission] = Field(default_factory=list)
    iterations: list[RepairIteration] = Field(default_factory=list)
    status: RepairOverallStatus
    max_attempts: Literal[3] = 3
    model_provider: str | None = None
    model_name: str | None = None
    created_at: str
    timings: RepairTimings = Field(default_factory=RepairTimings)
    warnings: list[str] = Field(default_factory=list)


class RepairSourceResponse(BaseModel):
    """Everything the repair page needs before the Repair button is pressed."""

    result_id: str
    original: RagQueryResponse
    hallucination: HallucinationCheckResponse
    overview: RepairOverview
    repair: RepairResult | None = None
    max_attempts: Literal[3] = 3


class RepairRunRequest(BaseModel):
    result_id: str = Field(
        min_length=1,
        max_length=128,
        description="Result identifier returned by POST /api/rag/query.",
    )
    max_attempts: Literal[3] = Field(
        default=3,
        description="Fixed maximum repair iterations.",
    )
