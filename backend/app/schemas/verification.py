"""Phase 4A claim-level verification schemas.

These models describe observable NLP outputs only. NLI confidence values are
model scores, not calibrated probabilities that a claim is factually true.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NliLabel = Literal["entailment", "contradiction", "neutral"]
ClaimVerdict = Literal["supported", "contradicted", "neutral"]
ClaimVerificationStatus = ClaimVerdict
ClaimDiagnosticStatus = Literal["unevaluable", "verification_error"]
VerificationRunStatus = Literal["disabled", "completed", "partial", "error"]
ComparisonRelevance = Literal[
    "relevant",
    "weakly_relevant",
    "irrelevant",
    "non_substantive",
    "unknown",
]
PremiseStrategy = Literal["full_evidence", "focused_window", "unavailable"]
EvidenceQuality = Literal[
    "substantive",
    "table_of_contents",
    "heading_or_index",
    "boilerplate",
    "short",
    "empty",
]


class DetectedLanguage(BaseModel):
    code: str | None = None
    confidence: float | None = None
    reliable: bool = False
    mixed: bool = False
    method: str = "unknown"
    reason: str | None = None


class TextSpan(BaseModel):
    start: int
    end: int
    text: str


class TokenSpan(BaseModel):
    index: int
    token_id: int
    text: str
    start: int | None = None
    end: int | None = None


class NamedEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float | None = None


class ExtractedClaim(BaseModel):
    claim_id: str
    sentence_index: int
    sentence: str
    sentence_start: int
    sentence_end: int
    claim: str
    claim_start: int
    claim_end: int
    hypothesis: str = ""
    context: str | None = None
    complete: bool = True
    evaluable: bool = True
    incomplete_reason: str | None = None
    language: DetectedLanguage
    entities: list[NamedEntity] = Field(default_factory=list)
    extraction_method: str
    warnings: list[str] = Field(default_factory=list)

    @field_validator("hypothesis")
    @classmethod
    def hypothesis_defaults_to_claim(cls, value: str, info: object) -> str:
        if value.strip():
            return value
        data = getattr(info, "data", {})
        claim_text = data.get("claim")
        return str(claim_text) if claim_text else value


class EvidenceReference(BaseModel):
    evidence_id: str
    document_id: str | None = None
    filename: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    text_preview: str = ""


class NliWindowPrediction(BaseModel):
    window_index: int
    start: int
    end: int
    text: str = ""
    label: NliLabel
    confidence: float
    probabilities: dict[str, float]
    logits: list[float] | None = None


class EvidenceVerification(BaseModel):
    claim_id: str = ""
    hypothesis_text: str = ""
    evidence: EvidenceReference
    evidence_text: str = ""
    rank: int
    similarity: float
    retrieval_similarity: float | None = None
    retrieval_score: float | None = None
    evaluated: bool = True
    label: NliLabel | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    logits: list[float] | None = None
    relevance: ComparisonRelevance = "unknown"
    relevance_score: float = 0.0
    claim_term_coverage: float = 0.0
    counts_for_verdict: bool = False
    premise_text: str = ""
    premise_strategy: PremiseStrategy = "unavailable"
    evidence_char_length: int = 0
    premise_char_length: int = 0
    truncated: bool = False
    window_start: int | None = None
    window_end: int | None = None
    evidence_quality: EvidenceQuality | None = None
    evidence_quality_reason: str | None = None
    full_pair_token_count: int | None = None
    input_pair_token_count: int | None = None
    max_pair_tokens: int | None = None
    nli_input_order: str = "premise_evidence,hypothesis_claim"
    conflict: bool = False
    explanation: str = ""
    windows: list[NliWindowPrediction] = Field(default_factory=list)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    claim: ExtractedClaim
    status: ClaimVerdict | None = None
    diagnostic_status: ClaimDiagnosticStatus | None = None
    label: NliLabel | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    logits: list[float] | None = None
    support_confidence: float | None = None
    contradiction_confidence: float | None = None
    conflict: bool = False
    composite_premise: str | None = None
    explanation: str = ""
    pairs_expected: int = 0
    pairs_evaluated: int = 0
    pairs_retried: int = 0
    pairs_failed: int = 0
    complete: bool = True
    incomplete_reason: str | None = None
    matched_evidence: list[EvidenceVerification] = Field(default_factory=list)
    aggregation: str = "exhaustive_pair_thresholds"
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class VerificationSummary(BaseModel):
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
    evidence_passed: int = 0
    evidence_total: int = 0
    pairs_expected: int = 0
    pairs_evaluated: int = 0
    pairs_missing: int = 0
    pairs_retried: int = 0
    pairs_failed: int = 0


class VerificationTimings(BaseModel):
    claim_extraction_ms: float = 0.0
    evidence_selection_ms: float = 0.0
    nli_ms: float = 0.0
    aggregation_ms: float = 0.0


class VerificationResult(BaseModel):
    status: VerificationRunStatus = "completed"
    enabled: bool = True
    answer_language: DetectedLanguage = Field(default_factory=DetectedLanguage)
    claim_extraction_method: str = "unknown"
    nli_model: str | None = None
    ner_model: str | None = None
    claims: list[ClaimVerificationResult] = Field(default_factory=list)
    summary: VerificationSummary = Field(default_factory=VerificationSummary)
    timings: VerificationTimings = Field(default_factory=VerificationTimings)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    verification_ms: float = 0.0


class VerificationEvidenceInput(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    evidence_id: str | None = None
    document_id: str | None = None
    filename: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    retrieval_similarity: float | None = None
    retrieval_score: float | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence text must not be blank")
        return value


class VerificationCheckRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)
    evidence: list[VerificationEvidenceInput] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be blank")
        return value

