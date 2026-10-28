"""Deterministic test doubles for hallucination checks."""

from __future__ import annotations

from typing import Sequence

from app.core.errors import HallucinationModelError
from app.schemas.rag import (
    RagAssessmentInfo,
    RagDiagnostics,
    RagGenerationInfo,
    RagQueryResponse,
    RagRetrievalInfo,
    RagSource,
    RagTimings,
    RagVerificationInfo,
)
from app.schemas.verification import (
    ClaimVerificationResult,
    DetectedLanguage,
    ExtractedClaim,
    VerificationResult,
    VerificationSummary,
)
from app.services.hallucination_detector import HallucinationPrediction


class FakeHallucinationDetector:
    model_name = "test/hallucination-guard"

    def __init__(
        self,
        default: float = 0.9,
        by_claim: dict[str, float] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self._default = default
        self._by_claim = dict(by_claim or {})
        self._fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        deadline: float | None = None,
        cancel_event=None,
    ) -> list[HallucinationPrediction]:
        if self._fail:
            raise HallucinationModelError("simulated local model failure")
        self.calls.append(list(pairs))
        predictions: list[HallucinationPrediction] = []
        for _, claim in pairs:
            probability = next(
                (
                    score
                    for marker, score in self._by_claim.items()
                    if marker.lower() in claim.lower()
                ),
                self._default,
            )
            predictions.append(
                HallucinationPrediction(
                    raw_label=(
                        "SUPPORTED" if probability >= 0.5 else "UNSUPPORTED"
                    ),
                    support_probability=probability,
                )
            )
        return predictions


def source(
    chunk_id: str,
    text: str,
    *,
    filename: str = "iso-27001.txt",
    page_number: int | None = 7,
) -> RagSource:
    return RagSource(
        document_id="doc-1",
        filename=filename,
        chunk_id=chunk_id,
        chunk_index=int(chunk_id.removeprefix("e") or 0),
        page_number=page_number,
        language="en",
        text=text,
        retrieval_score=0.12,
        similarity=0.88,
        passed_relevance_filter=True,
        relevance_reason="distance_within_threshold",
    )


def rag_response(
    answer: str,
    sources: Sequence[RagSource],
    *,
    verification: VerificationResult | None = None,
    used_llm: bool = True,
    result_id: str = "result-1",
) -> RagQueryResponse:
    verification_info = RagVerificationInfo(
        claim_level_verification=verification is not None,
        status="verified" if verification is not None else "not_performed",
        method="multilingual_nli" if verification is not None else "none",
        note="test verification",
        details=verification,
    )
    return RagQueryResponse(
        result_id=result_id,
        question="What does the standard require?",
        normalized_question="What does the standard require?",
        detected_language="en",
        language_confidence=0.99,
        language_reliable=True,
        answer=answer,
        answer_mode="verification_pending" if used_llm else "insufficient_evidence",
        answer_status="verification_pending" if used_llm else "insufficient_evidence",
        evidence_state="sufficient_evidence",
        sources=list(sources),
        retrieval=RagRetrievalInfo(
            top_k=5,
            returned_chunks=len(sources),
            sufficient_evidence=bool(sources),
            min_evidence_chunks=1,
        ),
        assessment=RagAssessmentInfo(
            state="sufficient_evidence",
            method="test",
            used_llm=True,
        ),
        verification=verification_info,
        generation=RagGenerationInfo(
            used_llm=used_llm,
            provider="test",
            model="test-answer-model",
            prompt_version="test",
        ),
        diagnostics=RagDiagnostics(),
        timings=RagTimings(
            retrieval_ms=1.0,
            generation_ms=1.0,
            total_ms=2.0,
        ),
    )


def nli_result(
    answer: str,
    statuses: Sequence[tuple[str, str, float | None]],
) -> VerificationResult:
    """Build NLI details from (claim, status, confidence) tuples."""
    claims: list[ClaimVerificationResult] = []
    for index, (text, status, confidence) in enumerate(statuses, start=1):
        start = answer.index(text)
        claims.append(
            ClaimVerificationResult(
                claim=ExtractedClaim(
                    claim_id=f"c{index}",
                    sentence_index=index,
                    sentence=text,
                    sentence_start=start,
                    sentence_end=start + len(text),
                    claim=text,
                    claim_start=start,
                    claim_end=start + len(text),
                    hypothesis=text,
                    language=DetectedLanguage(code="en", confidence=0.99, reliable=True),
                    extraction_method="test",
                ),
                status=status,  # type: ignore[arg-type]
                confidence=confidence,
                complete=True,
            )
        )
    supported = sum(claim.status == "supported" for claim in claims)
    contradicted = sum(claim.status == "contradicted" for claim in claims)
    neutral = sum(claim.status == "neutral" for claim in claims)
    return VerificationResult(
        status="completed",
        claims=claims,
        summary=VerificationSummary(
            claims_total=len(claims),
            claims_evaluable=len(claims),
            claims_checked=len(claims),
            claims_complete=len(claims),
            supported=supported,
            contradicted=contradicted,
            neutral=neutral,
            evidence_passed=1,
            evidence_total=1,
            pairs_expected=len(claims),
            pairs_evaluated=len(claims),
        ),
    )
