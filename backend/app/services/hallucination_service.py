"""Claim-level groundedness checks backed by a dedicated hallucination model."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

from app.core.config import Settings
from app.core.errors import (
    ClaimExtractionError,
    HallucinationCheckError,
    HallucinationModelError,
)
from app.schemas.hallucination import (
    HallucinationCheckResponse,
    HallucinationClaimResult,
    HallucinationEvidenceResult,
    HallucinationSummary,
    HallucinationTimings,
)
from app.schemas.rag import RagQueryResponse, RagSource
from app.schemas.verification import ClaimVerificationResult, ExtractedClaim
from app.services.generator import Generator
from app.services.hallucination_detector import (
    HallucinationDetector,
    HallucinationPrediction,
)
from app.services.verification.claims import (
    ClaimExtractor,
    build_claim_extractor,
)


class HallucinationCheckService:
    def __init__(
        self,
        settings: Settings,
        detector: HallucinationDetector,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._extractor = claim_extractor or build_claim_extractor(settings)
        self._cache: OrderedDict[str, HallucinationCheckResponse] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._result_locks: dict[str, threading.Lock] = {}
        self._result_locks_guard = threading.Lock()

    @property
    def detector(self) -> HallucinationDetector:
        """The shared detector instance, reused by iterative repair."""
        return self._detector

    def check(
        self,
        result: RagQueryResponse,
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> HallucinationCheckResponse:
        if not self._settings.hallucination_enabled:
            raise HallucinationCheckError(
                "Hallucination checks are disabled by configuration."
            )
        result_id = result.result_id
        if not result_id:
            raise HallucinationCheckError(
                "The Ask result has no identifier; rerun the question."
            )
        cached = self._get_cached(result_id)
        if cached is not None:
            return cached
        lock = self._lock_for(result_id)
        with lock:
            cached = self._get_cached(result_id)
            if cached is not None:
                return cached
            response = self._run(
                result,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            self._store_cached(result_id, response)
            return response

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def get_result(self, result_id: str) -> HallucinationCheckResponse | None:
        """Return the saved check for an Ask result without running inference."""
        return self._get_cached(result_id)

    def _run(
        self,
        result: RagQueryResponse,
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> HallucinationCheckResponse:
        started = time.perf_counter()
        answer = result.answer.strip()
        if not answer or not result.generation.used_llm:
            raise HallucinationCheckError(
                "This Ask result has no generated answer to check."
            )
        if not result.sources:
            raise HallucinationCheckError(
                "This Ask result has no retrieved evidence to check against."
            )

        extraction_started = time.perf_counter()
        try:
            extraction = self._extractor.extract(result.answer)
        except ClaimExtractionError as exc:
            raise HallucinationCheckError(
                f"Claim extraction failed: {exc}"
            ) from exc
        extraction_ms = (time.perf_counter() - extraction_started) * 1000
        evaluable_claims = [
            claim for claim in extraction.claims if claim.evaluable
        ]
        claims = list(evaluable_claims)
        warnings = list(extraction.warnings)
        if not claims:
            return HallucinationCheckResponse(
                check_id=uuid.uuid4().hex,
                result_id=result.result_id or "",
                question=result.question,
                model_name=self._detector.model_name,
                support_threshold=self._settings.hallucination_support_threshold,
                review_margin=self._settings.hallucination_review_margin,
                claims=[],
                summary=HallucinationSummary(),
                timings=HallucinationTimings(
                    claim_extraction_ms=round(extraction_ms, 3),
                    total_ms=round((time.perf_counter() - started) * 1000, 3),
                ),
                warnings=[*warnings, "No factual claims were extracted from the answer."],
            )

        evidence = list(
            result.sources[: self._settings.hallucination_max_evidence_per_claim]
        )
        max_pairs = self._settings.hallucination_max_pairs
        max_evidence = max(1, max_pairs // max(1, len(claims)))
        if len(evidence) > max_evidence:
            evidence = evidence[:max_evidence]
            warnings.append(
                "Evidence was limited to keep the number of model comparisons "
                "within the configured budget; every extracted claim was checked."
            )

        pairs: list[tuple[str, str]] = []
        pair_plan: list[tuple[ExtractedClaim, RagSource]] = []
        max_chars = self._settings.hallucination_max_evidence_chars
        for claim in claims:
            hypothesis = claim.hypothesis or claim.claim
            for source in evidence:
                pairs.append((source.text[:max_chars], hypothesis))
                pair_plan.append((claim, source))

        inference_started = time.perf_counter()
        try:
            predictions = self._detector.predict(
                pairs,
                deadline=deadline,
                cancel_event=cancel_event,
            )
        except HallucinationModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HallucinationModelError(
                "The local hallucination model failed while scoring this answer."
            ) from exc
        inference_ms = (time.perf_counter() - inference_started) * 1000
        if len(predictions) != len(pair_plan):
            raise HallucinationModelError(
                "The hallucination model returned an incomplete prediction set."
            )

        evidence_by_claim: dict[str, list[HallucinationEvidenceResult]] = {
            claim.claim_id: [] for claim in claims
        }
        for (claim, source), prediction in zip(pair_plan, predictions):
            evidence_by_claim[claim.claim_id].append(
                self._evidence_result(source, prediction)
            )

        nli_by_id, nli_by_text = self._nli_index(result)
        claim_results: list[HallucinationClaimResult] = []
        for claim in claims:
            claim_evidence = sorted(
                evidence_by_claim[claim.claim_id],
                key=lambda item: item.support_probability,
                reverse=True,
            )
            if not claim_evidence:
                claim_evidence = [
                    self._evidence_result(
                        source,
                        HallucinationPrediction(
                            raw_label="UNSCORED",
                            support_probability=0.0,
                        ),
                    )
                    for source in evidence
                ]
            best = max(
                claim_evidence,
                key=lambda item: item.support_probability,
            )
            verdict = self._verdict(best.raw_model_label)
            nli = nli_by_id.get(claim.claim_id) or nli_by_text.get(
                self._claim_key(claim.claim)
            )
            nli_status = nli.status if nli is not None else None
            nli_confidence = nli.confidence if nli is not None else None
            claim_results.append(
                HallucinationClaimResult(
                    claim_id=claim.claim_id,
                    sentence_index=claim.sentence_index,
                    sentence=claim.sentence,
                    claim=claim.claim,
                    claim_start=claim.claim_start,
                    claim_end=claim.claim_end,
                    verdict=verdict,
                    raw_model_label=best.raw_model_label,
                    support_probability=best.support_probability,
                    explanation=self._explanation(
                        best.support_probability,
                        verdict,
                        best.filename,
                        best.page_number,
                    ),
                    evidence=claim_evidence,
                    nli_status=nli_status,
                    nli_confidence=nli_confidence,
                    nli_disagrees=self._nli_disagrees(verdict, nli_status),
                )
            )

        summary = HallucinationSummary(
            claims_total=len(claim_results),
            supported=sum(item.verdict == "supported" for item in claim_results),
            potentially_hallucinated=sum(
                item.verdict == "potentially_hallucinated"
                for item in claim_results
            ),
            nli_disagreements=sum(item.nli_disagrees for item in claim_results),
            evidence_total=len(evidence),
            pairs_evaluated=len(predictions),
        )
        if result.verification is None or result.verification.details is None:
            warnings.append(
                "No independent NLI result was available for comparison."
            )
        return HallucinationCheckResponse(
            check_id=uuid.uuid4().hex,
            result_id=result.result_id or "",
            question=result.question,
            model_name=self._detector.model_name,
            support_threshold=self._settings.hallucination_support_threshold,
            review_margin=self._settings.hallucination_review_margin,
            claims=claim_results,
            summary=summary,
            timings=HallucinationTimings(
                claim_extraction_ms=round(extraction_ms, 3),
                inference_ms=round(inference_ms, 3),
                total_ms=round((time.perf_counter() - started) * 1000, 3),
            ),
            warnings=warnings,
        )

    @staticmethod
    def _verdict(raw_model_label: str) -> str:
        if raw_model_label.strip().upper() == "SUPPORTED":
            return "supported"
        return "potentially_hallucinated"

    @staticmethod
    def _evidence_result(
        source: RagSource,
        prediction: HallucinationPrediction,
    ) -> HallucinationEvidenceResult:
        return HallucinationEvidenceResult(
            evidence_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            chunk_id=source.chunk_id,
            chunk_index=source.chunk_index,
            page_number=source.page_number,
            retrieval_score=source.retrieval_score,
            retrieval_similarity=source.similarity,
            support_probability=prediction.support_probability,
            raw_model_label=prediction.raw_label,
        )

    @staticmethod
    def _explanation(
        probability: float,
        verdict: str,
        filename: str,
        page_number: int | None,
    ) -> str:
        location = filename if page_number is None else f"{filename}, page {page_number}"
        if verdict == "supported":
            return (
                f"The supplied evidence ({location}) scored "
                f"{probability:.3f} support for this claim."
            )
        return (
            f"The best supplied evidence ({location}) scored only "
            f"{probability:.3f} support; the claim is not supported by the "
            "supplied evidence."
        )

    @staticmethod
    def _nli_disagrees(
        verdict: str,
        nli_status: str | None,
    ) -> bool:
        if nli_status is None:
            return False
        if verdict == "supported":
            return nli_status in {"contradicted", "neutral"}
        return nli_status == "supported"

    def _nli_index(
        self, result: RagQueryResponse
    ) -> tuple[
        dict[str, ClaimVerificationResult],
        dict[str, ClaimVerificationResult],
    ]:
        by_id: dict[str, ClaimVerificationResult] = {}
        by_text: dict[str, ClaimVerificationResult] = {}
        details = (
            result.verification.details
            if result.verification is not None
            else None
        )
        if details is None:
            return by_id, by_text
        for claim in details.claims:
            by_id[claim.claim.claim_id] = claim
            by_text[self._claim_key(claim.claim.claim)] = claim
        return by_id, by_text

    @staticmethod
    def _claim_key(value: str) -> str:
        return " ".join(value.lower().split())

    def _get_cached(self, result_id: str) -> HallucinationCheckResponse | None:
        with self._cache_lock:
            cached = self._cache.get(result_id)
            if cached is not None:
                self._cache.move_to_end(result_id)
            return cached

    def _store_cached(
        self,
        result_id: str,
        response: HallucinationCheckResponse,
    ) -> None:
        with self._cache_lock:
            self._cache[result_id] = response
            self._cache.move_to_end(result_id)
            while len(self._cache) > self._settings.hallucination_cache_size:
                self._cache.popitem(last=False)

    def _lock_for(self, result_id: str) -> threading.Lock:
        with self._result_locks_guard:
            lock = self._result_locks.get(result_id)
            if lock is None:
                lock = threading.Lock()
                self._result_locks[result_id] = lock
            return lock


def build_hallucination_check_service(
    settings: Settings,
    generator: Generator | None = None,
    detector: object | None = None,
) -> HallucinationCheckService:
    from app.services.hallucination_detector import build_hallucination_detector

    return HallucinationCheckService(
        settings=settings,
        detector=build_hallucination_detector(settings),
        claim_extractor=build_claim_extractor(settings, generator, detector),
    )


__all__ = [
    "HallucinationCheckService",
    "build_hallucination_check_service",
]
