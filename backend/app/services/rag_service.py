"""RAG orchestration: language detection -> preprocessing -> embedding ->
ChromaDB retrieval -> relevance filtering -> evidence sufficiency -> prompt ->
LLM -> explicit answer status.

The route handler stays thin; every stage lives in its own service. In
particular, retrieval relevance and evidence sufficiency are separate gates:

* Retrieval relevance: reject chunks above the configured cosine-distance
  threshold before they are shown or sent to the model.
* Evidence sufficiency: decide whether the remaining evidence actually answers
  the question. This can yield sufficient, partial, insufficient, or
  irrelevant evidence.
* Claim verification: generated answers start as ``verification_pending``.
  NLI runs only when the user explicitly requests it for the stored result.

Generated answers are never labeled ``grounded`` before on-demand claim-level
verification has completed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

from app.core.config import Settings
from app.core.errors import EvidenceAssessmentError
from app.schemas.rag import (
    AnswerStatus,
    RagAssessmentInfo,
    RagDiagnostics,
    RagGenerationInfo,
    RagQueryResponse,
    RagRejectedChunk,
    RagRetrievalInfo,
    RagSource,
    RagTimings,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.evidence_assessor import (
    EvidenceAssessment,
    EvidenceAssessor,
    build_evidence_assessor,
)
from app.services.generator import ChatMessage, Generator
from app.services.prompt_builder import PromptBuilder, PromptSource
from app.services.query_language_detection import LanguageDetection, QueryLanguageDetector
from app.services.query_processor import preprocess_query
from app.services.rag_verification_service import initial_verification_info
from app.services.relevance_filter import (
    DISTANCE_METRIC,
    SCORE_INTERPRETATION,
    RelevanceFilterResult,
    filter_relevant_chunks,
)
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# System-generated abstention messages. They contain no factual claims about
# the documents; they only state that no answer was possible.
_INSUFFICIENT_MESSAGES: dict[str, str] = {
    "en": (
        "The uploaded documents do not contain sufficient relevant information "
        "to answer this question."
    ),
    "hi": (
        "इस प्रश्न का उत्तर देने के लिए अपलोड किए गए दस्तावेज़ों में पर्याप्त "
        "प्रासंगिक जानकारी नहीं है।"
    ),
    "ta": (
        "இந்தக் கேள்விக்கு பதிலளிக்கும் அளவுக்கு போதுமான தொடர்புடைய தகவல்கள் "
        "பதிவேற்றப்பட்ட ஆவணங்களில் இல்லை."
    ),
}
_IRRELEVANT_MESSAGES: dict[str, str] = {
    "en": (
        "No relevant supporting passage was found for this question, "
        "so no answer was generated from these passages."
    ),
    "hi": (
        "इस प्रश्न के लिए अपलोड किए गए दस्तावेज़ों में कोई प्रासंगिक सहायक अंश "
        "नहीं मिला। इसलिए इन अंशों से कोई उत्तर नहीं बनाया गया।"
    ),
    "ta": (
        "இந்தக் கேள்விக்கு பதிவேற்றப்பட்ட ஆவணங்களில் தொடர்புடைய ஆதாரப் பகுதி "
        "எதுவும் கிடைக்கவில்லை. எனவே இந்தப் பகுதிகளிலிருந்து பதில் உருவாக்கப்படவில்லை."
    ),
}


class RagService:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        generator: Generator,
        language_detector: QueryLanguageDetector,
        prompt_builder: PromptBuilder,
        evidence_assessor: EvidenceAssessor | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval_service
        self._generator = generator
        self._detector = language_detector
        self._prompt_builder = prompt_builder
        self._assessor = evidence_assessor or build_evidence_assessor(settings, generator)

    # -- public API --------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RagQueryResponse:
        total_start = time.perf_counter()

        detection = self._detector.detect(question)
        preprocessed = preprocess_query(
            question, max_length=self._settings.rag_max_question_length
        )
        effective_top_k = self._effective_top_k(top_k)

        retrieval_start = time.perf_counter()
        search = self._retrieval.search(
            query=preprocessed.normalized,
            top_k=effective_top_k,
            document_ids=document_ids,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        relevance = filter_relevant_chunks(search.results, self._settings)
        assessment_start = time.perf_counter()
        decision = self._decide(
            question=preprocessed.normalized,
            detection=detection,
            relevance=relevance,
        )
        assessment_ms = (time.perf_counter() - assessment_start) * 1000

        sources = self._sources_for_decision(decision, relevance)
        prompt_sources = [self._to_prompt_source(index, source) for index, source in enumerate(sources, start=1)]

        if decision.should_generate:
            answer_text, generation_info, generation_ms = self._generate(
                question=preprocessed.normalized,
                sources=sources,
                prompt_sources=prompt_sources,
                assessment=decision.assessment,
            )
        else:
            answer_text = decision.answer_text or self._abstention_message(
                detection, decision.status
            )
            generation_info = RagGenerationInfo(
                used_llm=False,
                prompt_version=self._prompt_builder.version,
            )
            generation_ms = 0.0

        answer_status = decision.status
        total_ms = (time.perf_counter() - total_start) * 1000
        verification = initial_verification_info(generation_info.used_llm)
        errors = list(decision.errors)

        logger.info(
            "RAG decision: lang=%s reliable=%s top_k=%s candidates=%s filtered=%s "
            "evidence=%s assessment=%s status=%s llm=%s "
            "retrieval_ms=%.1f assessment_ms=%.1f generation_ms=%.1f "
            "total_ms=%.1f",
            detection.code if detection.reliable else None,
            detection.reliable,
            effective_top_k,
            len(relevance.candidates),
            len(relevance.kept),
            decision.assessment.state,
            decision.assessment.method,
            answer_status,
            generation_info.used_llm,
            retrieval_ms,
            assessment_ms,
            generation_ms,
            total_ms,
        )
        logger.debug(
            "RAG diagnostics: preview=%r rejected=%s fallback=%s skip_reason=%s",
            preprocessed.normalized[:160],
            [item.chunk.chunk_id for item in relevance.rejected],
            decision.assessment.fallback_reason,
            decision.skip_reason,
        )

        return RagQueryResponse(
            question=question,
            normalized_question=preprocessed.normalized,
            detected_language=detection.code if detection.reliable else None,
            language_confidence=detection.confidence,
            language_reliable=detection.reliable,
            answer=answer_text,
            answer_mode=answer_status,
            answer_status=answer_status,
            evidence_state=decision.assessment.state,
            sources=sources,
            retrieval=RagRetrievalInfo(
                top_k=effective_top_k,
                candidate_chunks=len(relevance.candidates),
                filtered_chunks=len(relevance.kept),
                returned_chunks=len(sources),
                sufficient_evidence=decision.assessment.state == "sufficient_evidence",
                metric=DISTANCE_METRIC,
                score_interpretation=SCORE_INTERPRETATION,
                min_evidence_chunks=self._settings.rag_min_evidence_chunks,
                max_distance=relevance.threshold,
                distance_filter_applied=relevance.filter_enabled,
                rejected_chunks=[
                    self._to_rejected_chunk(item.chunk, item.reason)
                    for item in relevance.rejected
                ],
            ),
            assessment=RagAssessmentInfo(
                state=decision.assessment.state,
                method=decision.assessment.method,
                used_llm=decision.assessment.used_llm,
                reason=decision.assessment.reason,
                supported_aspects=list(decision.assessment.supported_aspects),
                missing_aspects=list(decision.assessment.missing_aspects),
                premise_contradicted=decision.assessment.premise_contradicted,
                fallback_reason=decision.assessment.fallback_reason,
            ),
            verification=verification,
            generation=generation_info,
            diagnostics=RagDiagnostics(
                question_preview=preprocessed.normalized[:160],
                candidates_retrieved=len(relevance.candidates),
                candidates_after_filter=len(relevance.kept),
                rejected_chunk_ids=[
                    item.chunk.chunk_id for item in relevance.rejected
                ],
                assessment_method=decision.assessment.method,
                assessment_fallback_reason=decision.assessment.fallback_reason,
                generation_skipped=not generation_info.used_llm,
                generation_skip_reason=decision.skip_reason,
                verification_error=None,
                errors=errors,
            ),
            timings=RagTimings(
                retrieval_ms=round(retrieval_ms, 3),
                assessment_ms=round(assessment_ms, 3),
                generation_ms=round(generation_ms, 3),
                verification_ms=0.0,
                total_ms=round(total_ms, 3),
            ),
        )

    # -- decision stage ----------------------------------------------------

    def _decide(
        self,
        question: str,
        detection: LanguageDetection,
        relevance: RelevanceFilterResult,
    ) -> "_Decision":
        if not relevance.candidates:
            assessment = EvidenceAssessment(
                state="insufficient_evidence",
                method="retrieval_only",
                reason="No candidate chunks were returned by retrieval.",
            )
            return _Decision(
                status="insufficient_evidence",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "insufficient_evidence"),
                skip_reason="no_candidates",
                should_generate=self._settings.rag_generation_without_evidence,
            )

        if not relevance.kept:
            threshold = relevance.threshold
            reason = (
                "All retrieved candidates were above the configured cosine-distance "
                f"threshold ({threshold:.4f})." if threshold is not None
                else "No retrieved candidate had a usable relevance score."
            )
            assessment = EvidenceAssessment(
                state="irrelevant_evidence",
                method="retrieval_only",
                reason=reason,
            )
            return _Decision(
                status="irrelevant_retrieval",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "irrelevant_retrieval"),
                skip_reason="no_chunk_passed_relevance_filter",
                should_generate=self._settings.rag_generation_without_evidence,
            )

        if len(relevance.kept) < self._settings.rag_min_evidence_chunks:
            assessment = EvidenceAssessment(
                state="insufficient_evidence",
                method="retrieval_only",
                reason=(
                    f"Only {len(relevance.kept)} chunk(s) passed relevance filtering, "
                    f"below RAG_MIN_EVIDENCE_CHUNKS="
                    f"{self._settings.rag_min_evidence_chunks}."
                ),
            )
            return _Decision(
                status="insufficient_evidence",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "insufficient_evidence"),
                skip_reason="below_min_evidence_chunks",
                should_generate=self._settings.rag_generation_without_evidence,
            )

        decision = self._assess_sufficiency(question, detection, relevance.kept)
        if decision.status in {"irrelevant_retrieval", "insufficient_evidence"}:
            decision.should_generate = self._settings.rag_generation_without_evidence
        return decision

    def _assess_sufficiency(
        self,
        question: str,
        detection: LanguageDetection,
        chunks: Sequence[RetrievedChunk],
    ) -> "_Decision":
        try:
            assessment = self._assessor.assess(question=question, chunks=chunks)
        except EvidenceAssessmentError as exc:
            logger.warning("Evidence assessment failed; abstaining: %s", exc)
            assessment = EvidenceAssessment(
                state="insufficient_evidence",
                method="assessment_failed",
                reason=(
                    "The evidence-sufficiency judge could not produce a reliable "
                    "decision, so the system abstained conservatively."
                ),
                fallback_reason=str(exc),
            )
            return _Decision(
                status="insufficient_evidence",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "insufficient_evidence"),
                skip_reason="evidence_assessment_failed",
                should_generate=self._settings.rag_generation_without_evidence,
                errors=[str(exc)],
            )

        if assessment.state == "irrelevant_evidence":
            return _Decision(
                status="irrelevant_retrieval",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "irrelevant_retrieval"),
                skip_reason="evidence_judged_irrelevant",
            )
        if assessment.state == "insufficient_evidence":
            return _Decision(
                status="insufficient_evidence",
                assessment=assessment,
                answer_text=self._abstention_message(detection, "insufficient_evidence"),
                skip_reason="evidence_judged_insufficient",
            )
        if assessment.state == "partial_evidence":
            return _Decision(
                status="partially_grounded",
                assessment=assessment,
                should_generate=True,
            )
        return _Decision(
            status="verification_pending",
            assessment=assessment,
            should_generate=True,
        )

    # -- generation --------------------------------------------------------

    def _generate(
        self,
        question: str,
        sources: Sequence[RagSource],
        prompt_sources: Sequence[PromptSource],
        assessment: EvidenceAssessment,
    ) -> tuple[str, RagGenerationInfo, float]:
        if prompt_sources:
            built = self._prompt_builder.build(
                question,
                prompt_sources,
                evidence_state=assessment.state,
                supported_aspects=assessment.supported_aspects,
                missing_aspects=assessment.missing_aspects,
                premise_contradicted=assessment.premise_contradicted,
            )
        else:
            built = self._prompt_builder.build_insufficient_evidence(question)
        messages = [
            ChatMessage(role="system", content=built.system_prompt),
            ChatMessage(role="user", content=built.user_message),
        ]

        started = time.perf_counter()
        result = self._generator.generate(messages)
        generation_ms = (time.perf_counter() - started) * 1000

        return (
            result.text,
            RagGenerationInfo(
                used_llm=True,
                provider=result.provider,
                model=result.model,
                prompt_version=built.prompt_version,
                finish_reason=result.finish_reason,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
            generation_ms,
        )

    # -- helpers -----------------------------------------------------------

    def _sources_for_decision(
        self, decision: "_Decision", relevance: RelevanceFilterResult
    ) -> list[RagSource]:
        if decision.assessment.state == "irrelevant_evidence":
            return []
        reason = (
            "distance_within_threshold"
            if relevance.filter_enabled
            else "distance_filter_disabled"
        )
        return [
            self._to_source(chunk, reason)
            for chunk in relevance.kept
        ]

    def _effective_top_k(self, top_k: int | None) -> int:
        requested = top_k if top_k is not None else self._settings.top_k
        return max(1, min(requested, 50))

    @staticmethod
    def _to_source(chunk: RetrievedChunk, reason: str) -> RagSource:
        return RagSource(
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            language=chunk.language,
            text=chunk.text,
            retrieval_score=chunk.distance,
            similarity=chunk.similarity,
            passed_relevance_filter=True,
            relevance_reason=reason,
        )

    @staticmethod
    def _to_prompt_source(index: int, source: RagSource) -> PromptSource:
        return PromptSource(
            index=index,
            document_id=source.document_id,
            filename=source.filename,
            chunk_id=source.chunk_id,
            page_number=source.page_number,
            text=source.text,
        )

    @staticmethod
    def _to_rejected_chunk(chunk: RetrievedChunk, reason: str) -> RagRejectedChunk:
        return RagRejectedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            retrieval_score=chunk.distance,
            similarity=chunk.similarity,
            reason=reason,
        )

    def _abstention_message(self, detection: LanguageDetection, status: AnswerStatus) -> str:
        code = detection.code if detection.code else "en"
        messages = (
            _IRRELEVANT_MESSAGES if status == "irrelevant_retrieval" else _INSUFFICIENT_MESSAGES
        )
        return messages.get(code, messages["en"])


class _Decision:
    """Internal result of the evidence decision stage."""

    def __init__(
        self,
        status: AnswerStatus,
        assessment: EvidenceAssessment,
        answer_text: str | None = None,
        skip_reason: str | None = None,
        should_generate: bool = False,
        errors: list[str] | None = None,
    ) -> None:
        self.status = status
        self.assessment = assessment
        self.answer_text = answer_text
        self.skip_reason = skip_reason
        self.should_generate = should_generate
        self.errors = errors or []
