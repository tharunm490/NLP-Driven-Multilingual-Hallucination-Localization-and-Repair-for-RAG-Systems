"""RAG orchestration: language detection -> preprocessing -> embedding ->
ChromaDB retrieval -> evidence policy -> prompt -> LLM -> response.

The route handler stays thin; every stage lives in its own service.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from app.core.config import Settings
from app.schemas.rag import (
    CompareQueryResponse,
    EvalResponse,
    RagGenerationInfo,
    RagQueryResponse,
    RagRetrievalInfo,
    RagSource,
    RagTimings,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.evaluation_service import EvaluationService
from app.services.generator import ChatMessage, Generator
from app.services.prompt_builder import PromptBuilder, PromptSource
from app.services.query_language_detection import LanguageDetection, QueryLanguageDetector
from app.services.query_processor import preprocess_query
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

# Insufficient-evidence messages. These are system-generated and contain no
# claims about the documents; they simply state that no answer was possible.
_INSUFFICIENT_MESSAGES: dict[str, str] = {
    "en": (
        "I could not find sufficiently relevant evidence in the uploaded documents "
        "to answer this question."
    ),
    "hi": (
        "इस प्रश्न का उत्तर देने के लिए अपलोड किए गए दस्तावेज़ों में पर्याप्त "
        "प्रासंगिक साक्ष्य नहीं मिला।"
    ),
    "ta": (
        "இந்தக் கேள்விக்கு பதிலளிக்கும் அளவுக்குப் போதுமான தொடர்புடைய ஆதாரங்கள் "
        "பதிவேற்றப்பட்ட ஆவணங்களில் இல்லை."
    ),
}

_INSUFFICIENT_NOTE = " (No retrieved chunk passed the configured retrieval policy.)"


class RagService:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        generator: Generator,
        language_detector: QueryLanguageDetector,
        prompt_builder: PromptBuilder,
        evaluation_service: EvaluationService,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval_service
        self._generator = generator
        self._detector = language_detector
        self._prompt_builder = prompt_builder
        self._evaluator = evaluation_service

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

        sources = [self._to_source(chunk) for chunk in search.results]
        sufficient = self._evaluate_evidence(search.results)

        answer_text, generation_info, generation_ms = self._generate(
            question=preprocessed.normalized,
            sources=sources,
            sufficient=sufficient,
            detection=detection,
        )

        total_ms = (time.perf_counter() - total_start) * 1000
        answer_mode = "grounded" if sufficient else "insufficient_evidence"

        logger.info(
            "RAG answer: lang=%s reliable=%s top_k=%s returned=%s sufficient=%s "
            "mode=%s retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f",
            detection.code if detection.reliable else None,
            detection.reliable,
            effective_top_k,
            len(sources),
            sufficient,
            answer_mode,
            retrieval_ms,
            generation_ms,
            total_ms,
        )

        return RagQueryResponse(
            question=question,
            normalized_question=preprocessed.normalized,
            detected_language=detection.code if detection.reliable else None,
            language_confidence=detection.confidence,
            language_reliable=detection.reliable,
            answer=answer_text,
            answer_mode=answer_mode,
            sources=sources,
            retrieval=RagRetrievalInfo(
                top_k=effective_top_k,
                returned_chunks=len(sources),
                sufficient_evidence=sufficient,
                min_evidence_chunks=self._settings.rag_min_evidence_chunks,
                max_distance=(
                    self._settings.rag_max_distance
                    if self._settings.rag_distance_filter_enabled
                    else None
                ),
                distance_filter_applied=self._settings.rag_distance_filter_enabled,
            ),
            generation=generation_info,
            timings=RagTimings(
                retrieval_ms=round(retrieval_ms, 3),
                generation_ms=round(generation_ms, 3),
                total_ms=round(total_ms, 3),
            ),
        )

    def evaluate(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> EvalResponse:
        """Run RAG + LLM-only, score both against the retrieved evidence chunks.

        The retrieved chunks from the PDF are used as the reference — the RAG
        answer should score high (it was grounded in them) while the LLM-only
        answer should score lower (it had no access to them), demonstrating
        pipeline robustness.
        """
        rag_response = self.answer(question=question, top_k=top_k, document_ids=document_ids)

        # Use retrieved chunk texts as the reference
        evidence_text = " ".join(s.text for s in rag_response.sources).strip()
        if not evidence_text:
            evidence_text = question  # fallback: no chunks retrieved

        bare_prompt = self._prompt_builder.build_bare_llm(question)
        messages = [
            ChatMessage(role="system", content=bare_prompt.system_prompt),
            ChatMessage(role="user", content=bare_prompt.user_message),
        ]
        started = time.perf_counter()
        llm_result = self._generator.generate(messages)
        llm_only_ms = (time.perf_counter() - started) * 1000

        rag_metrics = self._evaluator.score(rag_response.answer, evidence_text)
        llm_metrics = self._evaluator.score(llm_result.text, evidence_text)

        return EvalResponse(
            question=question,
            evidence_used_as_reference=evidence_text,
            rag_answer=rag_response.answer,
            rag_answer_mode=rag_response.answer_mode,
            llm_only_answer=llm_result.text,
            rag_metrics=rag_metrics,
            llm_only_metrics=llm_metrics,
            rag_sources_count=len(rag_response.sources),
            rag_retrieval_ms=rag_response.timings.retrieval_ms,
            rag_generation_ms=rag_response.timings.generation_ms,
            llm_only_generation_ms=round(llm_only_ms, 3),
        )

    # -- stages ------------------------------------------------------------

    def _generate(
        self,
        question: str,
        sources: Sequence[RagSource],
        sufficient: bool,
        detection: LanguageDetection,
    ) -> tuple[str, RagGenerationInfo, float]:
        if not sufficient and not self._settings.rag_generation_without_evidence:
            message = self._insufficient_message(detection)
            return (
                message,
                RagGenerationInfo(used_llm=False, prompt_version=self._prompt_builder.version),
                0.0,
            )

        prompt_sources = [
            PromptSource(
                index=index + 1,
                document_id=source.document_id,
                filename=source.filename,
                chunk_id=source.chunk_id,
                page_number=source.page_number,
                text=source.text,
            )
            for index, source in enumerate(sources)
        ]
        built = (
            self._prompt_builder.build(question, prompt_sources)
            if sources
            else self._prompt_builder.build_insufficient_evidence(question)
        )
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

    def _evaluate_evidence(self, results: Sequence[RetrievedChunk]) -> bool:
        """Configurable retrieval policy: enough chunks and close enough matches."""
        if len(results) < self._settings.rag_min_evidence_chunks:
            return False
        if not self._settings.rag_distance_filter_enabled:
            return True
        threshold = self._settings.rag_max_distance
        return any(chunk.distance <= threshold for chunk in results)

    def _effective_top_k(self, top_k: int | None) -> int:
        requested = top_k if top_k is not None else self._settings.top_k
        return max(1, min(requested, 50))

    @staticmethod
    def _to_source(chunk: RetrievedChunk) -> RagSource:
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
        )

    def compare(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> CompareQueryResponse:
        """Run the question through both the RAG pipeline and a bare LLM (no context)."""
        rag_response = self.answer(question=question, top_k=top_k, document_ids=document_ids)

        detection = self._detector.detect(question)
        bare_prompt = self._prompt_builder.build_bare_llm(question)
        messages = [
            ChatMessage(role="system", content=bare_prompt.system_prompt),
            ChatMessage(role="user", content=bare_prompt.user_message),
        ]
        started = time.perf_counter()
        result = self._generator.generate(messages)
        llm_only_ms = (time.perf_counter() - started) * 1000

        return CompareQueryResponse(
            question=question,
            detected_language=detection.code if detection.reliable else None,
            language_reliable=detection.reliable,
            rag=rag_response,
            llm_only_answer=result.text,
            llm_only_generation=RagGenerationInfo(
                used_llm=True,
                provider=result.provider,
                model=result.model,
                prompt_version="llm-only-baseline-1.0.0",
                finish_reason=result.finish_reason,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
            llm_only_generation_ms=round(llm_only_ms, 3),
        )

    def _insufficient_message(self, detection: LanguageDetection) -> str:
        code = detection.code if detection.code else "en"
        base = _INSUFFICIENT_MESSAGES.get(code, _INSUFFICIENT_MESSAGES["en"])
        return base + ("" if code == "en" else _INSUFFICIENT_NOTE)
