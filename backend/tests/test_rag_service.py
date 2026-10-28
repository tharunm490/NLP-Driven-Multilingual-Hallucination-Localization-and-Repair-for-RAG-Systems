"""Service-level tests for the RAG pipeline (mocked LLM, fake embeddings)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import QueryValidationError
from app.services.document_processor import DocumentProcessor
from app.services.generator import MockGenerator
from app.services.ingestion_service import IngestionService
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService

DOCUMENT_TEXT = (
    "Acme Corporation is a technology company. It was founded in 2015 by "
    "Ada Lovelace in Paris, France."
)

QUESTION = "When was Acme Corporation founded?"
EXPECTED = "2015"


def ingest(service: IngestionService, text: str, name: str = "acme.txt"):
    result = service.ingest(name, "text/plain", text.encode("utf-8"))
    assert result.status == "completed", result.error
    return result


def test_grounded_answer_uses_evidence(rag_service: RagService, mock_generator, ingestion_service):
    document = ingest(ingestion_service, DOCUMENT_TEXT)
    response = rag_service.answer(QUESTION)

    assert response.answer_mode == "verification_pending"
    assert response.answer_status == "verification_pending"
    assert response.evidence_state == "sufficient_evidence"
    assert response.verification.claim_level_verification is False
    assert response.retrieval.sufficient_evidence is True
    assert response.retrieval.returned_chunks >= 1
    assert response.sources
    source = response.sources[0]
    assert source.document_id == document.document_id
    assert source.filename == "acme.txt"
    assert source.chunk_id
    assert source.text
    assert source.retrieval_score == pytest.approx(source.retrieval_score)
    assert source.similarity == pytest.approx(1.0 - source.retrieval_score)

    assert response.generation.used_llm is True
    assert response.generation.provider == "mock"
    assert response.generation.prompt_version
    assert response.timings.total_ms >= 0.0


def test_evidence_reaches_the_generation_service(rag_service: RagService, mock_generator, ingestion_service):
    document = ingest(ingestion_service, DOCUMENT_TEXT)
    rag_service.answer(QUESTION)

    assert len(mock_generator.calls) == 1
    messages = mock_generator.calls[0]
    assert [message.role for message in messages] == ["system", "user"]
    user_message = messages[1].content
    assert document.original_filename in user_message
    assert document.document_id in user_message
    assert "[Source 1]" in user_message
    assert "untrusted" in user_message.lower()


def test_answer_preserves_original_question(rag_service: RagService, ingestion_service):
    ingest(ingestion_service, DOCUMENT_TEXT)
    response = rag_service.answer(f"  {QUESTION}  ")
    assert response.question == f"  {QUESTION}  "
    assert response.normalized_question == QUESTION


def test_empty_collection_is_insufficient(rag_service: RagService, mock_generator):
    response = rag_service.answer(QUESTION)
    assert response.answer_mode == "insufficient_evidence"
    assert response.answer_status == "insufficient_evidence"
    assert response.retrieval.sufficient_evidence is False
    assert response.sources == []
    assert response.generation.used_llm is False
    assert mock_generator.calls == []


def test_distance_filter_rejects_all_candidates(
    settings: Settings,
    retrieval_service: RetrievalService,
    mock_generator: MockGenerator,
    query_language_detector: QueryLanguageDetector,
    ingestion_service: IngestionService,
):
    ingest(ingestion_service, DOCUMENT_TEXT)
    strict = settings.model_copy(
        update={"rag_distance_filter_enabled": True, "rag_max_distance": 0.0}
    )
    service = RagService(
        settings=strict,
        retrieval_service=retrieval_service,
        generator=mock_generator,
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
    )
    response = service.answer(QUESTION)
    assert response.retrieval.sufficient_evidence is False
    assert response.answer_mode == "irrelevant_retrieval"
    assert response.answer_status == "irrelevant_retrieval"
    assert response.evidence_state == "irrelevant_evidence"
    assert response.sources == []
    assert response.retrieval.rejected_chunks
    assert mock_generator.calls == []


def test_generation_without_evidence_policy_calls_llm(
    settings: Settings,
    retrieval_service: RetrievalService,
    mock_generator: MockGenerator,
    query_language_detector: QueryLanguageDetector,
):
    permissive = settings.model_copy(update={"rag_generation_without_evidence": True})
    service = RagService(
        settings=permissive,
        retrieval_service=retrieval_service,
        generator=mock_generator,
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
    )
    response = service.answer(QUESTION)
    assert response.retrieval.sufficient_evidence is False
    assert response.answer_mode == "insufficient_evidence"
    assert response.answer_status == "insufficient_evidence"
    assert response.generation.used_llm is True
    assert len(mock_generator.calls) == 1
    assert "(no evidence retrieved)" in mock_generator.calls[0][1].content


def test_detected_language_is_reported(rag_service: RagService, ingestion_service):
    ingest(ingestion_service, DOCUMENT_TEXT)
    english = rag_service.answer(QUESTION)
    assert english.detected_language == "en"
    assert english.language_reliable is True
    hindi = rag_service.answer("कंपनी की स्थापना कब हुई?")
    assert hindi.detected_language == "hi"


def test_overlong_question_is_rejected(rag_service: RagService):
    with pytest.raises(QueryValidationError):
        rag_service.answer("a" * 500)


def test_top_k_is_clamped(rag_service: RagService, ingestion_service):
    ingest(ingestion_service, " ".join(f"token{i}" for i in range(300)))
    response = rag_service.answer(QUESTION, top_k=1000)
    assert 1 <= response.retrieval.top_k <= 50
