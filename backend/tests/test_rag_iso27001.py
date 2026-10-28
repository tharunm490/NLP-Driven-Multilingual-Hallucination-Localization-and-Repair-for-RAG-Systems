"""End-to-end RAG behaviour on an ISO/IEC 27001-derived fixture.

These tests deliberately use a fixed candidate list, a fixed evidence decision,
and a mock generator. They verify the orchestration boundary: only filtered
chunks reach either the evidence judge or answer generation, and the response
status is honest about what was actually verified.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from app.core.config import Settings
from app.services.generator import MockGenerator
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import (
    LanguageScore,
    QueryLanguageDetector,
)
from app.services.rag_service import RagService
from tests.rag_support import (
    FixedEvidenceAssessor,
    StubRetrievalService,
    assessment,
    chunk_by_id,
)

CIA_QUESTION = "What three information security properties does an ISMS preserve?"
FINE_QUESTION = "What is the exact fine for violating ISO/IEC 27001?"
PERSONAL_QUESTION = "What does this document say about Tharun M's personal behavior?"
CLAUSE_QUESTION = "What does Clause 6.1.2 address?"
AUDIT_PREMISE_QUESTION = "Does Clause 6.1.2 cover internal audits?"
PARTIAL_QUESTION = (
    "What does Clause 6.1.2 require for risk assessment and what is the exact "
    "audit frequency?"
)
HINDI_CIA_QUESTION = "सूचना सुरक्षा के तीन मुख्य गुण कौन-से हैं?"


class _UnicodeLanguageBackend:
    name = "unicode-test"

    def is_available(self) -> bool:
        return True

    def detect(self, text: str) -> list[LanguageScore]:
        if any("\u0900" <= char <= "\u097f" for char in text):
            return [LanguageScore("hi", 0.98)]
        return [LanguageScore("en", 0.95)]


def _strict(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"rag_distance_filter_enabled": True, "rag_max_distance": 0.55}
    )


def _service(
    settings: Settings,
    results: Sequence,
    assessor: FixedEvidenceAssessor,
    *,
    answer: str | None = None,
) -> tuple[RagService, MockGenerator]:
    generator = MockGenerator(answer=answer)
    detector = QueryLanguageDetector(
        enabled=True,
        backend=_UnicodeLanguageBackend(),
        fallback_backend=_UnicodeLanguageBackend(),
    )
    service = RagService(
        settings=settings,
        retrieval_service=StubRetrievalService(results),
        generator=generator,
        language_detector=detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=assessor,
    )
    return service, generator


def test_a_answerable_cia_question_uses_only_relevant_evidence(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment(
            "sufficient_evidence",
            supported=["confidentiality, integrity, availability"],
        )
    )
    service, generator = _service(
        _strict(settings),
        [
            chunk_by_id("iso-cia", 0.43),
            chunk_by_id("iso-scope", 0.50),
            chunk_by_id("iso-certification", 0.62),
        ],
        assessor,
        answer="An ISMS preserves confidentiality, integrity, and availability [Source 1].",
    )

    response = service.answer(CIA_QUESTION)

    assert response.answer_status == "verification_pending"
    assert response.evidence_state == "sufficient_evidence"
    assert response.retrieval.sufficient_evidence is True
    assert [source.chunk_id for source in response.sources] == ["iso-cia", "iso-scope"]
    assert "iso-certification" in response.diagnostics.rejected_chunk_ids
    assert "confidentiality" in response.answer.lower()

    assert len(assessor.calls) == 1
    assert {chunk.chunk_id for chunk in assessor.calls[0][1]} == {"iso-cia", "iso-scope"}

    prompt = generator.calls[0][1].content
    assert "iso-cia" in prompt
    assert "iso-certification" not in prompt


def test_b_missing_exact_fine_causes_abstention_not_invention(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment(
            "insufficient_evidence",
            reason="The evidence mentions certification but contains no fine amount.",
            missing=["exact fine amount"],
        )
    )
    service, generator = _service(
        _strict(settings),
        [
            chunk_by_id("iso-certification", 0.45),
            chunk_by_id("iso-scope", 0.54),
        ],
        assessor,
    )

    response = service.answer(FINE_QUESTION)

    assert response.answer_status == "insufficient_evidence"
    assert response.evidence_state == "insufficient_evidence"
    assert response.retrieval.sufficient_evidence is False
    assert response.generation.used_llm is False
    assert generator.calls == []
    assert "do not contain sufficient relevant information" in response.answer
    assert "$" not in response.answer
    assert "fine" not in response.answer.lower()
    assert response.verification.claim_level_verification is False
    assert "exact fine amount" in response.assessment.missing_aspects


def test_c_unrelated_personal_question_never_exposes_iso_chunks(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment(
            "irrelevant_evidence",
            reason="The passages are about information security, not a person's behaviour.",
        )
    )
    service, generator = _service(
        _strict(settings),
        [
            chunk_by_id("iso-policy", 0.44),
            chunk_by_id("iso-documented", 0.48),
        ],
        assessor,
    )

    response = service.answer(PERSONAL_QUESTION)

    assert response.answer_status == "irrelevant_retrieval"
    assert response.evidence_state == "irrelevant_evidence"
    assert response.sources == []
    assert response.answer_mode != "grounded"
    assert response.generation.used_llm is False
    assert generator.calls == []
    assert "Tharun" not in response.answer
    assert "No relevant supporting passage" in response.answer


def test_d_clause_question_uses_risk_evidence_and_preserves_source_metadata(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment("sufficient_evidence", supported=["risk assessment process"])
    )
    service, generator = _service(
        _strict(settings),
        [
            chunk_by_id("iso-risk", 0.52),
            chunk_by_id("iso-audit", 0.54),
        ],
        assessor,
        answer="Clause 6.1.2 addresses the risk assessment process [Source 1].",
    )

    response = service.answer(CLAUSE_QUESTION)

    assert response.answer_status == "verification_pending"
    assert response.evidence_state == "sufficient_evidence"
    assert [source.chunk_id for source in response.sources] == ["iso-risk", "iso-audit"]
    source = response.sources[0]
    assert source.filename == "iso27001.pdf"
    assert source.page_number == 8
    assert source.retrieval_score <= 0.55
    assert source.similarity == pytest.approx(1.0 - source.retrieval_score)
    assert source.passed_relevance_filter is True
    assert source.relevance_reason == "distance_within_threshold"
    assert "Clause 6.1.2" in generator.calls[0][1].content


def test_e_premise_question_does_not_become_blind_yes(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment(
            "partial_evidence",
            reason="The evidence explains risk assessment but says nothing about audits.",
            supported=["risk assessment process"],
            missing=["whether Clause 6.1.2 covers internal audits"],
        )
    )
    service, generator = _service(
        _strict(settings),
        [chunk_by_id("iso-risk", 0.53)],
        assessor,
        answer=(
            "The evidence describes risk assessment under Clause 6.1.2, but it does "
            "not establish that the clause covers internal audits [Source 1]."
        ),
    )

    response = service.answer(AUDIT_PREMISE_QUESTION)

    assert response.answer_status == "partially_grounded"
    assert response.evidence_state == "partial_evidence"
    assert response.answer_mode != "grounded"
    assert response.assessment.premise_contradicted is False
    assert any(
        "internal audits" in item for item in response.assessment.missing_aspects
    )
    assert response.generation.used_llm is True
    prompt = generator.calls[0][1].content
    assert "PARTIAL" in prompt
    assert "do not fill the missing parts" in prompt.lower()


def test_f_hindi_question_can_use_english_evidence(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment("sufficient_evidence", supported=["confidentiality, integrity, availability"])
    )
    service, generator = _service(
        _strict(settings),
        [chunk_by_id("iso-cia", 0.45)],
        assessor,
        answer="आईएसएमएस गोपनीयता, अखंडता और उपलब्धता बनाए रखता है [Source 1].",
    )

    response = service.answer(HINDI_CIA_QUESTION)

    assert response.detected_language == "hi"
    assert response.answer_status == "verification_pending"
    assert [source.chunk_id for source in response.sources] == ["iso-cia"]
    assert response.sources[0].language == "en"
    assert assessor.calls[0][0] == HINDI_CIA_QUESTION
    assert HINDI_CIA_QUESTION in generator.calls[0][1].content
    assert "गोपनीयता" in response.answer


def test_g_all_candidates_above_threshold_returns_no_evidence(settings: Settings):
    assessor = FixedEvidenceAssessor(assessment("sufficient_evidence"))
    service, generator = _service(
        _strict(settings),
        [chunk_by_id("iso-certification", 0.90), chunk_by_id("iso-scope", 1.10)],
        assessor,
    )

    response = service.answer(FINE_QUESTION)

    assert response.answer_status == "irrelevant_retrieval"
    assert response.evidence_state == "irrelevant_evidence"
    assert response.retrieval.candidate_chunks == 2
    assert response.retrieval.filtered_chunks == 0
    assert response.retrieval.rejected_chunks
    assert {item.reason for item in response.retrieval.rejected_chunks} == {
        "distance_above_threshold"
    }
    assert response.sources == []
    assert response.generation.used_llm is False
    assert assessor.calls == []
    assert generator.calls == []


def test_h_partial_answer_reports_supported_and_missing_parts(settings: Settings):
    assessor = FixedEvidenceAssessor(
        assessment(
            "partial_evidence",
            supported=["risk assessment requirements"],
            missing=["exact audit frequency"],
        )
    )
    service, generator = _service(
        _strict(settings),
        [chunk_by_id("iso-risk", 0.52), chunk_by_id("iso-audit", 0.54)],
        assessor,
        answer=(
            "The documents describe risk assessment requirements [Source 1], but they "
            "do not state an exact audit frequency."
        ),
    )

    response = service.answer(PARTIAL_QUESTION)

    assert response.answer_status == "partially_grounded"
    assert response.evidence_state == "partial_evidence"
    assert response.retrieval.sufficient_evidence is False
    assert response.assessment.supported_aspects == ["risk assessment requirements"]
    assert response.assessment.missing_aspects == ["exact audit frequency"]
    assert len(response.sources) == 2
    assert response.generation.used_llm is True
    prompt = generator.calls[0][1].content
    assert "risk assessment requirements" in prompt
    assert "exact audit frequency" in prompt
    assert "PARTIAL" in prompt
