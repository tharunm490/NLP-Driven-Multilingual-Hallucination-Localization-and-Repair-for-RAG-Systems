"""On-demand claim-level verification for stored RAG answers."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.retrieval import RetrievedChunk
from app.services.generator import MockGenerator
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_service import RagService
from app.services.rag_verification_service import RagVerificationService
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import ClaimVerificationEngine
from app.services.verification.matching import LexicalEvidenceMatcher
from evaluation.iso27001_dataset import CHUNKS_BY_ID
from tests.rag_support import (
    FixedEvidenceAssessor,
    StubRetrievalService,
    assessment,
    chunk_by_id,
)
from tests.verification_support import (
    FakeNerTagger,
    FakeNliModel,
    FailingVerificationEngine,
    StubVerificationEngine,
    supported_result,
)

QUESTION = "What three properties does an ISMS preserve?"
ANSWER = "An ISMS preserves confidentiality, integrity, and availability."
EVIDENCE = CHUNKS_BY_ID["iso-cia"].text


def _rag_service(
    settings: Settings,
    detector: QueryLanguageDetector,
    *,
    evidence_assessor: FixedEvidenceAssessor | None = None,
) -> tuple[RagService, MockGenerator]:
    generator = MockGenerator(answer=ANSWER)
    service = RagService(
        settings=settings.model_copy(update={"rag_max_distance": 0.75}),
        retrieval_service=StubRetrievalService([chunk_by_id("iso-cia", 0.4)]),
        generator=generator,
        language_detector=detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=evidence_assessor or FixedEvidenceAssessor(),
    )
    return service, generator


def _verify(
    settings: Settings,
    detector: QueryLanguageDetector,
    engine,
    *,
    evidence_assessor: FixedEvidenceAssessor | None = None,
):
    service, _ = _rag_service(
        settings,
        detector,
        evidence_assessor=evidence_assessor,
    )
    pending = service.answer(QUESTION)
    assert pending.answer_status in {"verification_pending", "partially_grounded"}
    assert pending.verification.status == "pending"
    verifier = RagVerificationService(
        settings=settings,
        verification_engine=engine,
    )
    return verifier.verify(pending)


def test_rag_answer_starts_unverified(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    service, _ = _rag_service(settings, query_language_detector)

    response = service.answer(QUESTION)

    assert response.answer == ANSWER
    assert response.answer_status == "verification_pending"
    assert response.verification.status == "pending"
    assert response.verification.details is None
    assert response.timings.verification_ms == 0.0


def test_supported_claims_change_status_to_grounded(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = StubVerificationEngine(supported_result(claim_text=ANSWER))

    response = _verify(settings, query_language_detector, engine)

    assert response.answer_status == "grounded"
    assert response.answer_mode == "grounded"
    assert response.verification.claim_level_verification is True
    assert response.verification.status == "verified"
    assert response.verification.supported == 1
    assert response.verification.details is not None
    assert response.timings.verification_ms >= 0.0
    assert len(engine.calls) == 1
    _, evidence = engine.calls[0]
    assert evidence[0].chunk_id == "iso-cia"
    assert evidence[0].text == EVIDENCE
    assert evidence[0].retrieval_score == pytest.approx(0.4)
    assert evidence[0].retrieval_similarity == pytest.approx(0.6)


def test_contradicted_claim_changes_status_to_contradicted(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = StubVerificationEngine(
        supported_result(
            supported=0,
            contradicted=1,
            claim_text=ANSWER,
        )
    )

    response = _verify(settings, query_language_detector, engine)

    assert response.answer_status == "contradicted"
    assert response.verification.contradicted == 1


def test_verification_failure_is_isolated_from_the_answer(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    response = _verify(
        settings, query_language_detector, FailingVerificationEngine()
    )

    assert response.answer == ANSWER
    assert response.answer_status == "verification_pending"
    assert response.verification.status == "error"
    assert response.verification.details is not None
    assert response.diagnostics.verification_error


def test_partial_verification_is_exposed_without_claiming_grounded(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    base = supported_result(claim_text=ANSWER)
    partial = base.model_copy(
        update={
            "status": "partial",
            "summary": base.summary.model_copy(
                update={
                    "claims_total": 2,
                    "claims_checked": 1,
                    "claims_complete": 1,
                    "claims_incomplete": 1,
                    "pairs_expected": 2,
                    "pairs_evaluated": 1,
                    "pairs_missing": 1,
                }
            ),
        }
    )
    engine = StubVerificationEngine(partial)

    response = _verify(settings, query_language_detector, engine)

    assert response.answer_status == "partially_grounded"
    assert response.verification.status == "partial"
    assert response.verification.claims_checked == 1
    assert response.verification.claims_complete == 1
    assert response.verification.claims_incomplete == 1
    assert response.verification.pairs_evaluated == 1
    assert response.verification.pairs_expected == 2
    assert response.verification.verification_error == 0
    assert response.verification.status != "verified"


def test_all_unevaluable_claims_do_not_report_a_grounded_answer(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    base = supported_result(claim_text=ANSWER)
    unevaluable_claim = base.claims[0].model_copy(
        update={
            "claim": base.claims[0].claim.model_copy(
                update={
                    "evaluable": False,
                    "complete": False,
                    "incomplete_reason": (
                        "The claim starts with a pronoun whose referent is "
                        "outside the claim."
                    ),
                }
            ),
            "status": "unevaluable",
            "label": None,
            "confidence": None,
            "probabilities": {},
            "pairs_expected": 0,
            "pairs_evaluated": 0,
            "complete": False,
            "matched_evidence": [],
            "aggregation": "not_evaluable",
            "explanation": "The claim was not independently evaluable.",
        }
    )
    result = base.model_copy(
        update={
            "claims": [unevaluable_claim],
            "summary": base.summary.model_copy(
                update={
                    "claims_total": 1,
                    "claims_evaluable": 0,
                    "claims_unevaluable": 1,
                    "claims_checked": 0,
                    "claims_complete": 0,
                    "claims_incomplete": 0,
                    "supported": 0,
                    "unevaluable": 1,
                    "pairs_expected": 0,
                    "pairs_evaluated": 0,
                }
            ),
        }
    )

    response = _verify(
        settings,
        query_language_detector,
        StubVerificationEngine(result),
    )

    assert response.answer_status == "insufficient_evidence"
    assert response.verification.claims_checked == 0
    assert response.verification.claims_unevaluable == 1
    assert response.verification.unevaluable == 1
    assert "independently evaluable" in response.verification.note


def test_all_supported_claims_ground_a_partial_evidence_answer(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    response = _verify(
        settings,
        query_language_detector,
        StubVerificationEngine(supported_result(claim_text=ANSWER)),
        evidence_assessor=FixedEvidenceAssessor(
            assessment("partial_evidence", missing=("audit schedule",))
        ),
    )

    assert response.evidence_state == "partial_evidence"
    assert response.answer_status == "grounded"
    assert response.verification.status == "verified"


def test_no_supported_claims_on_partial_evidence_is_insufficient(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    response = _verify(
        settings,
        query_language_detector,
        StubVerificationEngine(
            supported_result(supported=0, neutral=1, claim_text=ANSWER)
        ),
        evidence_assessor=FixedEvidenceAssessor(
            assessment("partial_evidence", missing=("audit schedule",))
        ),
    )

    assert response.answer_status == "insufficient_evidence"
    assert response.verification.neutral == 1


def test_verified_result_is_returned_without_rerunning_nli(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    engine = StubVerificationEngine(supported_result(claim_text=ANSWER))
    response = _verify(settings, query_language_detector, engine)

    again = RagVerificationService(
        settings=settings,
        verification_engine=engine,
    ).verify(response)

    assert again == response
    assert len(engine.calls) == 1


def test_real_engine_end_to_end_keeps_summary_and_answer_status_consistent(
    settings: Settings, query_language_detector: QueryLanguageDetector
):
    answer = "Mercury is the closest planet to the Sun."
    chunk = RetrievedChunk(
        chunk_id="planet-facts-1",
        document_id="planet-facts",
        filename="planet-facts.txt",
        chunk_index=0,
        page_number=1,
        language="en",
        text=(
            "Mercury is the closest planet to the Sun. "
            "Venus is the hottest planet in the Solar System."
        ),
        distance=0.2,
        similarity=0.8,
    )
    verification_engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger(),
        matcher=LexicalEvidenceMatcher(),
        nli_model=FakeNliModel(),
    )
    service = RagService(
        settings=settings.model_copy(update={"rag_max_distance": 0.75}),
        retrieval_service=StubRetrievalService([chunk]),
        generator=MockGenerator(answer=answer),
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=FixedEvidenceAssessor(),
    )
    pending = service.answer("Which planet is closest to the Sun?")
    response = RagVerificationService(
        settings=settings,
        verification_engine=verification_engine,
    ).verify(pending)

    assert response.answer_status == "grounded"
    assert response.verification.status == "verified"
    details = response.verification.details
    assert details is not None
    assert details.summary.claims_total == 1
    assert details.summary.claims_complete == 1
    assert details.summary.pairs_expected == 1
    assert details.summary.pairs_evaluated == 1
    assert details.summary.pairs_missing == 0
    assert details.summary.supported == 1
    assert details.summary.contradicted == 0
    assert (
        details.summary.supported
        + details.summary.contradicted
        + details.summary.neutral
        + details.summary.unevaluable
        + details.summary.verification_error
        == details.summary.claims_total
    )
    claim = details.claims[0]
    assert claim.status == "supported"
    assert claim.complete is True
    assert claim.explanation
    assert claim.matched_evidence[0].windows
    assert claim.matched_evidence[0].windows[0].text
    assert claim.matched_evidence[0].premise_text
    assert claim.matched_evidence[0].counts_for_verdict is True
    assert details.timings.claim_extraction_ms >= 0.0
    assert details.timings.evidence_selection_ms >= 0.0
    assert details.timings.nli_ms >= 0.0
    assert details.timings.aggregation_ms >= 0.0
