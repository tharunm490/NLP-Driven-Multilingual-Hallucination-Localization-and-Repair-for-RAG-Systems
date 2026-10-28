"""API tests for POST /api/verification/check."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import (
    get_rag_verification_service,
    get_request_limiter,
    get_verification_engine,
)
from app.core.config import Settings
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_verification_service import RagVerificationService
from app.services.request_limiter import RequestLimiter
from app.services.verification.claims import HeuristicClaimExtractor
from app.services.verification.engine import ClaimVerificationEngine
from tests.verification_support import (
    FakeNerTagger,
    FakeNliModel,
    FixedMatcher,
    StubVerificationEngine,
    fake_entity,
    supported_result,
)

ANSWER = "Acme Corporation was founded in 2015."
EVIDENCE = "Acme Corporation was founded in 2015 in Paris."
DOCUMENT_TEXT = (
    "Acme Corporation is a technology company founded in 2015 in Paris, France."
)
QUESTION = "When was Acme Corporation founded?"


def test_verification_endpoint_returns_claim_details(
    api_client: TestClient,
    settings: Settings,
    query_language_detector: QueryLanguageDetector,
):
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger([fake_entity("Acme Corporation")]),
        matcher=FixedMatcher(),
        nli_model=FakeNliModel(),
    )
    api_client.app.dependency_overrides[get_verification_engine] = lambda: engine

    response = api_client.post(
        "/api/verification/check",
        json={"answer": ANSWER, "evidence": [{"text": EVIDENCE, "filename": "acme.txt"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["summary"]["claims_total"] == 1
    assert body["summary"]["claims_complete"] == 1
    assert body["summary"]["pairs_expected"] == 1
    assert body["summary"]["pairs_evaluated"] == 1
    assert body["summary"]["pairs_missing"] == 0
    assert body["summary"]["supported"] == 1
    assert body["claims"][0]["status"] == "supported"
    assert body["claims"][0]["complete"] is True
    assert body["claims"][0]["pairs_evaluated"] == 1
    assert body["claims"][0]["claim"]["entities"][0]["text"] == "Acme Corporation"
    assert body["claims"][0]["matched_evidence"][0]["evidence"]["filename"] == "acme.txt"
    assert body["claims"][0]["matched_evidence"][0]["premise_text"] == EVIDENCE
    assert body["claims"][0]["matched_evidence"][0]["counts_for_verdict"] is True
    assert body["nli_model"] == "test/fake-nli"
    assert body["timings"]["claim_extraction_ms"] >= 0.0
    assert body["timings"]["evidence_selection_ms"] >= 0.0
    assert body["timings"]["nli_ms"] >= 0.0
    assert body["timings"]["aggregation_ms"] >= 0.0


def test_verification_endpoint_returns_all_claim_evidence_pairs(
    api_client: TestClient,
    settings: Settings,
    query_language_detector: QueryLanguageDetector,
):
    answer = (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2020."
    )
    evidence = [
        {"text": "Acme Corporation was founded in 2015.", "filename": "acme.txt"},
        {"text": "Beta Corporation was founded in 2020.", "filename": "beta.txt"},
    ]
    engine = ClaimVerificationEngine(
        settings=settings,
        detector=query_language_detector,
        claim_extractor=HeuristicClaimExtractor(),
        ner_tagger=FakeNerTagger(),
        matcher=FixedMatcher([0.9, 0.8]),
        nli_model=FakeNliModel(),
    )
    api_client.app.dependency_overrides[get_verification_engine] = lambda: engine

    response = api_client.post(
        "/api/verification/check",
        json={"answer": answer, "evidence": evidence},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["claims_total"] == 2
    assert body["summary"]["pairs_expected"] == 4
    assert body["summary"]["pairs_evaluated"] == 4
    assert body["summary"]["claims_complete"] == 2
    for claim in body["claims"]:
        assert claim["pairs_expected"] == 2
        assert claim["pairs_evaluated"] == 2
        assert len(claim["matched_evidence"]) == 2
        assert [row["rank"] for row in claim["matched_evidence"]] == [1, 2]
        assert all(row["evaluated"] for row in claim["matched_evidence"])
        assert all(row["premise_text"] for row in claim["matched_evidence"])


def test_verification_endpoint_rejects_blank_input(api_client: TestClient):
    assert (
        api_client.post(
            "/api/verification/check", json={"answer": " ", "evidence": []}
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/api/verification/check",
            json={"answer": "A factual answer.", "evidence": [{"text": "  "}]},
        ).status_code
        == 422
    )


def test_verification_endpoint_returns_503_when_busy(
    api_client: TestClient,
):
    limiter = RequestLimiter(max_concurrent=1, wait_seconds=0.0)
    limiter.acquire()
    api_client.app.dependency_overrides[get_request_limiter] = lambda: limiter
    try:
        response = api_client.post(
            "/api/verification/check",
            json={
                "answer": ANSWER,
                "evidence": [{"text": EVIDENCE, "filename": "acme.txt"}],
            },
        )
    finally:
        limiter.release()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert "busy" in response.json()["detail"].lower()


def _ask(api_client: TestClient) -> dict:
    upload = api_client.post(
        "/api/documents/upload",
        files={"files": ("acme.txt", DOCUMENT_TEXT.encode("utf-8"), "text/plain")},
    )
    assert upload.json()["results"][0]["status"] == "completed"
    response = api_client.post("/api/rag/query", json={"question": QUESTION})
    assert response.status_code == 200
    return response.json()


def test_result_verification_reuses_stored_answer_and_evidence(
    api_client: TestClient,
    settings: Settings,
):
    pending = _ask(api_client)
    assert pending["verification"]["status"] == "pending"
    engine = StubVerificationEngine(
        supported_result(claim_text="Acme Corporation was founded in 2015.")
    )
    api_client.app.dependency_overrides[get_rag_verification_service] = (
        lambda: RagVerificationService(
            settings=settings,
            verification_engine=engine,
        )
    )

    response = api_client.post(
        f"/api/verification/results/{pending['result_id']}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_id"] == pending["result_id"]
    assert body["question"] == pending["question"]
    assert body["answer"] == pending["answer"]
    assert body["sources"] == pending["sources"]
    assert body["answer_status"] == "grounded"
    assert body["verification"]["status"] == "verified"
    assert body["verification"]["details"]["status"] == "completed"
    assert body["verification"]["supported"] == 1
    assert body["timings"]["verification_ms"] >= 0.0
    assert len(engine.calls) == 1
    answer, evidence = engine.calls[0]
    assert answer == pending["answer"]
    assert [item.chunk_id for item in evidence] == [
        source["chunk_id"] for source in pending["sources"]
    ]

    shared = api_client.get(f"/api/rag/results/{pending['result_id']}")
    assert shared.status_code == 200
    assert shared.json()["verification"]["status"] == "verified"


def test_result_verification_is_idempotent(
    api_client: TestClient,
    settings: Settings,
):
    pending = _ask(api_client)
    engine = StubVerificationEngine()
    api_client.app.dependency_overrides[get_rag_verification_service] = (
        lambda: RagVerificationService(
            settings=settings,
            verification_engine=engine,
        )
    )

    first = api_client.post(f"/api/verification/results/{pending['result_id']}")
    second = api_client.post(f"/api/verification/results/{pending['result_id']}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["verification"]["status"] == "verified"
    assert second.json()["verification"] == first.json()["verification"]
    assert len(engine.calls) == 1


def test_result_verification_rejects_unknown_result(api_client: TestClient):
    response = api_client.post("/api/verification/results/missing-result")

    assert response.status_code == 404
    assert "no longer available" in response.json()["detail"].lower()
