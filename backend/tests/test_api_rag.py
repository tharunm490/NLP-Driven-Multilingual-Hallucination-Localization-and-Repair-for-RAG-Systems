"""API tests for POST /api/rag/query."""

from __future__ import annotations

from typing import Sequence

from fastapi.testclient import TestClient

from app.api.deps import get_rag_service, get_rag_verification_service
from app.core.config import Settings
from app.core.errors import GenerationError
from tests.rag_support import FixedEvidenceAssessor

from app.services.generator import (
    ChatMessage,
    GenerationResult,
    MockGenerator,
    UnconfiguredGenerator,
)
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_service import RagService
from app.services.rag_verification_service import RagVerificationService
from app.services.retrieval_service import RetrievalService
from tests.verification_support import StubVerificationEngine

DOCUMENT_TEXT = (
    "Acme Corporation is a technology company founded in 2015 in Paris, France."
)
QUESTION = "When was Acme Corporation founded?"


class FailingGenerator:
    provider = "stub"
    model = "stub"

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        raise GenerationError("simulated provider outage")


def upload(client: TestClient, text: str, name: str = "acme.txt") -> dict:
    response = client.post(
        "/api/documents/upload",
        files={"files": (name, text.encode("utf-8"), "text/plain")},
    )
    result = response.json()["results"][0]
    assert result["status"] == "completed", result
    return result


def test_rag_query_happy_path(api_client: TestClient):
    document = upload(api_client, DOCUMENT_TEXT)
    response = api_client.post("/api/rag/query", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()

    assert body["question"] == QUESTION
    assert body["detected_language"] == "en"
    assert body["answer_mode"] == "verification_pending"
    assert body["answer_status"] == "verification_pending"
    assert body["evidence_state"] == "sufficient_evidence"
    assert body["verification"]["claim_level_verification"] is False
    assert body["verification"]["status"] == "pending"
    assert body["verification"]["details"] is None
    assert body["timings"]["verification_ms"] == 0.0
    assert body["answer"]
    assert body["sources"]
    source = body["sources"][0]
    assert source["document_id"] == document["document_id"]
    assert source["filename"] == "acme.txt"
    assert source["chunk_id"]
    assert "retrieval_score" in source
    assert "similarity" in source
    assert body["retrieval"]["returned_chunks"] == len(body["sources"])
    assert body["retrieval"]["sufficient_evidence"] is True
    assert body["generation"]["used_llm"] is True
    assert body["generation"]["provider"] == "mock"
    assert body["timings"]["total_ms"] >= 0
    assert body["result_id"]

    shared = api_client.get(f"/api/rag/results/{body['result_id']}")
    assert shared.status_code == 200
    assert shared.json()["answer"] == body["answer"]
    assert shared.json()["sources"] == body["sources"]
    assert shared.json()["question"] == QUESTION


def test_rag_query_never_runs_claim_verification(
    api_client: TestClient,
    settings: Settings,
):
    upload(api_client, DOCUMENT_TEXT)
    engine = StubVerificationEngine()
    api_client.app.dependency_overrides[get_rag_verification_service] = (
        lambda: RagVerificationService(
            settings=settings,
            verification_engine=engine,
        )
    )

    response = api_client.post("/api/rag/query", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["verification"]["status"] == "pending"
    assert engine.calls == []


def test_rag_shared_result_is_not_found_after_restart(api_client: TestClient):
    response = api_client.get("/api/rag/results/missing-result")
    assert response.status_code == 404
    assert "no longer available" in response.json()["detail"].lower()


def test_rag_empty_store_returns_insufficient_evidence(api_client: TestClient):
    response = api_client.post("/api/rag/query", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "insufficient_evidence"
    assert body["retrieval"]["sufficient_evidence"] is False
    assert body["sources"] == []
    assert body["generation"]["used_llm"] is False


def test_rag_rejects_blank_question(api_client: TestClient):
    assert api_client.post("/api/rag/query", json={"question": ""}).status_code == 422
    assert api_client.post("/api/rag/query", json={"question": "   "}).status_code == 422


def test_rag_rejects_top_k_out_of_range(api_client: TestClient):
    response = api_client.post("/api/rag/query", json={"question": "hi", "top_k": 999})
    assert response.status_code == 422


def test_rag_rejects_overlong_question(api_client: TestClient):
    response = api_client.post("/api/rag/query", json={"question": "a" * 500})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"].lower()


def test_rag_returns_503_when_llm_unconfigured(
    api_client: TestClient,
    settings: Settings,
    retrieval_service: RetrievalService,
    query_language_detector: QueryLanguageDetector,
):
    upload(api_client, DOCUMENT_TEXT)
    service = RagService(
        settings=settings,
        retrieval_service=retrieval_service,
        generator=UnconfiguredGenerator("No LLM provider is configured."),
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=FixedEvidenceAssessor(),
    )
    api_client.app.dependency_overrides[get_rag_service] = lambda: service

    response = api_client.post("/api/rag/query", json={"question": QUESTION})
    assert response.status_code == 503
    assert "LLM" in response.json()["detail"]


def test_rag_returns_502_on_provider_error(
    api_client: TestClient,
    settings: Settings,
    retrieval_service: RetrievalService,
    query_language_detector: QueryLanguageDetector,
):
    upload(api_client, DOCUMENT_TEXT)
    service = RagService(
        settings=settings,
        retrieval_service=retrieval_service,
        generator=FailingGenerator(),
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=FixedEvidenceAssessor(),
    )
    api_client.app.dependency_overrides[get_rag_service] = lambda: service

    response = api_client.post("/api/rag/query", json={"question": QUESTION})
    assert response.status_code == 502
    assert "outage" in response.json()["detail"]


def test_rag_filters_by_document(
    api_client: TestClient,
    settings: Settings,
    retrieval_service: RetrievalService,
    query_language_detector: QueryLanguageDetector,
):
    first = upload(api_client, DOCUMENT_TEXT, "acme.txt")
    upload(api_client, "Completely unrelated document about gardening tools.", "garden.txt")
    service = RagService(
        settings=settings.model_copy(update={"rag_distance_filter_enabled": False}),
        retrieval_service=retrieval_service,
        generator=MockGenerator(),
        language_detector=query_language_detector,
        prompt_builder=PromptBuilder(),
        evidence_assessor=FixedEvidenceAssessor(),
    )
    api_client.app.dependency_overrides[get_rag_service] = lambda: service

    response = api_client.post(
        "/api/rag/query",
        json={"question": QUESTION, "document_ids": [first["document_id"]]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"]
    assert all(source["document_id"] == first["document_id"] for source in body["sources"])
