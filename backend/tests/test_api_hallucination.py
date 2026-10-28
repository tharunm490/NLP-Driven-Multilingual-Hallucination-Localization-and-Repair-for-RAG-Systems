"""API tests for POST /api/hallucination/check."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import get_hallucination_check_service
from app.core.config import Settings
from app.core.errors import HallucinationCheckError
from app.services.hallucination_service import HallucinationCheckService
from app.services.rag_result_store import RagResultStore
from tests.hallucination_support import (
    FakeHallucinationDetector,
    nli_result,
    rag_response,
    source,
)

ANSWER = (
    "Acme Corporation was founded in 2015. "
    "Beta Corporation was founded in 2020."
)
ACME = "Acme Corporation was founded in 2015."
BETA = "Beta Corporation was founded in 2020."
EVIDENCE = [
    source("e1", "Acme Corporation was founded in 2015 in Paris."),
    source("e2", "Beta Corporation was founded in 2020 in Lyon.", page_number=8),
]


def install_service(
    api_client: TestClient,
    settings: Settings,
    detector: FakeHallucinationDetector,
) -> HallucinationCheckService:
    service = HallucinationCheckService(settings, detector)
    api_client.app.dependency_overrides[get_hallucination_check_service] = (
        lambda: service
    )
    return service


def test_hallucination_check_reuses_result_and_checks_every_claim(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    detector = FakeHallucinationDetector(
        default=0.8,
        by_claim={ACME: 0.93, BETA: 0.12},
    )
    install_service(api_client, settings, detector)
    result = rag_response(
        ANSWER,
        EVIDENCE,
        verification=nli_result(
            ANSWER,
            [(ACME, "supported", 0.91), (BETA, "supported", 0.88)],
        ),
    )
    rag_result_store.put(result)

    response = api_client.post(
        "/api/hallucination/check",
        json={"result_id": result.result_id},
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["result_id"] == result.result_id
    assert body["question"] == result.question
    assert body["model_name"] == "test/hallucination-guard"
    assert body["model_input_order"] == "source_then_claim"
    assert body["summary"] == {
        "claims_total": 2,
        "supported": 1,
        "potentially_hallucinated": 1,
        "needs_review": 0,
        "nli_disagreements": 1,
        "evidence_total": 2,
        "pairs_evaluated": 4,
    }
    assert len(detector.calls) == 1
    assert detector.calls[0][0] == (
        EVIDENCE[0].text,
        ACME,
    )
    by_claim = {item["claim"]: item for item in body["claims"]}
    assert by_claim[ACME]["verdict"] == "supported"
    assert by_claim[ACME]["raw_model_label"] == "SUPPORTED"
    assert by_claim[ACME]["support_probability"] == 0.93
    assert by_claim[ACME]["evidence"][0]["page_number"] == 7
    assert by_claim[BETA]["verdict"] == "potentially_hallucinated"
    assert by_claim[BETA]["raw_model_label"] == "UNSUPPORTED"
    assert by_claim[BETA]["support_probability"] == 0.12
    assert by_claim[BETA]["nli_status"] == "supported"
    assert by_claim[BETA]["nli_disagrees"] is True
    assert ANSWER[by_claim[ACME]["claim_start"] : by_claim[ACME]["claim_end"]] == ACME


def test_hallucination_check_is_idempotent_for_same_result(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    detector = FakeHallucinationDetector(default=0.8)
    install_service(api_client, settings, detector)
    result = rag_response(ANSWER, EVIDENCE)
    rag_result_store.put(result)

    first = api_client.post(
        "/api/hallucination/check", json={"result_id": result.result_id}
    )
    second = api_client.post(
        "/api/hallucination/check", json={"result_id": result.result_id}
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["check_id"] == first.json()["check_id"]
    assert len(detector.calls) == 1


def test_hallucination_check_rejects_missing_shared_result(
    api_client: TestClient,
):
    response = api_client.post(
        "/api/hallucination/check", json={"result_id": "missing"}
    )
    assert response.status_code == 404
    assert "no longer available" in response.json()["detail"].lower()


def test_hallucination_check_requires_generated_answer(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    install_service(api_client, settings, FakeHallucinationDetector())
    result = rag_response(ANSWER, EVIDENCE, used_llm=False)
    rag_result_store.put(result)

    response = api_client.post(
        "/api/hallucination/check", json={"result_id": result.result_id}
    )
    assert response.status_code == 422
    assert "no generated answer" in response.json()["detail"].lower()


def test_hallucination_model_failure_is_not_a_verdict(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    install_service(
        api_client,
        settings,
        FakeHallucinationDetector(fail=True),
    )
    result = rag_response(ANSWER, EVIDENCE)
    rag_result_store.put(result)

    response = api_client.post(
        "/api/hallucination/check", json={"result_id": result.result_id}
    )
    assert response.status_code == 503
    assert "simulated local model failure" in response.json()["detail"]
    assert "verdict" not in response.json()


def test_hallucination_check_can_be_disabled(settings: Settings):
    disabled = settings.model_copy(update={"hallucination_enabled": False})
    service = HallucinationCheckService(
        disabled,
        FakeHallucinationDetector(),
    )
    result = rag_response(ANSWER, EVIDENCE)

    try:
        service.check(result)
    except HallucinationCheckError as exc:
        assert "disabled" in str(exc).lower()
    else:  # pragma: no cover - explicit failure if disabled checks run
        raise AssertionError("disabled hallucination check unexpectedly ran")


def test_hallucination_near_threshold_is_needs_review(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    detector = FakeHallucinationDetector(default=0.5)
    install_service(api_client, settings, detector)
    result = rag_response(ANSWER, EVIDENCE)
    rag_result_store.put(result)

    response = api_client.post(
        "/api/hallucination/check", json={"result_id": result.result_id}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["needs_review"] == 2
    assert all(
        claim["verdict"] == "needs_review" for claim in body["claims"]
    )
