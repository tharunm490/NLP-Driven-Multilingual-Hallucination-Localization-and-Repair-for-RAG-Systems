"""Tests for evidence-grounded iterative hallucination repair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_hallucination_repair_service
from app.core.config import Settings
from app.core.errors import RepairError
from app.schemas.hallucination import (
    HallucinationCheckResponse,
    HallucinationClaimResult,
    HallucinationEvidenceResult,
    HallucinationSummary,
    HallucinationTimings,
)
from app.schemas.rag import RagQueryResponse, RagSource
from app.schemas.verification import (
    VerificationEvidenceInput,
    VerificationResult,
)
from app.services.generator import GenerationResult
from app.services.hallucination_repair_service import (
    build_hallucination_repair_service,
)
from app.services.hallucination_service import HallucinationCheckService
from app.services.rag_result_store import RagResultStore
from app.services.rag_verification_service import RagVerificationService
from tests.hallucination_support import (
    FakeHallucinationDetector,
    nli_result,
    rag_response,
    source,
)
from tests.verification_support import StubVerificationEngine

ANSWER = (
    "Acme Corporation was founded in 2015. "
    "Beta Corporation was founded in 2020."
)
ACME = "Acme Corporation was founded in 2015."
BETA = "Beta Corporation was founded in 2020."
ACME_EVIDENCE = source("e1", "Acme Corporation was founded in 2015 in Paris.")
BETA_EVIDENCE = source(
    "e2", "Beta Corporation was founded in 2020 in Lyon.", page_number=8
)
EVIDENCE = [ACME_EVIDENCE, BETA_EVIDENCE]


@dataclass
class ScriptedGenerator:
    responses: list[str]
    provider: str = "test"
    model: str = "test-repair-model"
    calls: list[list[object]] = field(default_factory=list)

    def generate(self, messages: Sequence[object]) -> GenerationResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return GenerationResult(
            text=self.responses[index],
            provider=self.provider,
            model=self.model,
        )


class CandidateNliEngine:
    enabled = True

    def __init__(self, supported_marker: str = "2021") -> None:
        self._supported_marker = supported_marker
        self.calls: list[tuple[str, list[VerificationEvidenceInput]]] = []

    def verify(
        self,
        answer: str,
        evidence: Sequence[VerificationEvidenceInput],
        *,
        deadline: float | None = None,
        cancel_event: object | None = None,
    ) -> VerificationResult:
        self.calls.append((answer, list(evidence)))
        status = (
            "supported"
            if self._supported_marker in answer
            else "neutral"
        )
        return nli_result(answer, [(answer, status, 0.9)])


def repair_json(
    status: str,
    original: str,
    *,
    repaired: str | None = None,
    quote: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "status": status,
        "original_claim": original,
        "reason": "test",
        "supporting_evidence": [],
    }
    if repaired is not None:
        payload["repaired_claim"] = repaired
    if quote is not None:
        payload["supporting_evidence"] = [
            {"source": "iso-27001.txt", "page": 7, "evidence_quote": quote}
        ]
    return json.dumps(payload)


def original_result(
    answer: str = ANSWER,
    sources: Sequence[RagSource] = EVIDENCE,
    *,
    result_id: str = "result-1",
) -> RagQueryResponse:
    statuses = [(ACME, "supported", 0.9), (BETA, "contradicted", 0.9)]
    return rag_response(
        answer,
        sources,
        verification=nli_result(answer, statuses),
        result_id=result_id,
    )


def build_repair_service(
    settings: Settings,
    *,
    result: RagQueryResponse,
    detector: FakeHallucinationDetector,
    generator: ScriptedGenerator,
    nli_engine: CandidateNliEngine | None = None,
    check_service: HallucinationCheckService | None = None,
) -> tuple[object, FakeHallucinationDetector]:
    check = check_service or HallucinationCheckService(settings, detector)
    service = build_hallucination_repair_service(
        settings=settings,
        verification_engine=nli_engine or CandidateNliEngine(),
        rag_verification_service=RagVerificationService(
            settings=settings,
            verification_engine=StubVerificationEngine(result.verification.details),
        ),
        hallucination_service=check,
        detector=detector,
        generator=generator,
    )
    return service, detector


def test_repair_retries_until_both_systems_support_the_candidate(
    settings: Settings,
):
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2022.",
                quote=BETA_EVIDENCE.text,
            ),
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote=BETA_EVIDENCE.text,
            ),
        ]
    )
    nli_engine = CandidateNliEngine(supported_marker="2021")
    service, _ = build_repair_service(
        settings,
        result=original_result(),
        detector=detector,
        generator=generator,
        nli_engine=nli_engine,
    )

    repair = service.run(original_result())

    assert repair.status == "FULLY_VERIFIED"
    beta_claim = next(claim for claim in repair.claims if claim.original_claim == BETA)
    assert beta_claim.status == "VERIFIED"
    assert [attempt.status for attempt in beta_claim.attempts] == [
        "RETRY_REQUIRED",
        "VERIFIED",
    ]
    assert beta_claim.attempts[0].verification is not None
    assert beta_claim.attempts[0].verification.nli_status == "neutral"
    assert beta_claim.attempts[1].verification is not None
    assert beta_claim.attempts[1].verification.both_supported is True
    assert len(generator.calls) == 2
    assert "PREVIOUS ATTEMPT" in generator.calls[1][1].content
    assert repair.final_answer == (
        "Acme Corporation was founded in 2015. "
        "Beta Corporation was founded in 2021."
    )
    assert [item.verdict for item in repair.final_claims] == [
        "verified",
        "verified",
    ]
    assert repair.omissions == []


def test_repair_replaces_only_a_verified_claim_and_preserves_supported_claims(
    settings: Settings,
):
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            repair_json(
                "QUALIFIED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote=BETA_EVIDENCE.text,
            )
        ]
    )
    service, _ = build_repair_service(
        settings,
        result=original_result(),
        detector=detector,
        generator=generator,
    )

    repair = service.run(original_result())

    assert repair.final_answer.startswith(ACME)
    assert BETA not in repair.final_answer
    assert "Beta Corporation was founded in 2021." in repair.final_answer
    assert repair.claims[0].claim_id != repair.claims[1].claim_id
    assert repair.claims[0].status == "ORIGINAL"
    assert repair.claims[0].included_in_final is True
    assert repair.claims[1].final_decision == "QUALIFIED"
    assert repair.status == "FULLY_VERIFIED"


def test_guard_supported_nli_disagreement_is_not_reported_as_no_repair_needed(
    settings: Settings,
):
    answer = "Acme Corporation was founded in 2015."
    result = rag_response(
        answer,
        [ACME_EVIDENCE],
        verification=nli_result(answer, [(answer, "neutral", 0.6)]),
        result_id="result-disagreement",
    )
    service, _ = build_repair_service(
        settings,
        result=result,
        detector=FakeHallucinationDetector(default=0.9),
        generator=ScriptedGenerator([]),
    )

    repair = service.run(result)

    assert repair.status == "PARTIALLY_VERIFIED"
    assert repair.overview.nli_disagreements == 1
    assert repair.final_claims[0].verdict == "unresolved"


def test_repair_keeps_a_removed_claim_visible_in_the_answer(settings: Settings):
    answer = (
        "- Acme Corporation was founded in 2015.\n"
        "- Beta Corporation was founded in 2020.\n"
        "- Acme Corporation is based in Paris."
    )
    acme_two = "Acme Corporation is based in Paris."
    sources = [
        ACME_EVIDENCE,
        BETA_EVIDENCE,
        source("e3", "Acme Corporation is based in Paris."),
    ]
    result = rag_response(
        answer,
        sources,
        verification=nli_result(
            answer,
            [
                (ACME, "supported", 0.9),
                (BETA, "contradicted", 0.9),
                (acme_two, "supported", 0.9),
            ],
        ),
    )
    detector = FakeHallucinationDetector(default=0.9, by_claim={BETA: 0.1})
    generator = ScriptedGenerator(
        [repair_json("REMOVED", BETA)]
    )
    service, _ = build_repair_service(
        settings,
        result=result,
        detector=detector,
        generator=generator,
    )

    repair = service.run(result)

    assert repair.final_answer == (
        "- Acme Corporation was founded in 2015.\n"
        "- [Unresolved] Beta Corporation was founded in 2020.\n"
        "- Acme Corporation is based in Paris."
    )
    beta_claim = next(claim for claim in repair.claims if claim.original_claim == BETA)
    assert beta_claim.status == "REMOVED"
    assert beta_claim.included_in_final is True
    assert len(repair.omissions) == 1
    assert repair.status == "PARTIALLY_VERIFIED"


def test_repair_without_evidence_keeps_the_claim_visible(settings: Settings):
    result = original_result()
    detector = FakeHallucinationDetector(default=0.9, by_claim={BETA: 0.1})
    check = HallucinationCheckResponse(
        check_id="check-1",
        result_id="result-1",
        question=result.question,
        model_name="test/hallucination-guard",
        support_threshold=0.5,
        review_margin=0.1,
        claims=[
            HallucinationClaimResult(
                claim_id="hc1",
                sentence_index=1,
                sentence=ACME,
                claim=ACME,
                claim_start=0,
                claim_end=len(ACME),
                verdict="supported",
                raw_model_label="SUPPORTED",
                support_probability=0.9,
                explanation="test",
                evidence=[hallu_evidence(ACME_EVIDENCE, 0.9)],
            ),
            HallucinationClaimResult(
                claim_id="hc2",
                sentence_index=2,
                sentence=BETA,
                claim=BETA,
                claim_start=ANSWER.index(BETA),
                claim_end=ANSWER.index(BETA) + len(BETA),
                verdict="potentially_hallucinated",
                raw_model_label="UNSUPPORTED",
                support_probability=0.1,
                explanation="test",
                evidence=[],
            ),
        ],
        summary=HallucinationSummary(
            claims_total=2,
            supported=1,
            potentially_hallucinated=1,
            nli_disagreements=0,
            evidence_total=1,
            pairs_evaluated=1,
        ),
        timings=HallucinationTimings(),
    )

    class StubCheckService:
        def check(self, result, *, deadline=None, cancel_event=None):
            return check

    StubCheckService.detector = detector

    generator = ScriptedGenerator([repair_json("REMOVED", BETA)])
    service = build_hallucination_repair_service(
        settings=settings,
        verification_engine=CandidateNliEngine(),
        rag_verification_service=RagVerificationService(
            settings=settings,
            verification_engine=StubVerificationEngine(result.verification.details),
        ),
        hallucination_service=StubCheckService(),
        detector=detector,
        generator=generator,
    )

    repair = service.run(result)

    beta_claim = next(claim for claim in repair.claims if claim.original_claim == BETA)
    assert beta_claim.status == "INSUFFICIENT_EVIDENCE"
    assert beta_claim.attempts[0].status == "INSUFFICIENT_EVIDENCE"
    assert generator.calls == []
    assert "[Insufficient evidence] Beta Corporation was founded in 2020." in (
        repair.final_answer
    )
    assert repair.status == "PARTIALLY_VERIFIED"
    assert len(repair.omissions) == 1


def test_repair_prompt_keeps_complete_long_evidence(settings: Settings):
    long_evidence = (
        "Beta Corporation was founded in 2020 in Lyon. "
        + ("Additional context remains in this paragraph. " * 180)
        + "The source conclusion is preserved at the end of the passage."
    )
    long_source = source("e99", long_evidence, page_number=12)
    result = rag_response(
        answer=BETA,
        sources=[long_source],
        verification=nli_result(BETA, [(BETA, "contradicted", 0.9)]),
        result_id="long-evidence-result",
    )
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote="The source conclusion is preserved at the end of the passage.",
            )
        ]
    )
    service, _ = build_repair_service(
        settings,
        result=result,
        detector=detector,
        generator=generator,
    )

    repair = service.run(result)

    assert repair.status == "FULLY_VERIFIED"
    prompt = generator.calls[0][1].content
    assert long_evidence in prompt
    assert "truncated" not in "\n".join(repair.warnings).lower()


def test_repair_is_idempotent_and_reuses_the_cached_result(settings: Settings):
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote=BETA_EVIDENCE.text,
            )
        ]
    )
    service, _ = build_repair_service(
        settings,
        result=original_result(),
        detector=detector,
        generator=generator,
    )

    first = service.run(original_result())
    second = service.run(original_result())

    assert first.repair_id == second.repair_id
    assert len(generator.calls) == 1


def test_repair_retries_malformed_json(settings: Settings):
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            "not json",
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote=BETA_EVIDENCE.text,
            ),
        ]
    )
    service, _ = build_repair_service(
        settings,
        result=original_result(),
        detector=detector,
        generator=generator,
    )

    repair = service.run(original_result())

    beta_claim = next(claim for claim in repair.claims if claim.original_claim == BETA)
    assert [attempt.status for attempt in beta_claim.attempts] == [
        "REPAIR_ERROR",
        "VERIFIED",
    ]
    assert repair.status == "FULLY_VERIFIED"


def test_repair_can_be_disabled(settings: Settings):
    disabled = settings.model_copy(update={"repair_enabled": False})
    detector = FakeHallucinationDetector(default=0.9)
    service, _ = build_repair_service(
        disabled,
        result=original_result(),
        detector=detector,
        generator=ScriptedGenerator([repair_json("REMOVED", BETA)]),
    )

    with pytest.raises(RepairError, match="disabled"):
        service.run(original_result())


def test_repair_api_builds_source_runs_and_returns_cached_result(
    api_client: TestClient,
    settings: Settings,
    rag_result_store: RagResultStore,
):
    detector = FakeHallucinationDetector(
        default=0.9,
        by_claim={"founded in 2021": 0.9, BETA: 0.1},
    )
    generator = ScriptedGenerator(
        [
            repair_json(
                "REPAIRED",
                BETA,
                repaired="Beta Corporation was founded in 2021.",
                quote=BETA_EVIDENCE.text,
            )
        ]
    )
    result = original_result()
    rag_result_store.put(result)
    service, _ = build_repair_service(
        settings,
        result=result,
        detector=detector,
        generator=generator,
    )
    api_client.app.dependency_overrides[get_hallucination_repair_service] = (
        lambda: service
    )

    source_response = api_client.get(f"/api/repair/source/{result.result_id}")
    assert source_response.status_code == 200, source_response.json()
    source_body = source_response.json()
    assert source_body["result_id"] == result.result_id
    assert source_body["repair"] is None
    assert source_body["overview"]["potentially_hallucinated"] == 1
    assert source_body["max_attempts"] == 3

    run_response = api_client.post(
        "/api/repair/run", json={"result_id": result.result_id}
    )
    assert run_response.status_code == 200, run_response.json()
    repair = run_response.json()
    assert repair["status"] == "FULLY_VERIFIED"
    assert repair["final_answer"].endswith(
        "Beta Corporation was founded in 2021."
    )

    second_run = api_client.post(
        "/api/repair/run", json={"result_id": result.result_id}
    )
    assert second_run.status_code == 200
    assert second_run.json()["repair_id"] == repair["repair_id"]
    assert len(generator.calls) == 1

    cached = api_client.get(f"/api/repair/results/{result.result_id}")
    assert cached.status_code == 200
    assert cached.json()["repair_id"] == repair["repair_id"]


def test_repair_api_rejects_missing_result(api_client: TestClient):
    class UnusedService:
        pass

    api_client.app.dependency_overrides[get_hallucination_repair_service] = (
        lambda: UnusedService()
    )
    response = api_client.get("/api/repair/source/missing")
    assert response.status_code == 404
    assert "no longer available" in response.json()["detail"].lower()


def hallu_evidence(
    item: RagSource, probability: float
) -> HallucinationEvidenceResult:
    return HallucinationEvidenceResult(
        evidence_id=item.chunk_id,
        document_id=item.document_id,
        filename=item.filename,
        chunk_id=item.chunk_id,
        chunk_index=item.chunk_index,
        page_number=item.page_number,
        retrieval_score=item.retrieval_score,
        retrieval_similarity=item.similarity,
        support_probability=probability,
        raw_model_label="SUPPORTED" if probability >= 0.5 else "UNSUPPORTED",
    )
