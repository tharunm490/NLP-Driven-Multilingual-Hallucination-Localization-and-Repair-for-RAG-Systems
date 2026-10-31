"""Evidence-grounded, iterative claim repair.

The service reuses the existing Ask result, NLI verifier, and Hallucination
Guard. It never reruns retrieval or answer generation, never changes the stored
Ask result, and never accepts a correction until both existing verification
systems support it independently.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from app.core.config import Settings
from app.core.errors import (
    HallucinationCheckError,
    RepairError,
    RepairResponseError,
)
from app.schemas.hallucination import (
    HallucinationCheckResponse,
    HallucinationClaimResult,
    HallucinationVerdict,
)
from app.schemas.rag import RagQueryResponse, RagSource
from app.schemas.repair import (
    FinalAnswerClaim,
    RepairAttempt,
    RepairClaimResult,
    RepairEvidence,
    RepairIteration,
    RepairOmission,
    RepairOverview,
    RepairResult,
    RepairSourceResponse,
    RepairTimings,
    RepairVerification,
)
from app.schemas.verification import (
    ClaimVerificationResult,
    ClaimVerdict,
    VerificationEvidenceInput,
    VerificationResult,
)
from app.services.generator import Generator
from app.services.hallucination_detector import HallucinationDetector
from app.services.hallucination_policy import HallucinationVerdictPolicy
from app.services.hallucination_service import HallucinationCheckService
from app.services.rag_verification_service import (
    RagVerificationService,
    verification_inputs,
)
from app.services.repair_prompt import (
    build_repair_batch_messages,
    build_repair_messages,
    parse_repair_batch_response,
    parse_repair_response,
    validate_supporting_evidence,
)
from app.services.verification import VerificationEngine

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _PreparedSource:
    verified_result: RagQueryResponse
    check: HallucinationCheckResponse
    overview: RepairOverview


class HallucinationRepairService:
    def __init__(
        self,
        *,
        settings: Settings,
        verification_engine: VerificationEngine,
        rag_verification_service: RagVerificationService,
        hallucination_service: HallucinationCheckService,
        detector: HallucinationDetector,
        generator: Generator,
    ) -> None:
        self._settings = settings
        self._verification_engine = verification_engine
        self._rag_verification_service = rag_verification_service
        self._hallucination_service = hallucination_service
        self._detector = detector
        self._generator = generator
        self._policy = HallucinationVerdictPolicy()

        self._prepared: OrderedDict[str, tuple[float, _PreparedSource]] = (
            OrderedDict()
        )
        self._results: OrderedDict[str, tuple[float, RepairResult]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._preparation_locks: dict[str, threading.Lock] = {}
        self._result_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- public API --------------------------------------------------------

    def get_source(self, result: RagQueryResponse) -> RepairSourceResponse:
        prepared = self._ensure_prepared(result)
        result_id = result.result_id or ""
        return RepairSourceResponse(
            result_id=result_id,
            original=prepared.verified_result,
            hallucination=prepared.check,
            overview=prepared.overview,
            repair=self.get_result(result_id),
            max_attempts=3,
        )

    def run(
        self,
        result: RagQueryResponse,
        *,
        max_attempts: int | None = None,
    ) -> RepairResult:
        if not self._settings.repair_enabled:
            raise RepairError("Hallucination repair is disabled by configuration.")
        result_id = result.result_id
        if not result_id:
            raise RepairError(
                "The Ask result has no identifier; rerun the question."
            )
        lock = self._result_lock(result_id)
        with lock:
            existing = self.get_result(result_id)
            if existing is not None:
                return existing
            prepared = self._ensure_prepared(result)
            repair = self._run(
                result_id=result_id,
                question=result.question,
                prepared=prepared,
                attempts_limit=3,
            )
            self._store_result(result_id, repair)
            return repair

    def get_result(self, result_id: str) -> RepairResult | None:
        with self._cache_lock:
            self._purge_expired_locked()
            entry = self._results.get(result_id)
            if entry is None:
                return None
            self._results.move_to_end(result_id)
            return entry[1].model_copy(deep=True)

    def clear(self) -> None:
        with self._cache_lock:
            self._prepared.clear()
            self._results.clear()

    # -- preparation -------------------------------------------------------

    def _ensure_prepared(self, result: RagQueryResponse) -> _PreparedSource:
        if not self._settings.repair_enabled:
            raise RepairError("Hallucination repair is disabled by configuration.")
        result_id = result.result_id
        if not result_id:
            raise RepairError(
                "The Ask result has no identifier; rerun the question."
            )
        cached = self._get_prepared(result_id)
        if cached is not None:
            return cached
        lock = self._preparation_lock(result_id)
        with lock:
            cached = self._get_prepared(result_id)
            if cached is not None:
                return cached
            prepared = self._prepare(result)
            with self._cache_lock:
                self._prepared[result_id] = (
                    time.monotonic() + self._settings.repair_result_ttl_seconds,
                    prepared,
                )
                self._prepared.move_to_end(result_id)
                while len(self._prepared) > self._settings.repair_result_cache_size:
                    self._prepared.popitem(last=False)
            return prepared

    def _prepare(self, result: RagQueryResponse) -> _PreparedSource:
        deadline = time.monotonic() + self._settings.repair_deadline_seconds
        verified = result.model_copy(deep=True)
        try:
            check = self._hallucination_service.check(
                verified, deadline=deadline
            )
        except HallucinationCheckError as exc:
            raise RepairError(str(exc)) from exc
        overview = self._build_overview(verified, check)
        return _PreparedSource(
            verified_result=verified,
            check=check,
            overview=overview,
        )

    def _build_overview(
        self,
        verified: RagQueryResponse,
        check: HallucinationCheckResponse,
    ) -> RepairOverview:
        details = (
            verified.verification.details
            if verified.verification is not None
            else None
        )
        by_id, by_text = self._nli_index(details)
        disagreements = sum(
            1
            for claim in check.claims
            if self._nli_disagrees(
                claim.verdict,
                self._original_nli(claim, by_id, by_text)[0],
            )
        )
        summary = details.summary if details is not None else None
        return RepairOverview(
            claims_total=check.summary.claims_total,
            supported=check.summary.supported,
            potentially_hallucinated=check.summary.potentially_hallucinated,
            neutral=summary.neutral if summary is not None else 0,
            nli_disagreements=disagreements,
            claims_unevaluable=(
                summary.claims_unevaluable if summary is not None else 0
            ),
            invalid_fragments_skipped=(
                summary.invalid_fragments_skipped if summary is not None else 0
            ),
            evidence_total=check.summary.evidence_total,
        )

    # -- repair run --------------------------------------------------------

    def _run(
        self,
        *,
        result_id: str,
        question: str,
        prepared: _PreparedSource,
        attempts_limit: int,
    ) -> RepairResult:
        started = time.perf_counter()
        verified = prepared.verified_result
        check = prepared.check
        sources_by_chunk = {
            source.chunk_id: source for source in verified.sources
        }
        by_id, by_text = self._nli_index(
            verified.verification.details
            if verified.verification is not None
            else None
        )
        deadline = time.monotonic() + self._settings.repair_deadline_seconds
        warnings: list[str] = []

        ordered = sorted(check.claims, key=lambda claim: claim.claim_start)
        claim_results: list[RepairClaimResult] = []
        for claim in ordered:
            nli_status, nli_confidence = self._original_nli(
                claim, by_id, by_text
            )
            context = self._claim_context(verified.answer, claim.claim_start)
            if claim.verdict == "supported":
                claim_results.append(
                    self._supported_claim_result(
                        claim,
                        nli_status=nli_status,
                        nli_confidence=nli_confidence,
                        context=context,
                        check_claim=claim,
                    )
                )
                continue
            selected, evidence_warnings = self._select_evidence(
                claim, sources_by_chunk
            )
            warnings.extend(evidence_warnings)
            claim_results.append(self._initial_repair_claim(
                claim=claim,
                context=context,
                selected_evidence=selected,
                original_nli_status=nli_status,
                original_nli_confidence=nli_confidence,
            ))

        iterations: list[RepairIteration] = []
        queue = [
            claim for claim in claim_results
            if claim.requires_repair and claim.status == "REPAIRING"
        ]
        for iteration_number in range(1, 4):
            if not queue:
                break
            submitted = list(queue)
            submitted_ids = [claim.claim_id for claim in submitted]
            for claim in submitted:
                claim.attempts.append(
                    RepairAttempt(
                        attempt_number=iteration_number,
                        status="REPAIRED_PENDING_VERIFICATION",
                    )
                )
            batch = [
                (
                    claim.original_claim,
                    claim.original_context,
                    claim.selected_evidence,
                    claim.original_nli_status,
                    claim.original_nli_confidence,
                    claim.original_verdict,
                    claim.original_support_probability,
                    claim.attempts[-2] if len(claim.attempts) > 1 else None,
                )
                for claim in submitted
            ]
            outputs = None
            batch_error: str | None = None
            try:
                generated = self._generator.generate(
                    build_repair_batch_messages(question=question, claims=batch)
                )
                outputs = parse_repair_batch_response(
                    generated.text, [claim.original_claim for claim in submitted]
                )
            except Exception as exc:  # provider and response failures are surfaced per claim
                batch_error = str(exc)
                logger.warning("Repair batch failed: %s", batch_error)

            next_queue: list[RepairClaimResult] = []
            verified_ids: list[str] = []
            for claim in submitted:
                attempt = claim.attempts[-1]
                started_attempt = time.perf_counter()
                if batch_error is not None or outputs is None:
                    attempt.status = "REPAIR_ERROR"
                    attempt.error = batch_error or "No repair response was returned."
                    claim.status = (
                        "UNRESOLVED" if iteration_number == 3 else "REPAIRING"
                    )
                    if iteration_number < 3:
                        next_queue.append(claim)
                    continue
                output = outputs.get(_claim_key(claim.original_claim))
                if output is None:
                    attempt.status = "REPAIR_ERROR"
                    attempt.error = "The repair model omitted this submitted claim."
                    claim.status = (
                        "UNRESOLVED" if iteration_number == 3 else "REPAIRING"
                    )
                    if iteration_number < 3:
                        next_queue.append(claim)
                    continue
                try:
                    supporting = validate_supporting_evidence(
                        output, claim.selected_evidence
                    )
                except RepairResponseError as exc:
                    attempt.status = "REPAIR_ERROR"
                    attempt.error = str(exc)
                    claim.status = (
                        "UNRESOLVED" if iteration_number == 3 else "REPAIRING"
                    )
                    if iteration_number < 3:
                        next_queue.append(claim)
                    continue
                attempt.decision = output.status
                attempt.reason = output.reason or None
                attempt.supporting_evidence = supporting
                if output.status in {"REMOVED", "INSUFFICIENT_EVIDENCE"}:
                    attempt.status = output.status
                    claim.status = output.status
                    claim.final_decision = output.status
                    continue
                repaired = (output.repaired_claim or "").strip()
                attempt.repaired_claim = repaired
                verification = self._verify_candidate(
                    repaired, claim.selected_evidence, deadline=deadline
                )
                attempt.verification = verification
                if verification.both_supported:
                    attempt.status = "VERIFIED"
                    claim.status = "VERIFIED"
                    claim.final_claim = repaired
                    claim.final_decision = output.status
                    claim.included_in_final = True
                    verified_ids.append(claim.claim_id)
                else:
                    attempt.failure_reason = self._failure_reason(verification)
                    attempt.status = "RETRY_REQUIRED"
                    claim.final_claim = repaired
                    claim.final_decision = output.status
                    claim.status = (
                        "UNRESOLVED" if iteration_number == 3 else "REPAIRING"
                    )
                    if iteration_number < 3:
                        next_queue.append(claim)
                attempt.duration_ms = round(
                    (time.perf_counter() - started_attempt) * 1000, 3
                )
            queue = next_queue
            iterations.append(
                RepairIteration(
                    iteration_number=iteration_number,
                    submitted_claim_ids=submitted_ids,
                    verified_claim_ids=verified_ids,
                    remaining_claim_ids=[claim.claim_id for claim in queue],
                    status=(
                        "FAILED" if batch_error is not None
                        else "COMPLETED"
                    ),
                )
            )

        final_answer, final_claims, omissions = self._assemble_final_answer(
            verified.answer, claim_results
        )
        status = self._overall_status(claim_results, final_claims, omissions)
        if status == "REPAIR_INCOMPLETE":
            warnings.append(
                "At least one claim encountered a non-recoverable repair error."
            )
        if any(not claim.included_in_final for claim in claim_results):
            warnings.append(
                "Some original claims were omitted or remain unresolved; they "
                "are not presented as verified facts in the final answer."
            )
        repair = RepairResult(
            repair_id=uuid.uuid4().hex,
            result_id=result_id,
            question=question,
            original_answer=verified.answer,
            original_verification=verified.verification,
            hallucination_check=check,
            overview=prepared.overview,
            claims=claim_results,
            final_answer=final_answer,
            final_claims=final_claims,
            omissions=omissions,
            iterations=iterations,
            status=status,
            max_attempts=3,
            model_provider=getattr(self._generator, "provider", None),
            model_name=getattr(self._generator, "model", None),
            created_at=datetime.now(timezone.utc).isoformat(),
            timings=RepairTimings(
                repair_ms=round((time.perf_counter() - started) * 1000, 3),
                total_ms=round((time.perf_counter() - started) * 1000, 3),
            ),
            warnings=warnings,
        )
        return repair

    def _initial_repair_claim(
        self,
        *,
        claim: HallucinationClaimResult,
        context: str | None,
        selected_evidence: list[RepairEvidence],
        original_nli_status: ClaimVerdict | None,
        original_nli_confidence: float | None,
    ) -> RepairClaimResult:
        base = RepairClaimResult(
            claim_id=claim.claim_id,
            requires_repair=True,
            original_claim=claim.claim,
            original_sentence=claim.sentence,
            original_context=context,
            original_claim_start=claim.claim_start,
            original_claim_end=claim.claim_end,
            original_verdict=claim.verdict,
            original_support_probability=claim.support_probability,
            original_raw_model_label=claim.raw_model_label,
            original_nli_status=original_nli_status,
            original_nli_confidence=original_nli_confidence,
            original_nli_disagrees=self._nli_disagrees(
                claim.verdict, original_nli_status
            ),
            status="REPAIRING",
            selected_evidence=selected_evidence,
        )
        if not selected_evidence:
            base.status = "INSUFFICIENT_EVIDENCE"
            base.final_decision = "INSUFFICIENT_EVIDENCE"
            base.attempts = [
                RepairAttempt(
                    attempt_number=1,
                    status="INSUFFICIENT_EVIDENCE",
                    decision="INSUFFICIENT_EVIDENCE",
                    reason=(
                        "No claim-specific evidence from the original "
                        "verification was available."
                    ),
                )
            ]
        return base

    def _supported_claim_result(
        self,
        claim: HallucinationClaimResult,
        *,
        nli_status: ClaimVerdict | None,
        nli_confidence: float | None,
        context: str | None,
        check_claim: HallucinationClaimResult,
    ) -> RepairClaimResult:
        return RepairClaimResult(
            claim_id=claim.claim_id,
            requires_repair=False,
            original_claim=claim.claim,
            original_sentence=claim.sentence,
            original_context=context,
            original_claim_start=claim.claim_start,
            original_claim_end=claim.claim_end,
            original_verdict=claim.verdict,
            original_support_probability=claim.support_probability,
            original_raw_model_label=claim.raw_model_label,
            original_nli_status=nli_status,
            original_nli_confidence=nli_confidence,
            original_nli_disagrees=self._nli_disagrees(
                check_claim.verdict, nli_status
            ),
            status="ORIGINAL",
            final_claim=claim.claim,
            included_in_final=True,
        )

    def _repair_claim(
        self,
        *,
        claim: HallucinationClaimResult,
        question: str,
        context: str | None,
        selected_evidence: list[RepairEvidence],
        original_nli_status: ClaimVerdict | None,
        original_nli_confidence: float | None,
        attempts_limit: int,
        deadline: float,
    ) -> RepairClaimResult:
        base = RepairClaimResult(
            claim_id=claim.claim_id,
            requires_repair=True,
            original_claim=claim.claim,
            original_sentence=claim.sentence,
            original_context=context,
            original_claim_start=claim.claim_start,
            original_claim_end=claim.claim_end,
            original_verdict=claim.verdict,
            original_support_probability=claim.support_probability,
            original_raw_model_label=claim.raw_model_label,
            original_nli_status=original_nli_status,
            original_nli_confidence=original_nli_confidence,
            original_nli_disagrees=self._nli_disagrees(
                claim.verdict, original_nli_status
            ),
            status="REPAIRING",
            selected_evidence=selected_evidence,
        )
        if not selected_evidence:
            base.status = "INSUFFICIENT_EVIDENCE"
            base.final_decision = "INSUFFICIENT_EVIDENCE"
            base.attempts = [
                RepairAttempt(
                    attempt_number=1,
                    status="INSUFFICIENT_EVIDENCE",
                    decision="INSUFFICIENT_EVIDENCE",
                    reason=(
                        "No claim-specific evidence from the original "
                        "verification was available."
                    ),
                )
            ]
            return base

        attempts: list[RepairAttempt] = []
        for attempt_number in range(1, attempts_limit + 1):
            previous = attempts[-1] if attempts else None
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_number,
                    status="REPAIRED_PENDING_VERIFICATION",
                )
            )
            attempt = attempts[-1]
            started = time.perf_counter()
            messages = build_repair_messages(
                question=question,
                claim=claim.claim,
                context=context,
                evidence=selected_evidence,
                original_nli_status=original_nli_status,
                original_nli_confidence=original_nli_confidence,
                original_guard_verdict=claim.verdict,
                original_guard_support=claim.support_probability,
                previous_attempt=previous,
            )
            try:
                generated = self._generator.generate(messages)
            except Exception as exc:  # noqa: BLE001 - provider failure
                attempt.status = "REPAIR_ERROR"
                attempt.error = str(exc)
                attempt.duration_ms = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                base.status = "REPAIR_ERROR"
                break

            try:
                output = parse_repair_response(
                    generated.text, claim.claim
                )
                supporting = validate_supporting_evidence(
                    output, selected_evidence
                )
            except RepairResponseError as exc:
                attempt.status = "REPAIR_ERROR"
                attempt.error = str(exc)
                attempt.duration_ms = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                if attempt_number >= attempts_limit:
                    base.status = "REPAIR_ERROR"
                    break
                continue

            attempt.decision = output.status
            attempt.reason = output.reason or None
            attempt.supporting_evidence = supporting

            if output.status in {"REMOVED", "INSUFFICIENT_EVIDENCE"}:
                attempt.status = output.status
                attempt.duration_ms = round(
                    (time.perf_counter() - started) * 1000, 3
                )
                base.status = output.status
                base.final_decision = output.status
                break

            repaired = (output.repaired_claim or "").strip()
            attempt.repaired_claim = repaired
            verification = self._verify_candidate(
                repaired, selected_evidence, deadline=deadline
            )
            attempt.verification = verification
            attempt.duration_ms = round(
                (time.perf_counter() - started) * 1000, 3
            )
            if verification.both_supported:
                attempt.status = "VERIFIED"
                base.status = "VERIFIED"
                base.final_claim = repaired
                base.final_decision = output.status
                base.included_in_final = True
                break

            attempt.failure_reason = self._failure_reason(verification)
            attempt.status = "RETRY_REQUIRED"
            base.final_claim = repaired
            base.final_decision = output.status
            if attempt_number >= attempts_limit:
                base.status = "UNRESOLVED"
                break

        base.attempts = attempts
        if base.status == "REPAIRING":
            base.status = "UNRESOLVED"
        return base

    def _verify_candidate(
        self,
        candidate: str,
        evidence: Sequence[RepairEvidence],
        *,
        deadline: float,
    ) -> RepairVerification:
        nli_status, nli_confidence, nli_explanation = self._verify_with_nli(
            candidate, evidence, deadline=deadline
        )
        guard_verdict, guard_probability, guard_raw = self._verify_with_guard(
            candidate, evidence, deadline=deadline
        )
        return RepairVerification(
            nli_status=nli_status,
            nli_confidence=nli_confidence,
            nli_explanation=nli_explanation,
            guard_verdict=guard_verdict,
            guard_support_probability=guard_probability,
            guard_raw_label=guard_raw,
            both_supported=(
                nli_status == "supported" and guard_verdict == "supported"
            ),
        )

    def _verify_with_nli(
        self,
        candidate: str,
        evidence: Sequence[RepairEvidence],
        *,
        deadline: float,
    ) -> tuple[ClaimVerdict | None, float | None, str | None]:
        inputs = [self._verification_input(item) for item in evidence]
        try:
            result = self._verification_engine.verify(
                candidate, inputs, deadline=deadline
            )
        except Exception as exc:  # noqa: BLE001 - NLI is an independent signal
            logger.warning("NLI re-verification failed for repaired claim: %s", exc)
            return None, None, f"NLI verification failed: {exc}"
        return self._nli_result_summary(result)

    def _verify_with_guard(
        self,
        candidate: str,
        evidence: Sequence[RepairEvidence],
        *,
        deadline: float,
    ) -> tuple[HallucinationVerdict | None, float | None, str | None]:
        pairs = [(item.text, candidate) for item in evidence]
        try:
            predictions = self._detector.predict(pairs, deadline=deadline)
        except Exception as exc:  # noqa: BLE001 - Guard is an independent signal
            logger.warning(
                "Hallucination Guard re-verification failed for repaired claim: %s",
                exc,
            )
            return None, None, None
        if not predictions:
            return None, None, None
        best = max(predictions, key=lambda item: item.support_probability)
        return (
            self._policy.classify(best.raw_label),
            best.support_probability,
            best.raw_label,
        )

    @staticmethod
    def _nli_result_summary(
        result: VerificationResult,
    ) -> tuple[ClaimVerdict | None, float | None, str | None]:
        if result.status not in {"completed", "partial"} or not result.claims:
            reason = result.error or (
                "No independently evaluable claim was extracted from the "
                "repaired statement."
            )
            return None, None, reason
        statuses = [
            claim.status for claim in result.claims if claim.status is not None
        ]
        confidences = [
            claim.confidence
            for claim in result.claims
            if claim.confidence is not None
        ]
        confidence = max(confidences) if confidences else None
        if "contradicted" in statuses:
            return "contradicted", confidence, "The NLI verifier returned contradiction."
        if statuses and all(status == "supported" for status in statuses):
            return "supported", confidence, "The NLI verifier supported the repaired claim."
        if "neutral" in statuses:
            return "neutral", confidence, "The NLI verifier returned neutral."
        return None, confidence, "No usable NLI verdict was produced."

    @staticmethod
    def _failure_reason(verification: RepairVerification) -> str:
        if verification.nli_status != "supported":
            nli_reason = (
                f"NLI returned {verification.nli_status}"
                if verification.nli_status
                else "NLI returned no usable verdict"
            )
        else:
            nli_reason = "NLI supported the repaired claim"
        if verification.guard_verdict != "supported":
            guard_reason = (
                f"Hallucination Guard returned {verification.guard_verdict}"
                if verification.guard_verdict
                else "Hallucination Guard returned no usable verdict"
            )
        else:
            guard_reason = "Hallucination Guard supported the repaired claim"
        return f"{nli_reason}; {guard_reason}."

    # -- evidence selection ------------------------------------------------

    def _select_evidence(
        self,
        claim: HallucinationClaimResult,
        sources_by_chunk: dict[str, RagSource],
    ) -> tuple[list[RepairEvidence], list[str]]:
        warnings: list[str] = []
        ranked = sorted(
            claim.evidence,
            key=lambda item: (
                -item.support_probability,
                -item.retrieval_similarity,
                item.chunk_index,
            ),
        )
        selected: list[RepairEvidence] = []
        seen: set[str] = set()
        for item in ranked:
            if item.chunk_id in seen:
                continue
            source = sources_by_chunk.get(item.chunk_id)
            if source is None:
                continue
            text = source.text
            selected.append(
                RepairEvidence(
                    evidence_id=item.evidence_id,
                    document_id=item.document_id,
                    filename=item.filename,
                    chunk_id=item.chunk_id,
                    chunk_index=item.chunk_index,
                    page_number=item.page_number,
                    text=text,
                    retrieval_score=item.retrieval_score,
                    retrieval_similarity=item.retrieval_similarity,
                    guard_support_probability=item.support_probability,
                    guard_raw_label=item.raw_model_label,
                )
            )
            seen.add(item.chunk_id)
            if len(selected) >= self._settings.repair_max_evidence_per_claim:
                break
        return selected, warnings

    # -- final answer assembly --------------------------------------------

    def _assemble_final_answer(
        self,
        answer: str,
        claims: Sequence[RepairClaimResult],
    ) -> tuple[str, list[FinalAnswerClaim], list[RepairOmission]]:
        ordered = sorted(claims, key=lambda item: item.original_claim_start)
        parts: list[str] = []
        final_claims: list[FinalAnswerClaim] = []
        omissions: list[RepairOmission] = []
        cursor = 0
        offset = 0

        def append(text: str) -> int:
            nonlocal offset
            start = offset
            if text:
                parts.append(text)
                offset += len(text)
            return start

        for claim in ordered:
            if claim.original_claim_start < cursor:
                continue
            if not claim.requires_repair:
                append(answer[cursor : claim.original_claim_start])
                start = offset
                text = answer[
                    claim.original_claim_start : claim.original_claim_end
                ]
                append(text)
                final_claims.append(
                    FinalAnswerClaim(
                        claim_id=claim.claim_id,
                        text=text,
                        start=start,
                        end=offset,
                        verdict=(
                            "verified"
                            if claim.original_nli_status == "supported"
                            else "unresolved"
                        ),
                    )
                )
                claim.included_in_final = True
                cursor = claim.original_claim_end
                continue

            if claim.status == "VERIFIED" and claim.final_claim:
                append(answer[cursor : claim.original_claim_start])
                text = self._fit_replacement(
                    claim.final_claim, answer, claim.original_claim_end
                )
                start = offset
                append(text)
                final_claims.append(
                    FinalAnswerClaim(
                        claim_id=claim.claim_id,
                        text=text,
                        start=start,
                        end=offset,
                        verdict="verified",
                    )
                )
                claim.included_in_final = True
                cursor = claim.original_claim_end
                continue

            append(answer[cursor : claim.original_claim_start])
            start = offset
            if claim.status == "INSUFFICIENT_EVIDENCE":
                prefix = "[Insufficient evidence] "
                verdict = "insufficient_evidence"
            elif claim.status == "REMOVED":
                prefix = "[Unresolved] "
                verdict = "unresolved"
            else:
                prefix = "[Potentially hallucinated] "
                verdict = "unresolved"
            append(prefix + answer[
                claim.original_claim_start : claim.original_claim_end
            ])
            final_claims.append(
                FinalAnswerClaim(
                    claim_id=claim.claim_id,
                    text=prefix + answer[
                        claim.original_claim_start : claim.original_claim_end
                    ],
                    start=start,
                    end=offset,
                    verdict=verdict,
                )
            )
            claim.included_in_final = True
            omissions.append(
                RepairOmission(
                    claim_id=claim.claim_id,
                    original_claim=claim.original_claim,
                    reason=self._omission_reason(claim),
                )
            )
            cursor = claim.original_claim_end

        append(answer[cursor:])
        return "".join(parts), final_claims, omissions

    @staticmethod
    def _fit_replacement(text: str, answer: str, end: int) -> str:
        following = answer[end : end + 1]
        if following and following in ".,;:" and text.endswith(following):
            return text[:-1]
        return text

    @staticmethod
    def _removal_span(
        answer: str, claim: RepairClaimResult
    ) -> tuple[int, int]:
        start = claim.original_claim_start
        end = claim.original_claim_end
        line_start = answer.rfind("\n", 0, start) + 1
        line_end = answer.find("\n", end)
        if line_end == -1:
            line_end = len(answer)
        prefix = answer[line_start:start]
        suffix = answer[end:line_end]
        if _is_bullet_prefix(prefix) and suffix.strip() in {"", ".", ";", ",", ":"}:
            return line_start, min(len(answer), line_end + 1)
        return start, end

    @staticmethod
    def _omission_reason(claim: RepairClaimResult) -> str:
        if claim.status == "UNRESOLVED":
            return (
                "The repaired claim did not pass both independent verification "
                "systems within the attempt limit."
            )
        if claim.status == "INSUFFICIENT_EVIDENCE":
            return (
                "The supplied evidence did not establish the claim or support "
                "a reliable correction."
            )
        if claim.status == "REPAIR_ERROR":
            return "A non-recoverable error occurred while repairing the claim."
        return "The claim is not presented as a verified fact."

    # -- overall status ----------------------------------------------------

    @staticmethod
    def _overall_status(
        claims: Sequence[RepairClaimResult],
        final_claims: Sequence[FinalAnswerClaim],
        omissions: Sequence[RepairOmission],
    ) -> str:
        if any(claim.status == "REPAIR_ERROR" for claim in claims):
            return "REPAIR_INCOMPLETE"
        if any(item.verdict == "unresolved" for item in final_claims):
            return "PARTIALLY_VERIFIED"
        if not any(claim.requires_repair for claim in claims):
            return "NO_REPAIR_NEEDED"
        has_unresolved = any(
            not claim.included_in_final and claim.status != "REMOVED"
            for claim in claims
        )
        all_verified = bool(final_claims) and all(
            item.verdict == "verified" for item in final_claims
        )
        if all_verified and not has_unresolved:
            return "FULLY_VERIFIED"
        return "PARTIALLY_VERIFIED"

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _claim_context(answer: str, claim_start: int) -> str | None:
        prefix = answer[:claim_start]
        for line in reversed(prefix.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
            if len(stripped) <= 80 and stripped.endswith(":"):
                return stripped
            break
        return None

    @staticmethod
    def _verification_input(item: RepairEvidence) -> VerificationEvidenceInput:
        return VerificationEvidenceInput(
            text=item.text,
            evidence_id=item.evidence_id,
            document_id=item.document_id,
            filename=item.filename,
            chunk_id=item.chunk_id,
            chunk_index=item.chunk_index,
            page_number=item.page_number,
            retrieval_similarity=item.retrieval_similarity,
            retrieval_score=item.retrieval_score,
        )

    @staticmethod
    def _nli_index(
        details: VerificationResult | None,
    ) -> tuple[
        dict[str, ClaimVerificationResult],
        dict[str, ClaimVerificationResult],
    ]:
        by_id: dict[str, ClaimVerificationResult] = {}
        by_text: dict[str, ClaimVerificationResult] = {}
        if details is None:
            return by_id, by_text
        for claim in details.claims:
            by_id[claim.claim.claim_id] = claim
            by_text[_claim_key(claim.claim.claim)] = claim
        return by_id, by_text

    @staticmethod
    def _original_nli(
        claim: HallucinationClaimResult,
        by_id: dict[str, ClaimVerificationResult],
        by_text: dict[str, ClaimVerificationResult],
    ) -> tuple[ClaimVerdict | None, float | None]:
        match = by_id.get(claim.claim_id) or by_text.get(
            _claim_key(claim.claim)
        )
        if match is not None:
            return match.status, match.confidence
        return claim.nli_status, claim.nli_confidence

    @staticmethod
    def _nli_disagrees(
        verdict: HallucinationVerdict,
        nli_status: ClaimVerdict | None,
    ) -> bool:
        if nli_status is None:
            return False
        if verdict == "supported":
            return nli_status in {"contradicted", "neutral"}
        return nli_status == "supported"

    def _effective_attempts(self, requested: int | None) -> int:
        return 3

    def _get_prepared(self, result_id: str) -> _PreparedSource | None:
        with self._cache_lock:
            self._purge_expired_locked()
            entry = self._prepared.get(result_id)
            if entry is None:
                return None
            self._prepared.move_to_end(result_id)
            return entry[1]

    def _store_result(self, result_id: str, repair: RepairResult) -> None:
        with self._cache_lock:
            self._purge_expired_locked()
            self._results[result_id] = (
                time.monotonic() + self._settings.repair_result_ttl_seconds,
                repair,
            )
            self._results.move_to_end(result_id)
            while len(self._results) > self._settings.repair_result_cache_size:
                self._results.popitem(last=False)

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        for result_id, (expires_at, _) in list(self._prepared.items()):
            if expires_at <= now:
                self._prepared.pop(result_id, None)
        for result_id, (expires_at, _) in list(self._results.items()):
            if expires_at <= now:
                self._results.pop(result_id, None)

    def _preparation_lock(self, result_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._preparation_locks.get(result_id)
            if lock is None:
                lock = threading.Lock()
                self._preparation_locks[result_id] = lock
            return lock

    def _result_lock(self, result_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._result_locks.get(result_id)
            if lock is None:
                lock = threading.Lock()
                self._result_locks[result_id] = lock
            return lock


def _claim_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_bullet_prefix(prefix: str) -> bool:
    stripped = prefix.strip()
    if not stripped:
        return True
    if stripped in {"-", "*", "•", "·"}:
        return True
    if stripped[:-1].isdigit() and stripped[-1] in ".)":
        return True
    return False


def build_hallucination_repair_service(
    *,
    settings: Settings,
    verification_engine: VerificationEngine,
    rag_verification_service: RagVerificationService,
    hallucination_service: HallucinationCheckService,
    detector: HallucinationDetector,
    generator: Generator,
) -> HallucinationRepairService:
    return HallucinationRepairService(
        settings=settings,
        verification_engine=verification_engine,
        rag_verification_service=rag_verification_service,
        hallucination_service=hallucination_service,
        detector=detector,
        generator=generator,
    )


__all__ = [
    "HallucinationRepairService",
    "build_hallucination_repair_service",
]
