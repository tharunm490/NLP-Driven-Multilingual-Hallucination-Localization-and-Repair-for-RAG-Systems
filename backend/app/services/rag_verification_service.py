"""On-demand claim-level verification for stored RAG answers.

Ask Documents generates an answer without running NLI. This service applies the
existing verification engine to that exact answer and its already-retrieved
sources, then returns an updated result that can be stored and shared.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

from app.core.config import Settings
from app.schemas.rag import (
    AnswerStatus,
    RagQueryResponse,
    RagSource,
    RagVerificationInfo,
)
from app.schemas.verification import (
    VerificationEvidenceInput,
    VerificationResult,
    VerificationSummary,
)
from app.services.verification import VerificationEngine

logger = logging.getLogger(__name__)


class RagVerificationService:
    def __init__(
        self,
        *,
        settings: Settings,
        verification_engine: VerificationEngine,
    ) -> None:
        self._settings = settings
        self._verification_engine = verification_engine

    def verify(
        self,
        result: RagQueryResponse,
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RagQueryResponse:
        if result.verification is not None and result.verification.status in {
            "verified",
            "partial",
        }:
            return result
        if not self._can_verify(result):
            return result.model_copy(
                update={
                    "verification": initial_verification_info(
                        result.generation.used_llm
                    )
                },
                deep=True,
            )

        base_status = result.answer_status
        started = time.perf_counter()
        try:
            verification_result = self._verification_engine.verify(
                result.answer,
                verification_inputs(result.sources),
                deadline=deadline,
                cancel_event=cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - verification is isolated
            logger.warning("Claim verification failed; answer preserved: %s", exc)
            verification_result = VerificationResult(
                status="error",
                error=str(exc)[:1000],
                warnings=["Verification failed; the generated answer was preserved."],
            )
        verification_ms = (time.perf_counter() - started) * 1000
        answer_status = answer_status_after_verification(
            base_status, verification_result
        )
        verification_error = (
            verification_result.error
            if verification_result.status == "error"
            else None
        )
        errors = list(result.diagnostics.errors)
        if verification_error:
            errors.append(verification_error)

        return result.model_copy(
            update={
                "answer_mode": answer_status,
                "answer_status": answer_status,
                "verification": build_verification_info(
                    base_status,
                    answer_status,
                    result.generation.used_llm,
                    verification_result,
                ),
                "diagnostics": result.diagnostics.model_copy(
                    update={
                        "verification_error": verification_error,
                        "errors": errors,
                    }
                ),
                "timings": result.timings.model_copy(
                    update={"verification_ms": round(verification_ms, 3)}
                ),
            },
            deep=True,
        )

    @staticmethod
    def _can_verify(result: RagQueryResponse) -> bool:
        return (
            result.generation.used_llm
            and bool(result.sources)
            and result.answer_status in {"verification_pending", "partially_grounded"}
        )


def initial_verification_info(used_llm: bool) -> RagVerificationInfo:
    if not used_llm:
        return RagVerificationInfo(
            claim_level_verification=False,
            status="not_performed",
            method="none",
            note=(
                "No generated answer was available, so there were no factual "
                "claims to verify."
            ),
        )
    return RagVerificationInfo(
        claim_level_verification=False,
        status="pending",
        method="none",
        note=(
            "Claim-level verification has not been run. Use Verify Answer Claims "
            "to check this generated answer against the retrieved evidence."
        ),
    )


def verification_inputs(
    sources: Sequence[RagSource],
) -> list[VerificationEvidenceInput]:
    return [
        VerificationEvidenceInput(
            text=source.text,
            evidence_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            chunk_id=source.chunk_id,
            chunk_index=source.chunk_index,
            page_number=source.page_number,
            retrieval_similarity=source.similarity,
            retrieval_score=source.retrieval_score,
        )
        for source in sources
    ]


def answer_status_after_verification(
    current: AnswerStatus, result: VerificationResult
) -> AnswerStatus:
    if result.status not in {"completed", "partial"}:
        return current
    summary = result.summary
    if summary.claims_total == 0:
        return current
    if summary.contradicted > 0:
        return "contradicted"
    if summary.claims_unevaluable == summary.claims_total:
        return "insufficient_evidence"
    if summary.claims_checked == 0:
        if current in {"grounded", "partially_grounded"}:
            return "insufficient_evidence"
        return current
    if (
        summary.supported == summary.claims_total
        and summary.claims_complete == summary.claims_total
    ):
        return "grounded"
    if summary.supported > 0 or summary.claims_incomplete > 0:
        return "partially_grounded"
    return "insufficient_evidence"


def _verification_counts(summary: VerificationSummary) -> dict[str, int]:
    return {
        "claims_total": summary.claims_total,
        "claims_evaluable": summary.claims_evaluable,
        "claims_unevaluable": summary.claims_unevaluable,
        "claims_checked": summary.claims_checked,
        "claims_complete": summary.claims_complete,
        "claims_incomplete": summary.claims_incomplete,
        "supported": summary.supported,
        "contradicted": summary.contradicted,
        "neutral": summary.neutral,
        "unevaluable": summary.unevaluable,
        "verification_error": summary.verification_error,
        "evidence_total": summary.evidence_total,
        "pairs_expected": summary.pairs_expected,
        "pairs_evaluated": summary.pairs_evaluated,
        "pairs_missing": summary.pairs_missing,
    }


def build_verification_info(
    base_status: AnswerStatus,
    answer_status: AnswerStatus,
    used_llm: bool,
    result: VerificationResult | None,
) -> RagVerificationInfo:
    if result is None:
        return initial_verification_info(used_llm)

    if result.status == "disabled":
        return RagVerificationInfo(
            claim_level_verification=False,
            status="not_performed",
            method="none",
            note="Claim-level verification is disabled by configuration.",
            details=result,
        )
    summary = result.summary
    if result.status == "error":
        return RagVerificationInfo(
            claim_level_verification=summary.claims_total > 0,
            status="error",
            method="error",
            note=(
                "Claim-level verification failed for every extracted claim, "
                "so the generated answer is reported without a verification "
                "decision."
            ),
            **_verification_counts(summary),
            details=result,
        )

    if result.status == "partial":
        coverage = (
            f"Evaluated {summary.pairs_evaluated} of "
            f"{summary.pairs_expected} expected claim-evidence pair(s); "
            f"{summary.claims_complete} of {summary.claims_total} "
            "extracted claim(s) have complete evidence coverage"
        )
        limitations: list[str] = []
        if summary.claims_incomplete > 0:
            limitations.append(
                f"{summary.claims_incomplete} claim(s) have partial "
                "evidence-pair coverage"
            )
        if summary.verification_error > 0:
            limitations.append(
                f"{summary.verification_error} claim(s) could not be "
                "verified because no usable pair result was available"
            )
        if summary.unevaluable > 0:
            limitations.append(
                f"{summary.unevaluable} claim(s) could not be evaluated as "
                "independent propositions"
            )
        outcomes = (
            f"{summary.supported} supported, {summary.contradicted} "
            f"contradicted, {summary.neutral} neutral"
        )
        note = f"{coverage}; {outcomes}."
        if limitations:
            note = f"{note} Limitations: {'; '.join(limitations)}."
        if base_status != answer_status:
            note = (
                f"{note} Overall status changed from {base_status} to "
                f"{answer_status}."
            )
        return RagVerificationInfo(
            claim_level_verification=True,
            status="partial",
            method="multilingual_nli",
            note=note,
            **_verification_counts(summary),
            details=result,
        )

    if summary.claims_total == 0:
        note = "Verification ran, but no factual claims were extracted from the answer."
    elif summary.contradicted > 0:
        note = (
            f"{summary.contradicted} claim(s) were contradicted by the "
            f"retrieved evidence out of {summary.claims_total} extracted "
            "claim(s)."
        )
    elif summary.supported == summary.claims_total:
        note = (
            f"All {summary.claims_total} extracted claim(s) were supported by "
            "the retrieved evidence."
        )
    elif summary.supported > 0:
        note = (
            f"{summary.supported} of {summary.claims_total} extracted claim(s) "
            "were supported; the remainder were neutral or not independently "
            "evaluable."
        )
        if summary.neutral > 0:
            note = (
                f"{note} {summary.neutral} claim(s) had no evaluated evidence "
                "that met a support or contradiction threshold."
            )
        if summary.unevaluable > 0:
            note = (
                f"{note} {summary.unevaluable} claim(s) were not independently "
                "evaluable."
            )
    elif summary.neutral > 0:
        note = (
            f"None of the {summary.claims_total} extracted claim(s) were "
            "established or contradicted by the evaluated evidence; "
            f"{summary.neutral} received a neutral verdict."
        )
    elif summary.unevaluable > 0:
        note = (
            "No claim was independently evaluable, so the answer cannot be "
            "treated as grounded by claim-level verification."
        )
    else:
        note = (
            "None of the extracted claims were established by the evaluated "
            "evidence windows."
        )
    if base_status != answer_status:
        note = f"{note} Overall status changed from {base_status} to {answer_status}."
    return RagVerificationInfo(
        claim_level_verification=True,
        status="verified",
        method="multilingual_nli",
        note=note,
        **_verification_counts(summary),
        details=result,
    )


__all__ = [
    "RagVerificationService",
    "initial_verification_info",
]
