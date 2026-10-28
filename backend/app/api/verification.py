"""Standalone claim-level verification endpoint (Phase 4A)."""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_rag_result_store,
    get_rag_verification_service,
    get_request_limiter,
    get_verification_engine,
)
from app.core.config import Settings, get_settings
from app.core.errors import RagResultNotFoundError
from app.schemas.rag import RagQueryResponse
from app.schemas.verification import VerificationCheckRequest, VerificationResult
from app.services.rag_result_store import RagResultStore
from app.services.rag_verification_service import RagVerificationService
from app.services.request_limiter import RequestLimiter
from app.services.verification import VerificationEngine

router = APIRouter(prefix="/verification", tags=["verification"])


@router.post("/check", response_model=VerificationResult)
async def check(
    payload: VerificationCheckRequest,
    engine: VerificationEngine = Depends(get_verification_engine),
    limiter: RequestLimiter = Depends(get_request_limiter),
    settings: Settings = Depends(get_settings),
) -> VerificationResult:
    """Extract claims from an answer and test them against supplied evidence.

    NLI confidence values are model scores, not calibrated probabilities that a
    claim is factually true.
    """
    cancel_event = threading.Event()
    deadline = time.monotonic() + settings.verification_deadline_seconds

    def run() -> VerificationResult:
        with limiter.slot():
            return engine.verify(
                payload.answer,
                payload.evidence,
                deadline=deadline,
                cancel_event=cancel_event,
            )

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, run)
    except asyncio.CancelledError:
        cancel_event.set()
        raise


@router.post("/results/{result_id}", response_model=RagQueryResponse)
async def verify_result(
    result_id: str,
    result_store: RagResultStore = Depends(get_rag_result_store),
    service: RagVerificationService = Depends(get_rag_verification_service),
    limiter: RequestLimiter = Depends(get_request_limiter),
    settings: Settings = Depends(get_settings),
) -> RagQueryResponse:
    """Verify the claims in an existing Ask answer against its stored evidence.

    This reuses the exact answer and retrieved sources. It never reruns
    retrieval or generation, and already-verified results are returned as-is.
    """
    result = result_store.get(result_id)
    if result is None:
        raise RagResultNotFoundError(
            "That Ask result is no longer available. Run the question again "
            "from Ask Documents."
        )

    cancel_event = threading.Event()
    deadline = time.monotonic() + settings.verification_deadline_seconds

    def run() -> RagQueryResponse:
        with limiter.slot():
            return service.verify(
                result,
                deadline=deadline,
                cancel_event=cancel_event,
            )

    loop = asyncio.get_running_loop()
    try:
        updated = await loop.run_in_executor(None, run)
    except asyncio.CancelledError:
        cancel_event.set()
        raise
    return result_store.put(updated)
