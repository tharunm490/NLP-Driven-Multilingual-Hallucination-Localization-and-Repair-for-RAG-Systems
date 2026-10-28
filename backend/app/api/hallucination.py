"""Dedicated hallucination groundedness endpoint."""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_hallucination_check_service,
    get_rag_result_store,
    get_request_limiter,
)
from app.core.config import Settings, get_settings
from app.core.errors import RagResultNotFoundError
from app.schemas.hallucination import (
    HallucinationCheckRequest,
    HallucinationCheckResponse,
)
from app.services.hallucination_service import HallucinationCheckService
from app.services.rag_result_store import RagResultStore
from app.services.request_limiter import RequestLimiter

router = APIRouter(prefix="/hallucination", tags=["hallucination"])


@router.post("/check", response_model=HallucinationCheckResponse)
async def check(
    payload: HallucinationCheckRequest,
    result_store: RagResultStore = Depends(get_rag_result_store),
    service: HallucinationCheckService = Depends(get_hallucination_check_service),
    limiter: RequestLimiter = Depends(get_request_limiter),
    settings: Settings = Depends(get_settings),
) -> HallucinationCheckResponse:
    """Run the local hallucination detector against a completed Ask result."""
    result = result_store.get(payload.result_id)
    if result is None:
        raise RagResultNotFoundError(
            "That Ask result is no longer available. Run the question again "
            "from Ask Documents."
        )

    cancel_event = threading.Event()
    deadline = time.monotonic() + settings.hallucination_deadline_seconds

    def run() -> HallucinationCheckResponse:
        with limiter.slot():
            return service.check(
                result,
                deadline=deadline,
                cancel_event=cancel_event,
            )

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, run)
    except asyncio.CancelledError:
        cancel_event.set()
        raise
