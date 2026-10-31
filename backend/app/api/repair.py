"""Evidence-grounded, iterative hallucination repair endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_hallucination_repair_service,
    get_rag_result_store,
    get_request_limiter,
)
from app.core.errors import RagResultNotFoundError
from app.schemas.repair import (
    RepairResult,
    RepairRunRequest,
    RepairSourceResponse,
)
from app.services.hallucination_repair_service import HallucinationRepairService
from app.services.rag_result_store import RagResultStore
from app.services.request_limiter import RequestLimiter

router = APIRouter(prefix="/repair", tags=["repair"])


@router.get("/source/{result_id}", response_model=RepairSourceResponse)
async def get_source(
    result_id: str,
    result_store: RagResultStore = Depends(get_rag_result_store),
    service: HallucinationRepairService = Depends(
        get_hallucination_repair_service
    ),
    limiter: RequestLimiter = Depends(get_request_limiter),
) -> RepairSourceResponse:
    """Return the original answer, guard findings, and any saved repair."""
    result = result_store.get(result_id)
    if result is None:
        raise RagResultNotFoundError(
            "That Ask result is no longer available. Run the question again "
            "from Ask Documents."
        )

    def run() -> RepairSourceResponse:
        with limiter.slot():
            return service.get_source(result)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run)


@router.post("/run", response_model=RepairResult)
async def run_repair(
    payload: RepairRunRequest,
    result_store: RagResultStore = Depends(get_rag_result_store),
    service: HallucinationRepairService = Depends(
        get_hallucination_repair_service
    ),
    limiter: RequestLimiter = Depends(get_request_limiter),
) -> RepairResult:
    """Repair unsupported claims, re-verifying every model-proposed change."""
    result = result_store.get(payload.result_id)
    if result is None:
        raise RagResultNotFoundError(
            "That Ask result is no longer available. Run the question again "
            "from Ask Documents."
        )

    def run() -> RepairResult:
        with limiter.slot():
            return service.run(result)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run)


@router.get("/results/{result_id}", response_model=RepairResult)
async def get_repair_result(
    result_id: str,
    service: HallucinationRepairService = Depends(
        get_hallucination_repair_service
    ),
) -> RepairResult:
    """Return a previously completed repair, if it is still cached."""
    repair = service.get_result(result_id)
    if repair is None:
        raise RagResultNotFoundError(
            "No repair result is available for that Ask result yet."
        )
    return repair
