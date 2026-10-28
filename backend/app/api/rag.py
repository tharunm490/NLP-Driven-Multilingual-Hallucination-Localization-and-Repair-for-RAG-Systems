"""RAG question-answering endpoint (Phase 3).

Pipeline: language detection -> conservative preprocessing -> query embedding
(same model as ingestion) -> ChromaDB top-K retrieval -> evidence policy ->
LLM answer grounded in the retrieved evidence. This endpoint does not verify
or localize hallucinations; that is a later phase.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_rag_result_store,
    get_rag_service,
    get_request_limiter,
)
from app.core.errors import RagResultNotFoundError
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.rag_result_store import RagResultStore
from app.services.rag_service import RagService
from app.services.request_limiter import RequestLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
async def query(
    payload: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
    limiter: RequestLimiter = Depends(get_request_limiter),
    result_store: RagResultStore = Depends(get_rag_result_store),
) -> RagQueryResponse:
    """Answer a question using retrieved document chunks as evidence.

    The answer and its sources are returned immediately. Claim-level NLI is
    intentionally not run here; use the stored result's verification endpoint.
    ``sources[].retrieval_score`` is a ChromaDB cosine distance (lower is more
    similar) and is not a probability or a factual-verification result.
    """

    def run() -> RagQueryResponse:
        with limiter.slot():
            return service.answer(
                question=payload.question,
                top_k=payload.top_k,
                document_ids=payload.document_ids,
            )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, run)
    return result_store.put(response)


@router.get("/results/{result_id}", response_model=RagQueryResponse)
def get_result(
    result_id: str,
    result_store: RagResultStore = Depends(get_rag_result_store),
) -> RagQueryResponse:
    """Return a recently completed answer without rerunning retrieval or generation."""
    result = result_store.get(result_id)
    if result is None:
        raise RagResultNotFoundError(
            "That Ask result is no longer available. Run the question again "
            "from Ask Documents."
        )
    return result
