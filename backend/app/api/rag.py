"""RAG question-answering endpoint (Phase 3).

Pipeline: language detection -> conservative preprocessing -> query embedding
(same model as ingestion) -> ChromaDB top-K retrieval -> evidence policy ->
LLM answer grounded in the retrieved evidence. This endpoint does not verify
or localize hallucinations; that is a later phase.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.schemas.rag import RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def query(
    payload: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> RagQueryResponse:
    """Answer a question using retrieved document chunks as evidence.

    ``sources[].retrieval_score`` is a ChromaDB cosine distance (lower is more
    similar) and is not a probability or a factual-verification result.
    """
    return service.answer(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
