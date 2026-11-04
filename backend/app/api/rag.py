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
from app.schemas.rag import CompareQueryResponse, EvalRequest, EvalResponse, RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def query(
    payload: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> RagQueryResponse:
    """Answer a question using retrieved document chunks as evidence."""
    return service.answer(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )


@router.post("/evaluate", response_model=EvalResponse)
def evaluate(
    payload: EvalRequest,
    service: RagService = Depends(get_rag_service),
) -> EvalResponse:
    """Run RAG + LLM-only on the same question, score both against a reference answer.

    Returns ROUGE-L and semantic similarity for each pipeline so callers can
    quantify the improvement the RAG pipeline provides over the LLM baseline.
    """
    return service.evaluate(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )


@router.post("/compare", response_model=CompareQueryResponse)
def compare(
    payload: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> CompareQueryResponse:
    """Run the question through both the RAG pipeline and a bare LLM (no retrieval).

    Returns both answers side-by-side so callers can evaluate the improvement
    the RAG pipeline provides over a baseline LLM-only response.
    """
    return service.compare(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
