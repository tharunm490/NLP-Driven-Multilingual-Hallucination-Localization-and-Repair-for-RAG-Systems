"""Retrieval endpoint used to test Phase 2 storage and search."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_retrieval_service
from app.schemas.retrieval import SearchRequest, SearchResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Return the most relevant stored chunks for a query.

    Scores are ChromaDB cosine distances (lower is more similar). They are not
    probabilities or confidence values. This endpoint does not generate answers.
    """
    return service.search(
        query=payload.query,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
