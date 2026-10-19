"""Retrieval service for Phase 2 testing (no answer generation)."""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.core.errors import VectorStoreError
from app.schemas.retrieval import RetrievedChunk, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        store: ChromaStore,
    ) -> None:
        self._settings = settings
        self._embeddings = embedding_service
        self._store = store

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise VectorStoreError("Search query must not be empty.")

        active_model = self._embeddings.model_name
        stored_model = self._store.embedding_model
        if stored_model and stored_model != active_model:
            raise VectorStoreError(
                f"The collection was built with '{stored_model}' but the active "
                f"embedding model is '{active_model}'. Query and document embeddings "
                "must come from the same model."
            )

        k = top_k if top_k is not None else self._settings.top_k
        k = max(1, min(k, 50))

        query_embedding = self._embeddings.embed_query(query)
        raw_results = self._store.query(
            query_embedding,
            top_k=k,
            document_ids=document_ids,
        )

        results = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                filename=item["filename"],
                chunk_index=item["chunk_index"],
                page_number=item.get("page_number"),
                language=item.get("language"),
                text=item["text"],
                distance=item["distance"],
                similarity=item["similarity"],
            )
            for item in raw_results
        ]

        logger.info("Retrieved %s chunks for query (top_k=%s)", len(results), k)
        return SearchResponse(
            query=query,
            top_k=k,
            metric="cosine_distance",
            embedding_model=active_model,
            results=results,
            total=len(results),
        )
