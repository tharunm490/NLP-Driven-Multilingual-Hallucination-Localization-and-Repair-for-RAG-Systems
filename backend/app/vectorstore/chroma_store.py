"""Persistent ChromaDB storage for document chunks.

Embeddings are always supplied explicitly, so ChromaDB's default embedding
function is never used. The embedding model name is recorded in the collection
metadata and on every chunk; dimension is verified against stored vectors. This
keeps stored chunks and queries on the same model without modifying collections
after creation (which ChromaDB restricts for HNSW settings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from app.core.errors import VectorStoreError
from app.services.text_chunker import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
        batch_size: int = 256,
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._configured_model = embedding_model
        self._batch_size = max(1, batch_size)
        self._stored_model_cache: str | None = None
        self._dimension_cache: int | None = None

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
                embedding_function=None,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clear app error
            logger.exception("Could not initialise ChromaDB")
            raise VectorStoreError(
                f"Could not initialise ChromaDB at '{persist_directory}'. "
                "Check that the directory is writable and that CHROMA_COLLECTION_NAME "
                "is valid."
            ) from exc

        if self._stored_model() and self._stored_model() != embedding_model:
            raise VectorStoreError(
                f"Collection '{collection_name}' was created with embedding model "
                f"'{self._stored_model()}', but the active model is '{embedding_model}'. "
                "Use a different CHROMA_COLLECTION_NAME or remove the existing Chroma "
                "data before switching models."
            )

        logger.info(
            "ChromaDB ready: collection='%s', path='%s', model='%s', existing_chunks=%s",
            collection_name,
            persist_directory,
            embedding_model,
            self._collection.count(),
        )

    # -- properties --------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def configured_model(self) -> str:
        return self._configured_model

    @property
    def embedding_model(self) -> str | None:
        return self._stored_model()

    @property
    def embedding_dimension(self) -> int | None:
        return self._stored_dimension()

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("Could not count stored chunks.") from exc

    def heartbeat(self) -> int | None:
        try:
            return int(self._client.heartbeat())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("ChromaDB heartbeat failed.") from exc

    # -- metadata discovery ------------------------------------------------

    def _stored_model(self) -> str | None:
        """Model recorded for this collection (metadata, else first chunk)."""
        if self._stored_model_cache:
            return self._stored_model_cache
        metadata = self._collection.metadata or {}
        model = metadata.get("embedding_model")
        if not model:
            try:
                result = self._collection.get(limit=1, include=["metadatas"])
                metadatas = result.get("metadatas") or []
                if metadatas and metadatas[0]:
                    model = metadatas[0].get("embedding_model")
            except Exception as exc:  # noqa: BLE001 - fall back to no guard
                logger.warning("Could not read stored embedding model: %s", exc)
        if model:
            self._stored_model_cache = str(model)
        return self._stored_model_cache

    def _stored_dimension(self) -> int | None:
        """Dimension of an already stored vector, if any chunks exist."""
        if self._dimension_cache:
            return self._dimension_cache
        try:
            result = self._collection.get(limit=1, include=["embeddings"])
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                first = embeddings[0]
                self._dimension_cache = len(first) if first is not None else None
        except Exception as exc:  # noqa: BLE001 - fall back to no guard
            logger.warning("Could not read stored embedding dimension: %s", exc)
        return self._dimension_cache

    def _guard_compatibility(self, embedding_model: str, dimension: int) -> None:
        stored_model = self._stored_model()
        if stored_model and stored_model != embedding_model:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' was created with embedding "
                f"model '{stored_model}', but the active model is '{embedding_model}'. "
                "Use a different CHROMA_COLLECTION_NAME or remove the existing Chroma "
                "data before switching models."
            )
        stored_dimension = self._stored_dimension()
        if stored_dimension and stored_dimension != dimension:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' contains {stored_dimension}-"
                f"dimensional vectors, but the active model produced {dimension}. "
                "Use a different collection or re-ingest with a consistent model."
            )

    # -- writes ------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Number of chunks and embeddings differ; refusing to store partial data."
            )
        if not chunks:
            return 0

        dimension = len(embeddings[0]) if embeddings else 0
        if not dimension:
            raise VectorStoreError("Cannot store empty embeddings.")
        if any(len(vector) != dimension for vector in embeddings):
            raise VectorStoreError("Embedding vectors have inconsistent dimensions.")

        self._guard_compatibility(embedding_model, dimension)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            metadata: dict[str, Any] = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "embedding_model": embedding_model,
            }
            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(metadata)

        stored = 0
        try:
            for start in range(0, len(ids), self._batch_size):
                end = start + self._batch_size
                self._collection.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=[list(vector) for vector in embeddings[start:end]],
                    metadatas=metadatas[start:end],
                )
                stored += len(ids[start:end])
                logger.debug("Stored %s/%s chunks", stored, len(ids))
        except Exception as exc:  # noqa: BLE001
            logger.exception("ChromaDB add failed")
            raise VectorStoreError(
                "Failed to store chunks in ChromaDB. The document was not marked "
                "as successfully ingested."
            ) from exc

        self._dimension_cache = dimension
        self._stored_model_cache = embedding_model
        return stored

    # -- reads -------------------------------------------------------------

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if not query_embedding:
            raise VectorStoreError("Query embedding is empty.")

        stored_model = self._stored_model()
        if stored_model and stored_model != self._configured_model:
            raise VectorStoreError(
                f"The collection was built with '{stored_model}' but the active "
                f"embedding model is '{self._configured_model}'."
            )
        stored_dimension = self._stored_dimension()
        if stored_dimension and len(query_embedding) != stored_dimension:
            raise VectorStoreError(
                f"Query embedding has dimension {len(query_embedding)}, but the "
                f"collection contains {stored_dimension}-dimensional vectors."
            )

        where: dict[str, Any] | None = None
        if document_ids is not None:
            ids_filter = [doc_id for doc_id in document_ids if doc_id]
            if not ids_filter:
                return []
            where = (
                {"document_id": ids_filter[0]}
                if len(ids_filter) == 1
                else {"document_id": {"$in": ids_filter}}
            )

        try:
            total = self._collection.count()
            if total == 0:
                return []
            n_results = min(top_k, total)
            response = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ChromaDB query failed")
            raise VectorStoreError("Retrieval from ChromaDB failed.") from exc

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0] or []
        metadatas = (response.get("metadatas") or [[]])[0] or []
        distances = (response.get("distances") or [[]])[0] or []

        results: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata or {}
            distance = float(distances[index]) if index < len(distances) else 0.0
            results.append(
                {
                    "chunk_id": metadata.get("chunk_id", chunk_id),
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", ""),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "page_number": metadata.get("page_number"),
                    "language": metadata.get("language"),
                    "text": documents[index] if index < len(documents) else "",
                    "distance": distance,
                    # For cosine space, Chroma distance = 1 - cosine similarity.
                    # This is a similarity score, not a probability or confidence.
                    "similarity": 1.0 - distance,
                }
            )
        return results

    def count_document(self, document_id: str) -> int:
        try:
            result = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            return len(result.get("ids") or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not count chunks for %s: %s", document_id, exc)
            return 0

    # -- deletes -----------------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        try:
            existing = self.count_document(document_id)
            self._collection.delete(where={"document_id": document_id})
            return existing
        except Exception as exc:  # noqa: BLE001
            logger.exception("ChromaDB delete failed for %s", document_id)
            raise VectorStoreError("Failed to delete the document from ChromaDB.") from exc
