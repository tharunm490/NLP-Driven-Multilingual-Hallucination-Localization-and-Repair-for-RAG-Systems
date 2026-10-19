"""Document ingestion orchestration.

Pipeline: validate -> extract -> clean -> chunk -> embed -> store -> registry.
Expected per-file failures are recorded in the registry and returned as a
structured failed result, so a multi-file upload can continue.
"""

from __future__ import annotations

import hashlib
import logging
import uuid

from app.core.config import Settings
from app.core.errors import (
    DocumentProcessingError,
    DocumentValidationError,
    EmbeddingError,
    VectorStoreError,
)
from app.schemas.document import UploadItemResult
from app.services.document_processor import DocumentProcessor, ExtractedPage
from app.services.document_registry import DocumentRecord, DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.language_detection import LanguageDetector
from app.services.text_chunker import TextChunker
from app.services.text_cleaner import clean_text, has_meaningful_text
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        processor: DocumentProcessor,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        store: ChromaStore,
        registry: DocumentRegistry,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        self._settings = settings
        self._processor = processor
        self._chunker = chunker
        self._embeddings = embedding_service
        self._store = store
        self._registry = registry
        self._language_detector = language_detector or LanguageDetector(
            enabled=settings.language_detection_enabled
        )

    def ingest(
        self,
        filename: str | None,
        content_type: str | None,
        data: bytes,
    ) -> UploadItemResult:
        """Ingest one file. Validation errors for rejected files propagate."""
        safe_name, _extension, _media_type = self._processor.validate(filename, content_type)
        if not data:
            raise DocumentValidationError("The uploaded file is empty.")

        checksum = hashlib.sha256(data).hexdigest()

        if not self._settings.allow_duplicate_documents:
            existing = self._registry.find_completed_by_checksum(checksum)
            if existing is not None:
                logger.info(
                    "Duplicate upload detected (%s); reusing document %s",
                    safe_name,
                    existing.document_id,
                )
                return self._from_record(
                    existing,
                    original_filename=safe_name,
                    duplicate=True,
                    message=(
                        f"Identical content was already ingested as "
                        f"'{existing.filename}'. Reusing that document."
                    ),
                )

        document_id = str(uuid.uuid4())
        self._registry.create(
            document_id=document_id,
            filename=safe_name,
            checksum=checksum,
            size_bytes=len(data),
            content_type=content_type,
        )

        page_count: int | None = None
        character_count: int | None = None
        token_count: int | None = None
        chunk_count = 0

        try:
            extracted = self._processor.extract(data, safe_name, content_type)
            page_count = extracted.page_count

            cleaned_pages: list[ExtractedPage] = []
            text_parts: list[str] = []
            for page in extracted.pages:
                cleaned = clean_text(page.text)
                if cleaned and has_meaningful_text(cleaned):
                    cleaned_pages.append(
                        ExtractedPage(page_number=page.page_number, text=cleaned)
                    )
                    text_parts.append(cleaned)

            if not cleaned_pages:
                raise DocumentProcessingError(
                    "No usable text remained after extraction and cleaning."
                )

            full_text = "\n\n".join(text_parts)
            character_count = len(full_text)
            token_count = self._chunker.count_tokens(full_text)
            language = self._language_detector.detect(full_text)

            chunks = self._chunker.chunk_document(
                cleaned_pages,
                document_id=document_id,
                filename=safe_name,
            )
            chunk_count = len(chunks)
            if not chunks:
                raise DocumentProcessingError("No chunks could be created from the document.")

            embeddings = self._embeddings.embed_documents([chunk.text for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise EmbeddingError(
                    "The embedding service returned an unexpected number of vectors."
                )

            stored = self._store.add_chunks(
                chunks,
                embeddings,
                embedding_model=self._embeddings.model_name,
            )

            updated = self._registry.update(
                document_id,
                status="completed",
                page_count=page_count,
                character_count=character_count,
                token_count=token_count,
                chunk_count=chunk_count,
                chunks_stored=stored,
                language=language,
                embedding_model=self._embeddings.model_name,
                error=None,
            )
            logger.info(
                "Ingested '%s' as %s (%s chunks, %s chars)",
                safe_name,
                document_id,
                stored,
                character_count,
            )
            return self._from_record(updated, original_filename=safe_name)

        except (DocumentValidationError, DocumentProcessingError, EmbeddingError, VectorStoreError) as exc:
            self._cleanup_orphans(document_id)
            self._mark_failed(document_id, exc.message)
            logger.warning("Ingestion failed for '%s': %s", safe_name, exc.message)
            return UploadItemResult(
                document_id=document_id,
                original_filename=safe_name,
                status="failed",
                size_bytes=len(data),
                content_type=content_type,
                page_count=page_count,
                character_count=character_count,
                token_count=token_count,
                chunk_count=chunk_count,
                chunks_stored=0,
                error=exc.message,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive, log details
            logger.exception("Unexpected ingestion failure for '%s'", safe_name)
            self._cleanup_orphans(document_id)
            message = "Unexpected error while processing the document. Check backend logs."
            self._mark_failed(document_id, message)
            return UploadItemResult(
                document_id=document_id,
                original_filename=safe_name,
                status="failed",
                size_bytes=len(data),
                content_type=content_type,
                page_count=page_count,
                character_count=character_count,
                token_count=token_count,
                chunk_count=chunk_count,
                chunks_stored=0,
                error=message,
            )

    def delete_document(self, document_id: str) -> int:
        record = self._registry.get(document_id)
        if record is None:
            from app.core.errors import DocumentNotFoundError

            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        deleted_chunks = self._store.delete_document(document_id)
        self._registry.delete(document_id)
        logger.info("Deleted document %s (%s chunks)", document_id, deleted_chunks)
        return deleted_chunks

    # -- helpers -----------------------------------------------------------

    def _cleanup_orphans(self, document_id: str) -> None:
        try:
            self._store.delete_document(document_id)
        except Exception:  # noqa: BLE001 - cleanup best effort
            logger.warning("Could not clean up orphaned chunks for %s", document_id)

    def _mark_failed(self, document_id: str, message: str) -> None:
        try:
            self._registry.update(document_id, status="failed", error=message)
        except Exception:  # noqa: BLE001
            logger.exception("Could not mark document %s as failed", document_id)

    def _from_record(
        self,
        record: DocumentRecord,
        original_filename: str,
        duplicate: bool = False,
        message: str | None = None,
    ) -> UploadItemResult:
        return UploadItemResult(
            document_id=record.document_id,
            original_filename=original_filename,
            status=record.status,  # type: ignore[arg-type]
            size_bytes=record.size_bytes,
            content_type=record.content_type,
            page_count=record.page_count,
            character_count=record.character_count,
            token_count=record.token_count,
            chunk_count=record.chunk_count,
            chunks_stored=record.chunks_stored,
            language=record.language,
            duplicate=duplicate,
            error=record.error,
            message=message,
        )
