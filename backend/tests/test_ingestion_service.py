import pytest

from app.core.config import Settings
from app.core.errors import DocumentValidationError
from app.services.document_processor import DocumentProcessor
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.language_detection import LanguageDetector
from app.services.text_chunker import TextChunker
from app.vectorstore.chroma_store import ChromaStore
from tests.conftest import make_pdf_bytes

SAMPLE_TEXT = (
    "The Eiffel Tower is located in Paris. "
    "It was completed in 1889 for the World's Fair. "
    "The tower is 330 metres tall. "
    "यह एक बहुभाषी दस्तावेज़ है। "
    "இது ஒரு பன்மொழி ஆவணம்."
)


def test_ingest_txt_end_to_end(ingestion_service: IngestionService, registry, store):
    result = ingestion_service.ingest("sample.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    assert result.status == "completed"
    assert result.document_id is not None
    assert result.chunks_stored == result.chunk_count > 0
    assert result.character_count and result.character_count > 0
    assert result.token_count and result.token_count > 0
    assert result.error is None

    record = registry.get(result.document_id)
    assert record is not None and record.status == "completed"
    assert store.count_document(result.document_id) == result.chunks_stored


def test_ingest_pdf_end_to_end(ingestion_service: IngestionService):
    data = make_pdf_bytes(["Page one has facts.", "Page two has more facts."])
    result = ingestion_service.ingest("doc.pdf", "application/pdf", data)
    assert result.status == "completed"
    assert result.page_count == 2
    assert result.chunks_stored > 0


def test_duplicate_upload_is_reused(
    ingestion_service: IngestionService,
    registry: DocumentRegistry,
):
    first = ingestion_service.ingest("a.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    second = ingestion_service.ingest("b.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    assert second.duplicate is True
    assert second.document_id == first.document_id
    assert registry.count() == 1


def test_duplicates_allowed_when_configured(
    settings: Settings,
    processor: DocumentProcessor,
    chunker: TextChunker,
    embedding_service: EmbeddingService,
    store: ChromaStore,
    registry: DocumentRegistry,
):
    settings = settings.model_copy(update={"allow_duplicate_documents": True})
    service = IngestionService(
        settings,
        processor,
        chunker,
        embedding_service,
        store,
        registry,
        LanguageDetector(enabled=False),
    )
    first = service.ingest("a.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    second = service.ingest("b.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    assert second.duplicate is False
    assert second.document_id != first.document_id
    assert registry.count() == 2


def test_corrupt_pdf_is_reported_as_failed(
    ingestion_service: IngestionService,
    registry: DocumentRegistry,
    store: ChromaStore,
):
    result = ingestion_service.ingest(
        "broken.pdf", "application/pdf", b"%PDF-1.4 not a real pdf"
    )
    assert result.status == "failed"
    assert result.error
    record = registry.get(result.document_id)
    assert record is not None and record.status == "failed"
    assert store.count_document(result.document_id) == 0


def test_image_only_pdf_reports_ocr_message(ingestion_service: IngestionService):
    data = make_pdf_bytes([None])
    result = ingestion_service.ingest("scan.pdf", "application/pdf", data)
    assert result.status == "failed"
    assert "OCR" in (result.error or "")


def test_unsupported_extension_is_rejected_by_validation(ingestion_service: IngestionService):
    with pytest.raises(DocumentValidationError):
        ingestion_service.ingest("notes.docx", "application/octet-stream", b"hello")


def test_empty_payload_is_rejected(ingestion_service: IngestionService):
    with pytest.raises(DocumentValidationError):
        ingestion_service.ingest("empty.txt", "text/plain", b"")


def test_delete_document_removes_chunks_and_record(
    ingestion_service: IngestionService,
    registry: DocumentRegistry,
    store: ChromaStore,
):
    result = ingestion_service.ingest("del.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    deleted = ingestion_service.delete_document(result.document_id)
    assert deleted == result.chunks_stored
    assert registry.get(result.document_id) is None
    assert store.count_document(result.document_id) == 0
