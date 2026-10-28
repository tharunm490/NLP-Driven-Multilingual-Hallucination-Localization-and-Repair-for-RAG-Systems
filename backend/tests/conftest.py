"""Shared pytest fixtures.

Tests never download a real embedding model: a deterministic fake encoder is
injected through ``EmbeddingService``'s model factory. ChromaDB and the SQLite
registry run against per-test temporary directories.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_document_processor,
    get_document_registry,
    get_document_registry_status,
    get_embedding_status,
    get_ingestion_service,
    get_rag_result_store,
    get_rag_service,
    get_rag_verification_service,
    get_retrieval_service,
    get_vector_store_status,
)
from app.core.config import Settings, get_settings
from app.main import create_app
from app.schemas.health import ComponentStatus
from app.services.document_processor import DocumentProcessor
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.evidence_assessor import EvidenceAssessor
from app.services.generator import MockGenerator
from app.services.ingestion_service import IngestionService
from app.services.language_detection import LanguageDetector
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import (
    LanguageScore,
    QueryLanguageDetector,
)
from app.services.rag_result_store import RagResultStore
from app.services.rag_service import RagService
from app.services.rag_verification_service import RagVerificationService
from app.services.retrieval_service import RetrievalService
from app.services.text_chunker import TextChunker
from app.vectorstore.chroma_store import ChromaStore
from tests.rag_support import FixedEvidenceAssessor
from tests.verification_support import StubVerificationEngine

EMBEDDING_DIMENSION = 16


class WordTokenCounter:
    """Deterministic whitespace token counter for chunking tests."""

    def count(self, text: str) -> int:
        return len(text.split())


class FakeEncoderModel:
    """Deterministic, dependency-free stand-in for a sentence-transformer."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION, fail: bool = False) -> None:
        self.max_seq_length = 512
        self._dimension = dimension
        self._fail = fail
        self.encode_calls: list[int] = []
        self.last_inputs: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, batch_size=None, normalize_embeddings=False, **kwargs):
        if self._fail:
            raise RuntimeError("simulated inference failure")
        self.encode_calls.append(len(texts))
        self.last_inputs = list(texts)
        vectors = [self._vector(text) for text in texts]
        if normalize_embeddings:
            vectors = [_normalise(vector) for vector in vectors]
        return vectors

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255.0 + 0.001 for index in range(self._dimension)]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class FakeLanguageBackend:
    """Deterministic language backend for tests (no Lingua/langdetect needed)."""

    name = "fake"

    def __init__(self, scores: list[LanguageScore] | None = None) -> None:
        self._scores = scores

    def is_available(self) -> bool:
        return True

    def detect(self, text: str) -> list[LanguageScore]:
        if self._scores is not None:
            return list(self._scores)
        if any("\u0900" <= char <= "\u097f" for char in text):
            return [LanguageScore("hi", 0.98)]
        if any("\u0b80" <= char <= "\u0bff" for char in text):
            return [LanguageScore("ta", 0.98)]
        return [LanguageScore("en", 0.95)]


class UnavailableLanguageBackend:
    name = "unavailable"

    def is_available(self) -> bool:
        return False

    def detect(self, text: str) -> list[LanguageScore]:  # pragma: no cover
        return []


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        frontend_origin="http://localhost:3000",
        log_level="WARNING",
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="test_documents",
        chroma_batch_size=8,
        embedding_model_name="test/fake-embedding-model",
        embedding_device="cpu",
        embedding_batch_size=2,
        embedding_query_prefix="",
        embedding_passage_prefix="",
        chunk_size=60,
        chunk_overlap=10,
        top_k=3,
        max_upload_size_mb=1,
        max_files_per_request=3,
        allow_duplicate_documents=False,
        language_detection_enabled=False,
        data_directory=tmp_path,
        document_registry_path=tmp_path / "registry.sqlite3",
        query_language_min_chars=3,
        query_language_min_confidence=0.5,
        query_language_min_margin=0.1,
        query_language_fallback="",
        rag_max_question_length=200,
        rag_min_evidence_chunks=1,
        rag_distance_filter_enabled=True,
        rag_max_distance=2.0,
        rag_generation_without_evidence=False,
        rag_evidence_assessment_mode="heuristic",
        rag_evidence_assessment_fallback_enabled=True,
        llm_provider="mock",
        llm_model="test-mock-model",
        llm_api_key="",
        llm_base_url="",
        llm_temperature=0.0,
        llm_timeout_seconds=5.0,
        llm_max_output_tokens=128,
    )


@pytest.fixture
def token_counter() -> WordTokenCounter:
    return WordTokenCounter()


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor()


@pytest.fixture
def chunker(settings: Settings, token_counter: WordTokenCounter) -> TextChunker:
    return TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_counter=token_counter,
    )


@pytest.fixture
def embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(settings, model_factory=lambda: FakeEncoderModel())


@pytest.fixture
def registry(settings: Settings) -> DocumentRegistry:
    return DocumentRegistry(settings.resolved_registry_path)


@pytest.fixture
def store(settings: Settings) -> ChromaStore:
    return ChromaStore(
        persist_directory=settings.resolved_chroma_directory,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_name,
        batch_size=settings.chroma_batch_size,
    )


@pytest.fixture
def ingestion_service(
    settings: Settings,
    processor: DocumentProcessor,
    chunker: TextChunker,
    embedding_service: EmbeddingService,
    store: ChromaStore,
    registry: DocumentRegistry,
) -> IngestionService:
    return IngestionService(
        settings=settings,
        processor=processor,
        chunker=chunker,
        embedding_service=embedding_service,
        store=store,
        registry=registry,
        language_detector=LanguageDetector(enabled=False),
    )


@pytest.fixture
def retrieval_service(
    settings: Settings,
    embedding_service: EmbeddingService,
    store: ChromaStore,
) -> RetrievalService:
    return RetrievalService(settings, embedding_service, store)


# -- Phase 3: RAG fixtures --------------------------------------------------


@pytest.fixture
def language_backend() -> FakeLanguageBackend:
    return FakeLanguageBackend()


@pytest.fixture
def query_language_detector(
    settings: Settings, language_backend: FakeLanguageBackend
) -> QueryLanguageDetector:
    return QueryLanguageDetector(
        enabled=True,
        backend=language_backend,
        fallback_backend=UnavailableLanguageBackend(),
        min_chars=settings.query_language_min_chars,
        min_confidence=settings.query_language_min_confidence,
        min_margin=settings.query_language_min_margin,
        fallback_language=settings.query_language_fallback,
    )


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def mock_generator() -> MockGenerator:
    return MockGenerator()


@pytest.fixture
def evidence_assessor() -> EvidenceAssessor:
    return FixedEvidenceAssessor()


@pytest.fixture
def rag_service(
    settings: Settings,
    retrieval_service: RetrievalService,
    mock_generator: MockGenerator,
    query_language_detector: QueryLanguageDetector,
    prompt_builder: PromptBuilder,
    evidence_assessor: EvidenceAssessor,
) -> RagService:
    return RagService(
        settings=settings,
        retrieval_service=retrieval_service,
        generator=mock_generator,
        language_detector=query_language_detector,
        prompt_builder=prompt_builder,
        evidence_assessor=evidence_assessor,
    )


@pytest.fixture
def rag_result_store(settings: Settings) -> RagResultStore:
    return RagResultStore(
        max_entries=settings.rag_result_cache_size,
        ttl_seconds=settings.rag_result_ttl_seconds,
    )


@pytest.fixture
def rag_verification_service(settings: Settings) -> RagVerificationService:
    return RagVerificationService(
        settings=settings,
        verification_engine=StubVerificationEngine(),
    )


@pytest.fixture
def api_client(
    settings: Settings,
    processor: DocumentProcessor,
    ingestion_service: IngestionService,
    retrieval_service: RetrievalService,
    rag_service: RagService,
    rag_verification_service: RagVerificationService,
    rag_result_store: RagResultStore,
    registry: DocumentRegistry,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_document_processor] = lambda: processor
    app.dependency_overrides[get_document_registry] = lambda: registry
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    app.dependency_overrides[get_rag_verification_service] = (
        lambda: rag_verification_service
    )
    app.dependency_overrides[get_rag_result_store] = lambda: rag_result_store
    app.dependency_overrides[get_vector_store_status] = lambda: ComponentStatus(
        status="ok", detail="fake vector store"
    )
    app.dependency_overrides[get_document_registry_status] = lambda: ComponentStatus(
        status="ok", detail="fake registry"
    )
    app.dependency_overrides[get_embedding_status] = lambda: ComponentStatus(
        status="ok", detail="fake embeddings"
    )
    return TestClient(app)


# -- file builders ----------------------------------------------------------


def make_txt_bytes(text: str, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


def make_pdf_bytes(pages: list[str | None]) -> bytes:
    """Build a small PDF; ``None`` produces a blank page (no text layer)."""
    document = pymupdf.open()
    for content in pages:
        page = document.new_page()
        if content:
            page.insert_text((72, 96), content, fontsize=11)
    data = document.tobytes()
    document.close()
    return data
