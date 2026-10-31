"""FastAPI dependency providers.

Each provider is a small function so tests can replace services with fakes via
``app.dependency_overrides`` without loading large models.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.schemas.health import ComponentStatus
from app.services.document_processor import DocumentProcessor
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.evidence_assessor import EvidenceAssessor, build_evidence_assessor
from app.services.generator import Generator, build_generator
from app.services.hallucination_detector import (
    HallucinationDetector,
    build_hallucination_detector,
)
from app.services.hallucination_repair_service import (
    HallucinationRepairService,
    build_hallucination_repair_service,
)
from app.services.hallucination_service import (
    HallucinationCheckService,
    build_hallucination_check_service,
)
from app.services.ingestion_service import IngestionService
from app.services.language_detection import LanguageDetector
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_result_store import RagResultStore
from app.services.rag_service import RagService
from app.services.rag_verification_service import RagVerificationService
from app.services.request_limiter import RequestLimiter
from app.services.retrieval_service import RetrievalService
from app.services.text_chunker import TextChunker
from app.services.token_counter import HuggingFaceTokenCounter
from app.services.verification import VerificationEngine, build_verification_engine
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()


@lru_cache(maxsize=1)
def get_text_chunker() -> TextChunker:
    settings = get_settings()
    counter = HuggingFaceTokenCounter(settings.effective_tokenizer_model_name)
    return TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_counter=counter,
    )


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    return DocumentRegistry(get_settings().resolved_registry_path)


@lru_cache(maxsize=1)
def get_chroma_store() -> ChromaStore:
    settings = get_settings()
    return ChromaStore(
        persist_directory=settings.resolved_chroma_directory,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_name,
        batch_size=settings.chroma_batch_size,
    )


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        settings=settings,
        processor=get_document_processor(),
        chunker=get_text_chunker(),
        embedding_service=get_embedding_service(),
        store=get_chroma_store(),
        registry=get_document_registry(),
        language_detector=LanguageDetector(enabled=settings.language_detection_enabled),
    )


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        settings=get_settings(),
        embedding_service=get_embedding_service(),
        store=get_chroma_store(),
    )


# -- Phase 3: RAG pipeline --------------------------------------------------


@lru_cache(maxsize=1)
def get_query_language_detector() -> QueryLanguageDetector:
    settings = get_settings()
    return QueryLanguageDetector(
        enabled=settings.language_detection_enabled,
        min_chars=settings.query_language_min_chars,
        min_confidence=settings.query_language_min_confidence,
        min_margin=settings.query_language_min_margin,
        fallback_language=settings.query_language_fallback,
    )


@lru_cache(maxsize=1)
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return build_generator(get_settings())


@lru_cache(maxsize=1)
def get_evidence_assessor() -> EvidenceAssessor:
    return build_evidence_assessor(get_settings(), get_generator())


@lru_cache(maxsize=1)
def get_request_limiter() -> RequestLimiter:
    settings = get_settings()
    return RequestLimiter(
        max_concurrent=settings.verification_max_concurrent_requests,
        wait_seconds=settings.verification_queue_wait_seconds,
        max_waiters=settings.verification_max_queued_requests,
    )


@lru_cache(maxsize=1)
def get_verification_engine() -> VerificationEngine:
    settings = get_settings()
    return build_verification_engine(
        settings=settings,
        embedder=get_embedding_service(),
        generator=get_generator(),
        detector=get_query_language_detector(),
    )


@lru_cache(maxsize=1)
def get_hallucination_detector() -> HallucinationDetector:
    return build_hallucination_detector(get_settings())


@lru_cache(maxsize=1)
def get_hallucination_check_service() -> HallucinationCheckService:
    return build_hallucination_check_service(
        get_settings(), get_generator(), get_query_language_detector()
    )


@lru_cache(maxsize=1)
def get_hallucination_repair_service() -> HallucinationRepairService:
    hallucination_service = get_hallucination_check_service()
    return build_hallucination_repair_service(
        settings=get_settings(),
        verification_engine=get_verification_engine(),
        rag_verification_service=get_rag_verification_service(),
        hallucination_service=hallucination_service,
        detector=hallucination_service.detector,
        generator=get_generator(),
    )


@lru_cache(maxsize=1)
def get_rag_result_store() -> RagResultStore:
    settings = get_settings()
    return RagResultStore(
        max_entries=settings.rag_result_cache_size,
        ttl_seconds=settings.rag_result_ttl_seconds,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(
        settings=get_settings(),
        retrieval_service=get_retrieval_service(),
        generator=get_generator(),
        language_detector=get_query_language_detector(),
        prompt_builder=get_prompt_builder(),
        evidence_assessor=get_evidence_assessor(),
    )


@lru_cache(maxsize=1)
def get_rag_verification_service() -> RagVerificationService:
    return RagVerificationService(
        settings=get_settings(),
        verification_engine=get_verification_engine(),
    )


# -- health component probes ------------------------------------------------


def get_vector_store_status() -> ComponentStatus:
    try:
        store = get_chroma_store()
        heartbeat = store.heartbeat()
        return ComponentStatus(
            status="ok",
            detail=(
                f"collection='{store.collection_name}', chunks={store.count()}, "
                f"heartbeat={heartbeat}"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - health must not raise
        logger.warning("Vector store health check failed: %s", exc)
        return ComponentStatus(status="error", detail=str(exc))


def get_document_registry_status() -> ComponentStatus:
    try:
        registry = get_document_registry()
        return ComponentStatus(status="ok", detail=f"documents={registry.count()}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Document registry health check failed: %s", exc)
        return ComponentStatus(status="error", detail=str(exc))


def get_embedding_status() -> ComponentStatus:
    settings = get_settings()
    service = get_embedding_service()
    loaded = "loaded" if service.is_loaded else "not loaded yet (lazy)"
    return ComponentStatus(
        status="ok",
        detail=f"model='{settings.embedding_model_name}', device='{settings.embedding_device}', {loaded}",
    )
