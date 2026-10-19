r"""Real embedding-model RAG integration test (opt-in, no real LLM call).

Runs the full pipeline with a real embedding model and a deterministic mock
generator, so no API credits are spent:

    $env:RUN_EMBEDDING_INTEGRATION=1
    $env:EMBEDDING_MODEL_NAME="intfloat/multilingual-e5-base"
    .\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s

The default model is BAAI/bge-m3 (~2.2 GB on first download).
"""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.generator import MockGenerator
from app.services.ingestion_service import IngestionService
from app.services.language_detection import LanguageDetector
from app.services.prompt_builder import PromptBuilder
from app.services.query_language_detection import QueryLanguageDetector
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.text_chunker import TextChunker
from app.services.token_counter import HeuristicTokenCounter
from app.vectorstore.chroma_store import ChromaStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1",
        reason="set RUN_EMBEDDING_INTEGRATION=1 to run real-model tests",
    ),
]

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

SAMPLE_TEXT = (
    "The Eiffel Tower is located in Paris, France. It is 330 metres tall and was "
    "completed in 1889. नई दिल्ली भारत की राजधानी है। "
    "தமிழ்நாட்டின் தலைநகரம் சென்னை."
)
QUESTION = "Where is the Eiffel Tower and how tall is it?"


def build_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        embedding_model_name=MODEL_NAME,
        embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=2,
        language_detection_enabled=False,
        chroma_persist_directory=tmp_path / "chroma",
        document_registry_path=tmp_path / "registry.sqlite3",
        data_directory=tmp_path,
        chunk_size=120,
        chunk_overlap=20,
        rag_distance_filter_enabled=False,
        **overrides,
    )


def test_real_rag_pipeline_retrieves_and_grounds(tmp_path):
    settings = build_settings(tmp_path)
    embeddings = EmbeddingService(settings)
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_counter=HeuristicTokenCounter(),
    )
    store = ChromaStore(
        settings.resolved_chroma_directory,
        settings.chroma_collection_name,
        embedding_model=embeddings.model_name,
    )
    registry = DocumentRegistry(settings.resolved_registry_path)
    ingestion = IngestionService(
        settings=settings,
        processor=DocumentProcessor(),
        chunker=chunker,
        embedding_service=embeddings,
        store=store,
        registry=registry,
        language_detector=LanguageDetector(enabled=False),
    )

    result = ingestion.ingest("eiffel.txt", "text/plain", SAMPLE_TEXT.encode("utf-8"))
    assert result.status == "completed", result.error

    generator = MockGenerator()
    rag = RagService(
        settings=settings,
        retrieval_service=RetrievalService(settings, embeddings, store),
        generator=generator,
        language_detector=QueryLanguageDetector(enabled=True, fallback_backend=None),
        prompt_builder=PromptBuilder(),
    )

    response = rag.answer(QUESTION, top_k=3)
    assert response.retrieval.returned_chunks >= 1
    assert response.retrieval.sufficient_evidence is True
    assert response.sources
    assert any("Eiffel" in source.text for source in response.sources)

    assert len(generator.calls) == 1
    user_message = generator.calls[0][1].content
    assert QUESTION in user_message
    assert "[Source 1]" in user_message
    assert "eiffel" in user_message.lower()
