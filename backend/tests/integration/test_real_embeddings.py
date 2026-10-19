r"""Real embedding-model integration test (opt-in).

Run with the downloaded model of your choice:

    $env:RUN_EMBEDDING_INTEGRATION=1
    $env:EMBEDDING_MODEL_NAME="intfloat/multilingual-e5-base"
    .\.venv\Scripts\python.exe -m pytest tests/integration -m integration -s

The default model is BAAI/bge-m3 (~2.2 GB on first download).
"""

from __future__ import annotations

import math
import os

import pytest

from app.core.config import Settings
from app.services.document_processor import ExtractedPage
from app.services.embedding_service import EmbeddingService
from app.services.text_chunker import TextChunker
from app.services.token_counter import HeuristicTokenCounter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1",
        reason="set RUN_EMBEDDING_INTEGRATION=1 to run real-model tests",
    ),
]

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")


def build_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        embedding_model_name=MODEL_NAME,
        embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=2,
        language_detection_enabled=False,
        **overrides,
    )


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_real_model_generates_consistent_embeddings():
    service = EmbeddingService(build_settings())
    passages = [
        "The Eiffel Tower is located in Paris and is 330 metres tall.",
        "नई दिल्ली भारत की राजधानी है।",
    ]
    vectors = service.embed_documents(passages)
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) == service.dimension
    query = service.embed_query("Where is the Eiffel Tower?")
    english_similarity = cosine(query, vectors[0])
    hindi_similarity = cosine(query, vectors[1])
    assert english_similarity > hindi_similarity


def test_real_model_chunk_and_retrieve(tmp_path):
    settings = build_settings(
        chroma_persist_directory=tmp_path / "chroma",
        document_registry_path=tmp_path / "registry.sqlite3",
        data_directory=tmp_path,
        chunk_size=120,
        chunk_overlap=20,
    )
    from app.services.document_registry import DocumentRegistry
    from app.vectorstore.chroma_store import ChromaStore

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

    document_id = "integration-doc"
    registry.create(document_id, "sample.txt", "checksum", 100, "text/plain")
    pages = [
        ExtractedPage(
            1,
            "Paris is the capital of France. The Eiffel Tower is in Paris. "
            "The Louvre museum is also in Paris.",
        )
    ]
    chunks = chunker.chunk_document(pages, document_id, "sample.txt")
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    stored = store.add_chunks(chunks, vectors, embeddings.model_name)
    assert stored == len(chunks)

    query_vector = embeddings.embed_query("capital of France")
    results = store.query(query_vector, top_k=1)
    assert results
    assert "Paris" in results[0]["text"]
