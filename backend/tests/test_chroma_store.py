import pytest

from app.core.errors import VectorStoreError
from app.services.text_chunker import DocumentChunk
from app.vectorstore.chroma_store import ChromaStore

EMBEDDING_MODEL = "test/fake-embedding-model"


def vector_for(text: str, dimension: int = 4) -> list[float]:
    values = [float((sum(ord(char) for char in text) + index) % 7 + 1) for index in range(dimension)]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


def make_chunks(document_id: str, texts: list[str], page_numbers: list[int | None] | None = None):
    chunks = []
    for index, text in enumerate(texts):
        page = page_numbers[index] if page_numbers else None
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}:{index:06d}:hash{index}",
                document_id=document_id,
                filename="sample.txt",
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                page_number=page,
            )
        )
    return chunks


def test_add_and_query_round_trip(store: ChromaStore):
    chunks = make_chunks("doc-a", ["alpha beta", "gamma delta"], [1, 2])
    stored = store.add_chunks(chunks, [vector_for(c.text) for c in chunks], EMBEDDING_MODEL)
    assert stored == 2
    assert store.count() == 2
    assert store.embedding_model == EMBEDDING_MODEL
    assert store.embedding_dimension == 4

    results = store.query(vector_for("alpha beta"), top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "alpha beta"
    assert results[0]["document_id"] == "doc-a"
    assert results[0]["page_number"] == 1
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_persists_across_store_instances(settings):
    first = ChromaStore(
        settings.resolved_chroma_directory,
        settings.chroma_collection_name,
        embedding_model=EMBEDDING_MODEL,
        batch_size=8,
    )
    chunks = make_chunks("doc-persist", ["persistent text"])
    first.add_chunks(chunks, [vector_for("persistent text")], EMBEDDING_MODEL)

    second = ChromaStore(
        settings.resolved_chroma_directory,
        settings.chroma_collection_name,
        embedding_model=EMBEDDING_MODEL,
        batch_size=8,
    )
    assert second.count() == 1
    assert second.embedding_model == EMBEDDING_MODEL
    results = second.query(vector_for("persistent text"), top_k=1)
    assert results[0]["text"] == "persistent text"


def test_query_filters_by_document(store: ChromaStore):
    for doc_id, text in (("doc-1", "apple pie"), ("doc-2", "banana bread")):
        chunks = make_chunks(doc_id, [text])
        store.add_chunks(chunks, [vector_for(text)], EMBEDDING_MODEL)

    results = store.query(vector_for("banana bread"), top_k=5, document_ids=["doc-2"])
    assert len(results) == 1
    assert results[0]["document_id"] == "doc-2"


def test_delete_document_removes_chunks(store: ChromaStore):
    chunks = make_chunks("doc-del", ["one", "two"])
    store.add_chunks(chunks, [vector_for("one"), vector_for("two")], EMBEDDING_MODEL)
    assert store.count_document("doc-del") == 2
    deleted = store.delete_document("doc-del")
    assert deleted == 2
    assert store.count() == 0


def test_rejects_embedding_model_change(settings):
    store = ChromaStore(
        settings.resolved_chroma_directory,
        settings.chroma_collection_name,
        embedding_model="model-a",
    )
    store.add_chunks(make_chunks("doc-m", ["text"]), [vector_for("text")], "model-a")
    with pytest.raises(VectorStoreError):
        store.add_chunks(make_chunks("doc-m2", ["other"]), [vector_for("other")], "model-b")


def test_rejects_dimension_mismatch(store: ChromaStore):
    chunks = make_chunks("doc-d", ["text"])
    store.add_chunks(chunks, [vector_for("text", 4)], EMBEDDING_MODEL)
    with pytest.raises(VectorStoreError):
        store.add_chunks(make_chunks("doc-d2", ["other"]), [vector_for("other", 6)], EMBEDDING_MODEL)


def test_rejects_inconsistent_vectors(store: ChromaStore):
    chunks = make_chunks("doc-i", ["a", "b"])
    with pytest.raises(VectorStoreError):
        store.add_chunks(chunks, [vector_for("a")], EMBEDDING_MODEL)


def test_query_dimension_guard(store: ChromaStore):
    chunks = make_chunks("doc-q", ["text"])
    store.add_chunks(chunks, [vector_for("text", 4)], EMBEDDING_MODEL)
    with pytest.raises(VectorStoreError):
        store.query([0.1, 0.2], top_k=1)


def test_empty_collection_query_returns_nothing(store: ChromaStore):
    assert store.query([0.1, 0.2, 0.3, 0.4], top_k=3) == []
