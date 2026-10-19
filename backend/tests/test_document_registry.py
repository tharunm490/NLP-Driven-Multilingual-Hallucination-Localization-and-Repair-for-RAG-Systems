import pytest

from app.services.document_registry import DocumentRegistry


def test_create_update_and_get(registry: DocumentRegistry):
    record = registry.create("doc-1", "file.txt", "checksum-1", 100, "text/plain")
    assert record.status == "processing"
    assert record.chunk_count == 0

    updated = registry.update(
        "doc-1",
        status="completed",
        chunk_count=3,
        chunks_stored=3,
        character_count=1234,
        token_count=300,
        language="hi",
        embedding_model="test/fake",
    )
    assert updated.status == "completed"
    assert updated.chunk_count == 3
    assert updated.language == "hi"
    assert registry.get("doc-1").chunks_stored == 3


def test_update_rejects_unknown_fields(registry: DocumentRegistry):
    registry.create("doc-2", "file.txt", "checksum-2", 10, "text/plain")
    with pytest.raises(ValueError):
        registry.update("doc-2", not_a_field="x")


def test_list_returns_newest_first(registry: DocumentRegistry):
    registry.create("doc-a", "a.txt", "ca", 1, "text/plain")
    registry.create("doc-b", "b.txt", "cb", 1, "text/plain")
    ids = [record.document_id for record in registry.list()]
    assert set(ids) == {"doc-a", "doc-b"}
    assert registry.count() == 2


def test_delete(registry: DocumentRegistry):
    registry.create("doc-c", "c.txt", "cc", 1, "text/plain")
    assert registry.delete("doc-c") is True
    assert registry.get("doc-c") is None
    assert registry.delete("doc-c") is False


def test_find_completed_by_checksum(registry: DocumentRegistry):
    registry.create("doc-d", "d.txt", "shared", 1, "text/plain")
    assert registry.find_completed_by_checksum("shared") is None
    registry.update("doc-d", status="completed")
    found = registry.find_completed_by_checksum("shared")
    assert found is not None
    assert found.document_id == "doc-d"


def test_persistence_across_instances(registry: DocumentRegistry, settings):
    registry.create("doc-e", "e.txt", "ce", 1, "text/plain")
    reopened = DocumentRegistry(settings.resolved_registry_path)
    assert reopened.get("doc-e") is not None
