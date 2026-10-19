import pytest

from app.services.document_processor import ExtractedPage
from app.services.text_chunker import TextChunker
from tests.conftest import WordTokenCounter


def build_chunker(size: int = 10, overlap: int = 2) -> TextChunker:
    return TextChunker(chunk_size=size, chunk_overlap=overlap, token_counter=WordTokenCounter())


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)


def test_empty_text_produces_no_chunks():
    assert build_chunker().chunk_document([], "doc", "f.txt") == []
    assert build_chunker().chunk_document(
        [ExtractedPage(1, "   \n  ")], "doc", "f.txt"
    ) == []


def test_sentence_aware_chunks_have_overlap():
    text = "a. b. c. d. e. f. g. h."
    chunks = build_chunker(size=3, overlap=1).chunk_document(
        [ExtractedPage(1, text)], "doc-1", "f.txt"
    )
    assert [chunk.text for chunk in chunks] == ["a. b. c.", "c. d. e.", "e. f. g.", "g. h."]
    for chunk in chunks:
        assert chunk.token_count <= 3


def test_overlapping_chunks_share_boundary_words():
    text = " ".join(f"token{index}" for index in range(18))
    chunks = build_chunker(size=10, overlap=3).chunk_document(
        [ExtractedPage(1, text)], "doc-2", "f.txt"
    )
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-3:]
    assert chunks[1].text.split()[:3] == first_tail


def test_chunk_ids_are_stable_and_unique():
    text = "one two three four five six seven eight nine ten eleven twelve"
    pages = [ExtractedPage(1, text)]
    first = build_chunker(size=5, overlap=1).chunk_document(pages, "doc-3", "f.txt")
    second = build_chunker(size=5, overlap=1).chunk_document(pages, "doc-3", "f.txt")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_duplicate_chunks_are_dropped():
    chunker = build_chunker(size=20, overlap=0)
    pages = [
        ExtractedPage(1, "Repeated header line."),
        ExtractedPage(2, "Repeated header line."),
    ]
    chunks = chunker.chunk_document(pages, "doc-4", "f.txt")
    assert len(chunks) == 1


def test_chunks_do_not_cross_page_boundaries():
    pages = [
        ExtractedPage(1, "apple banana cherry"),
        ExtractedPage(2, "dog elephant fox"),
    ]
    chunks = build_chunker(size=50, overlap=0).chunk_document(pages, "doc-5", "f.txt")
    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert "apple" in chunks[0].text and "dog" not in chunks[0].text
    assert "dog" in chunks[1].text and "apple" not in chunks[1].text


def test_oversized_single_unit_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = build_chunker(size=10, overlap=2).chunk_document(
        [ExtractedPage(1, text)], "doc-6", "f.txt"
    )
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= 10
    assert chunks[1].text.split()[:2] == chunks[0].text.split()[-2:]


def test_source_order_is_preserved():
    text = " ".join(f"t{i}" for i in range(30))
    chunks = build_chunker(size=8, overlap=0).chunk_document(
        [ExtractedPage(1, text)], "doc-7", "f.txt"
    )
    indices = [int(token[1:]) for chunk in chunks for token in chunk.text.split()]
    assert indices == sorted(indices)
