"""Sentence-aware recursive text chunking with token overlap."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Sequence

from app.services.document_processor import ExtractedPage
from app.services.token_counter import HeuristicTokenCounter, TokenCounter

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？।॥])\s*"
    r"|(?<=[.!?])\s+(?=\S)"
    r"|(?<=[؟۔])\s+(?=\S)"
)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None


class TextChunker:
    """Chunk cleaned pages without crossing page boundaries."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._count = token_counter or HeuristicTokenCounter()

    def count_tokens(self, text: str) -> int:
        return self._count.count(text)

    def chunk_document(
        self,
        pages: Sequence[ExtractedPage],
        document_id: str,
        filename: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        seen_text: set[str] = set()

        for page in pages:
            for text in self._chunk_page_text(page.text):
                normalised = _WHITESPACE.sub(" ", text).strip()
                if not normalised:
                    continue
                fingerprint = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
                if fingerprint in seen_text:
                    logger.debug(
                        "Skipping duplicate chunk for document %s on page %s",
                        document_id,
                        page.page_number,
                    )
                    continue
                seen_text.add(fingerprint)

                chunk_index = len(chunks)
                chunk_id = _stable_chunk_id(document_id, chunk_index, text)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index,
                        text=text,
                        token_count=self._count.count(text),
                        page_number=page.page_number,
                    )
                )
        return chunks

    # -- internals ---------------------------------------------------------

    def _chunk_page_text(self, text: str) -> list[str]:
        units = self._split_units(text)
        if not units:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._count.count(unit)

            if unit_tokens > self.chunk_size:
                # Flush the current chunk, then split the oversized unit itself.
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_oversized_unit(unit))
                continue

            if current and current_tokens + unit_tokens > self.chunk_size:
                chunks.append(" ".join(current))
                tail = self._overlap_tail(current)
                tail_tokens = sum(self._count.count(item) for item in tail)
                if tail and tail_tokens + unit_tokens <= self.chunk_size:
                    current = tail
                    current_tokens = tail_tokens
                else:
                    current = []
                    current_tokens = 0

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(" ".join(current))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = _WHITESPACE.sub(" ", paragraph).strip()
            if not paragraph:
                continue
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)
        return units

    def _overlap_tail(self, units: list[str]) -> list[str]:
        if self.chunk_overlap <= 0:
            return []
        tail: list[str] = []
        total = 0
        for unit in reversed(units):
            unit_tokens = self._count.count(unit)
            if total + unit_tokens > self.chunk_overlap:
                break
            tail.insert(0, unit)
            total += unit_tokens
        return tail

    def _split_oversized_unit(self, unit: str) -> list[str]:
        """Split a single very long unit on word boundaries with overlap."""
        words = unit.split()
        if len(words) <= 1:
            return [unit] if unit.strip() else []

        pieces: list[str] = []
        start = 0
        total_words = len(words)

        while start < total_words:
            end = start
            while end < total_words:
                candidate = " ".join(words[start : end + 1])
                if end > start and self._count.count(candidate) > self.chunk_size:
                    break
                end += 1
            piece = " ".join(words[start:end]).strip()
            if piece:
                pieces.append(piece)
            if end >= total_words:
                break

            if self.chunk_overlap <= 0:
                start = end
                continue

            next_start = end
            while next_start > start:
                suffix = " ".join(words[next_start - 1 : end])
                if self._count.count(suffix) > self.chunk_overlap:
                    break
                next_start -= 1
            start = end if next_start == start else next_start

        return pieces


def _stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}:{chunk_index:06d}:{digest}"
