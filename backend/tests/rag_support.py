"""Deterministic test doubles for RAG relevance/evidence tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.schemas.retrieval import RetrievedChunk, SearchResponse
from app.services.evidence_assessor import EvidenceAssessment
from evaluation.iso27001_dataset import CHUNKS_BY_ID, FixtureChunk


def make_chunk(
    fixture: FixtureChunk,
    distance: float,
    *,
    filename: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=fixture.chunk_id,
        document_id=fixture.document_id,
        filename=filename or fixture.filename,
        chunk_index=fixture.chunk_index,
        page_number=fixture.page_number,
        language="en",
        text=fixture.text,
        distance=distance,
        similarity=1.0 - distance,
    )


def chunk_by_id(chunk_id: str, distance: float) -> RetrievedChunk:
    return make_chunk(CHUNKS_BY_ID[chunk_id], distance)


class StubRetrievalService:
    """Return a fixed candidate list, optionally keyed by question."""

    def __init__(
        self,
        results: Sequence[RetrievedChunk] | None = None,
        by_question: dict[str, Sequence[RetrievedChunk]] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._by_question = dict(by_question or {})
        self.calls: list[dict] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> SearchResponse:
        self.calls.append(
            {"query": query, "top_k": top_k, "document_ids": document_ids}
        )
        results = list(self._by_question.get(query, self._results))
        return SearchResponse(
            query=query,
            top_k=top_k or len(results) or 1,
            metric="cosine_distance",
            embedding_model="test/stub",
            results=results,
            total=len(results),
        )


class FixedEvidenceAssessor:
    """Return a predetermined evidence decision and record calls."""

    method = "test_stub"

    def __init__(self, assessment: EvidenceAssessment | None = None) -> None:
        self.assessment = assessment or EvidenceAssessment(
            state="sufficient_evidence",
            method=self.method,
            reason="Deterministic test assessment.",
        )
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def assess(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> EvidenceAssessment:
        self.calls.append((question, list(chunks)))
        return self.assessment


def assessment(
    state: str,
    *,
    reason: str = "Deterministic test assessment.",
    supported: Sequence[str] = (),
    missing: Sequence[str] = (),
    premise_contradicted: bool = False,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        state=state,  # type: ignore[arg-type]
        method="test_stub",
        reason=reason,
        supported_aspects=tuple(supported),
        missing_aspects=tuple(missing),
        premise_contradicted=premise_contradicted,
    )
