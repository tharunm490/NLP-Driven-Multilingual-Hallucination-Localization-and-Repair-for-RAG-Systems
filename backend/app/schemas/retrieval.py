"""Pydantic schemas for retrieval endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document IDs to restrict the search to.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    language: str | None = None
    text: str
    distance: float = Field(
        description="ChromaDB cosine distance. Lower is more similar; it is not a probability."
    )
    similarity: float = Field(
        description="1 - cosine distance. Range is approximately -1..1; it is not a confidence."
    )


class SearchResponse(BaseModel):
    query: str
    top_k: int
    metric: str = "cosine_distance"
    embedding_model: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    total: int = 0
