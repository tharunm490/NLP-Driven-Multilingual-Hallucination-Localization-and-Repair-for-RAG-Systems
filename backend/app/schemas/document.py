"""Pydantic schemas for document ingestion endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    size_bytes: int = 0
    content_type: str | None = None
    page_count: int | None = None
    character_count: int | None = None
    token_count: int | None = None
    chunk_count: int = 0
    chunks_stored: int = 0
    language: str | None = None
    embedding_model: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class UploadItemResult(BaseModel):
    document_id: str | None = None
    original_filename: str
    status: Literal["completed", "failed", "rejected"]
    size_bytes: int = 0
    content_type: str | None = None
    page_count: int | None = None
    character_count: int | None = None
    token_count: int | None = None
    chunk_count: int = 0
    chunks_stored: int = 0
    language: str | None = None
    duplicate: bool = False
    error: str | None = None
    message: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadItemResult] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = 0


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_deleted: int | None = None
