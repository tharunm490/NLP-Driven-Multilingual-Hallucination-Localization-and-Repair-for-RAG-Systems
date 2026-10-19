"""Pydantic schemas for the health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    embedding_model: str
    embedding_device: str | None = None
    embedding_loaded: bool | None = None
    max_upload_size_mb: int = 20
    max_files_per_request: int = 5
    components: dict[str, ComponentStatus] = Field(default_factory=dict)
