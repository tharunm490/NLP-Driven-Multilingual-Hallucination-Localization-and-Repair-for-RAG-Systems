"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_document_registry_status,
    get_embedding_service,
    get_embedding_status,
    get_vector_store_status,
)
from app.core.config import Settings, get_settings
from app.schemas.health import ComponentStatus, HealthResponse
from app.services.embedding_service import EmbeddingService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    vector_store: ComponentStatus = Depends(get_vector_store_status),
    registry: ComponentStatus = Depends(get_document_registry_status),
    embedding: ComponentStatus = Depends(get_embedding_status),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> HealthResponse:
    components = {
        "vector_store": vector_store,
        "document_registry": registry,
        "embedding_model": embedding,
    }
    status = "ok" if all(component.status == "ok" for component in components.values()) else "degraded"
    return HealthResponse(
        status=status,
        version=settings.app_version,
        environment=settings.app_env,
        embedding_model=settings.embedding_model_name,
        embedding_device=embedding_service.device or settings.embedding_device,
        embedding_loaded=embedding_service.is_loaded,
        max_upload_size_mb=settings.max_upload_size_mb,
        max_files_per_request=settings.max_files_per_request,
        components=components,
    )
