"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import documents, health, rag, retrieval
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    settings.resolved_data_directory.mkdir(parents=True, exist_ok=True)
    settings.resolved_chroma_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting %s v%s (env=%s, embedding=%s, collection=%s, llm_provider=%s)",
        app.title,
        settings.app_version,
        settings.app_env,
        settings.embedding_model_name,
        settings.chroma_collection_name,
        settings.llm_provider_normalised or "unconfigured",
    )
    yield
    logger.info("Shutting down %s", app.title)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.normalised_log_level)

    app = FastAPI(
        title="Multilingual Hallucination Localization API",
        description=(
            "Phase 1-3 backend: document ingestion, multilingual embeddings, "
            "vector retrieval, and evidence-grounded question answering. "
            "Hallucination detection, localization, and repair are added in "
            "later phases."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error for %s %s", request.method, request.url.path)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Internal server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check backend logs for details."},
        )

    api_router = APIRouter(prefix="/api")
    api_router.include_router(health.router)
    api_router.include_router(documents.router)
    api_router.include_router(retrieval.router)
    api_router.include_router(rag.router)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": app.title,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
