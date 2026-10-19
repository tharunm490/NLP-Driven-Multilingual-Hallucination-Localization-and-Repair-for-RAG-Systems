"""Application error types and FastAPI exception handlers."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected application errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DocumentValidationError(AppError):
    """The uploaded file is not acceptable (extension, size, encoding, ...)."""


class DocumentProcessingError(AppError):
    """The document could not be extracted or processed."""


class ImageOnlyPdfError(DocumentProcessingError):
    """A PDF has no extractable text layer (OCR is not implemented)."""


class EmbeddingError(AppError):
    """Embedding generation failed."""


class VectorStoreError(AppError):
    """The vector store could not complete an operation."""


class DocumentNotFoundError(AppError):
    """A requested document does not exist."""


class QueryValidationError(AppError):
    """The RAG question or its parameters are invalid."""


class GenerationConfigurationError(AppError):
    """The generative LLM is not configured (missing provider, key, or model)."""


class GenerationError(AppError):
    """The generative LLM could not produce an answer (timeout, outage, ...)."""


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentNotFoundError)
    async def _not_found(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.message})

    @app.exception_handler(DocumentValidationError)
    async def _validation(_: Request, exc: DocumentValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.message})

    @app.exception_handler(DocumentProcessingError)
    async def _processing(_: Request, exc: DocumentProcessingError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.message})

    @app.exception_handler(EmbeddingError)
    async def _embedding(_: Request, exc: EmbeddingError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(VectorStoreError)
    async def _vector_store(_: Request, exc: VectorStoreError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(QueryValidationError)
    async def _query_validation(_: Request, exc: QueryValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.message})

    @app.exception_handler(GenerationConfigurationError)
    async def _generation_config(_: Request, exc: GenerationConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(GenerationError)
    async def _generation(_: Request, exc: GenerationError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": exc.message})
