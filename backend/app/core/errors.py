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


class EvidenceAssessmentError(AppError):
    """The evidence-sufficiency judge could not return a usable decision."""


class RequestOverloadedError(AppError):
    """The bounded local inference queue is full or timed out."""


class VerificationError(AppError):
    """The claim-level verification pipeline could not complete."""


class ClaimExtractionError(VerificationError):
    """Factual claims could not be extracted from an answer."""


class NliError(VerificationError):
    """The natural-language-inference model failed."""


class NliInputTooLong(NliError):
    """A claim-evidence pair exceeds the NLI model's usable context."""


class HallucinationCheckError(AppError):
    """The dedicated hallucination check could not complete."""


class HallucinationModelError(HallucinationCheckError):
    """The local hallucination model could not be loaded or run."""


class RagResultNotFoundError(AppError):
    """A previously completed Ask result is no longer available."""


class NerError(VerificationError):
    """The named-entity recognizer failed."""


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentNotFoundError)
    async def _not_found(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.message})

    @app.exception_handler(RagResultNotFoundError)
    async def _rag_result_not_found(
        _: Request, exc: RagResultNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.message})

    @app.exception_handler(HallucinationModelError)
    async def _hallucination_model(
        _: Request, exc: HallucinationModelError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": exc.message})

    @app.exception_handler(HallucinationCheckError)
    async def _hallucination_check(
        _: Request, exc: HallucinationCheckError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.message})

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

    @app.exception_handler(RequestOverloadedError)
    async def _overloaded(_: Request, exc: RequestOverloadedError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": exc.message},
            headers={"Retry-After": "5"},
        )
