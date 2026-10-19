"""Document upload, listing, detail, and deletion endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_document_registry, get_document_processor, get_ingestion_service
from app.core.config import Settings, get_settings
from app.core.errors import DocumentNotFoundError, DocumentValidationError
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadItemResult,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.document_registry import DocumentRecord, DocumentRegistry
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_CHUNK_SIZE = 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="PDF or TXT files to ingest")],
    settings: Settings = Depends(get_settings),
    service: IngestionService = Depends(get_ingestion_service),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> UploadResponse:
    """Upload and process one or more documents.

    The HTTP request succeeds even when individual files are rejected or fail
    processing; each file gets its own structured result entry.
    """
    if not files:
        raise DocumentValidationError("No files were provided.")
    if len(files) > settings.max_files_per_request:
        raise DocumentValidationError(
            f"Too many files: at most {settings.max_files_per_request} per request."
        )

    results: list[UploadItemResult] = []
    for upload in files:
        raw_name = upload.filename
        try:
            processor.validate(raw_name, upload.content_type)
        except DocumentValidationError as exc:
            results.append(
                UploadItemResult(
                    original_filename=_display_filename(raw_name),
                    status="rejected",
                    content_type=upload.content_type,
                    error=exc.message,
                )
            )
            await upload.close()
            continue

        try:
            data = await _read_limited(upload, settings.max_upload_size_bytes)
        except DocumentValidationError as exc:
            results.append(
                UploadItemResult(
                    original_filename=_display_filename(raw_name),
                    status="rejected",
                    content_type=upload.content_type,
                    error=exc.message,
                )
            )
            await upload.close()
            continue

        if not data:
            results.append(
                UploadItemResult(
                    original_filename=_display_filename(raw_name),
                    status="rejected",
                    content_type=upload.content_type,
                    error="The uploaded file is empty.",
                )
            )
            await upload.close()
            continue

        result = await run_in_threadpool(
            service.ingest, raw_name, upload.content_type, data
        )
        results.append(result)
        await upload.close()

    return UploadResponse(results=results)


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
def list_documents(
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentListResponse:
    records = registry.list()
    return DocumentListResponse(
        documents=[_to_document_info(record) for record in records],
        total=len(records),
    )


@router.get("/{document_id}", response_model=DocumentInfo)
def get_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_document_registry),
) -> DocumentInfo:
    record = registry.get(document_id)
    if record is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
    return _to_document_info(record)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DeleteDocumentResponse:
    chunks_deleted = service.delete_document(document_id)
    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
        chunks_deleted=chunks_deleted,
    )


# -- helpers ----------------------------------------------------------------


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while True:
        chunk = await upload.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise DocumentValidationError(
                f"File exceeds the maximum upload size of "
                f"{max_bytes // (1024 * 1024)} MB."
            )
    return bytes(data)


def _display_filename(raw_name: str | None) -> str:
    if not raw_name:
        return "unnamed"
    name = raw_name.replace("\\", "/").split("/")[-1].strip()
    return name or "unnamed"


def _to_document_info(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,  # type: ignore[arg-type]
        size_bytes=record.size_bytes,
        content_type=record.content_type,
        page_count=record.page_count,
        character_count=record.character_count,
        token_count=record.token_count,
        chunk_count=record.chunk_count,
        chunks_stored=record.chunks_stored,
        language=record.language,
        embedding_model=record.embedding_model,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
