"""Persistent document registry backed by SQLite.

ChromaDB only stores chunks. This registry keeps documents that are still
processing or failed, plus aggregate metadata needed by the document list UI.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_DOCUMENT_FIELDS = (
    "document_id",
    "filename",
    "checksum",
    "status",
    "size_bytes",
    "content_type",
    "page_count",
    "character_count",
    "token_count",
    "chunk_count",
    "chunks_stored",
    "language",
    "embedding_model",
    "error",
    "created_at",
    "updated_at",
)

_UPDATABLE_FIELDS = {
    "status",
    "size_bytes",
    "content_type",
    "page_count",
    "character_count",
    "token_count",
    "chunk_count",
    "chunks_stored",
    "language",
    "embedding_model",
    "error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    checksum: str
    status: str
    size_bytes: int
    content_type: str | None
    page_count: int | None
    character_count: int | None
    token_count: int | None
    chunk_count: int
    chunks_stored: int
    language: str | None
    embedding_model: str | None
    error: str | None
    created_at: str
    updated_at: str


class DocumentRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    content_type TEXT,
                    page_count INTEGER,
                    character_count INTEGER,
                    token_count INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    chunks_stored INTEGER NOT NULL DEFAULT 0,
                    language TEXT,
                    embedding_model TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_checksum "
                "ON documents (checksum, status)"
            )

    def create(
        self,
        document_id: str,
        filename: str,
        checksum: str,
        size_bytes: int,
        content_type: str | None,
    ) -> DocumentRecord:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, checksum, status, size_bytes,
                    content_type, chunk_count, chunks_stored, created_at, updated_at
                ) VALUES (?, ?, ?, 'processing', ?, ?, 0, 0, ?, ?)
                """,
                (document_id, filename, checksum, size_bytes, content_type, now, now),
            )
        record = self.get(document_id)
        assert record is not None
        return record

    def update(self, document_id: str, **fields: Any) -> DocumentRecord:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update unknown document fields: {sorted(unknown)}")
        if not fields:
            record = self.get(document_id)
            if record is None:
                raise KeyError(document_id)
            return record

        assignments = ", ".join(f"{field} = ?" for field in fields)
        values = list(fields.values())
        values.append(_utc_now())
        values.append(document_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {assignments}, updated_at = ? WHERE document_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)
        return self.get(document_id)  # type: ignore[return-value]

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return _record_from_row(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
        return int(row["total"]) if row else 0

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
        return cursor.rowcount > 0

    def find_completed_by_checksum(self, checksum: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE checksum = ? AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (checksum,),
            ).fetchone()
        return _record_from_row(row) if row else None


def _record_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(**{field: row[field] for field in _DOCUMENT_FIELDS})
