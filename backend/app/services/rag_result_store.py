"""Short-lived in-memory storage for completed RAG responses.

The browser only needs a non-sensitive identifier to retrieve the latest Ask
Documents result from another page. Keeping the full answer and evidence in the
backend avoids putting document content in browser storage.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

from app.schemas.rag import RagQueryResponse


class RagResultStore:
    def __init__(
        self,
        *,
        max_entries: int = 32,
        ttl_seconds: float = 3600.0,
        clock=time.monotonic,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl_seconds = max(1.0, ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, RagQueryResponse]] = OrderedDict()

    def put(self, response: RagQueryResponse) -> RagQueryResponse:
        result_id = response.result_id or uuid.uuid4().hex
        stored = response.model_copy(update={"result_id": result_id}, deep=True)
        with self._lock:
            self._purge_expired_locked()
            self._entries[result_id] = (
                self._clock() + self._ttl_seconds,
                stored,
            )
            self._entries.move_to_end(result_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return stored.model_copy(deep=True)

    def get(self, result_id: str) -> RagQueryResponse | None:
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(result_id)
            if entry is None:
                return None
            self._entries.move_to_end(result_id)
            return entry[1].model_copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            result_id
            for result_id, (expires_at, _) in self._entries.items()
            if expires_at <= now
        ]
        for result_id in expired:
            self._entries.pop(result_id, None)
