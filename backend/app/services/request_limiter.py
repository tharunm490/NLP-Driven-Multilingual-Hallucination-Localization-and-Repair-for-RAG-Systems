"""Bounded admission control for expensive local inference requests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from app.core.errors import RequestOverloadedError


class RequestLimiter:
    """A semaphore with a bounded wait and a bounded waiter queue.

    The semaphore prevents concurrent CPU-heavy inference. The waiter counter
    prevents a burst of duplicate requests from creating an unbounded set of
    waiting threads: once the queue is full, callers are rejected immediately
    with an overload response instead of being admitted later.
    """

    def __init__(
        self,
        max_concurrent: int,
        wait_seconds: float,
        max_waiters: int = 2,
    ) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent))
        self._wait_seconds = max(0.0, wait_seconds)
        self._max_concurrent = max(1, max_concurrent)
        self._max_waiters = max(0, max_waiters)
        self._waiters = 0
        self._waiter_lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def waiting(self) -> int:
        with self._waiter_lock:
            return self._waiters

    def acquire(self) -> None:
        if self._semaphore.acquire(blocking=False):
            return

        with self._waiter_lock:
            if self._waiters >= self._max_waiters:
                raise RequestOverloadedError(
                    "The local verification queue is full. Wait for the current "
                    "request to finish and try again."
                )
            self._waiters += 1

        try:
            if self._semaphore.acquire(timeout=self._wait_seconds):
                return
        finally:
            with self._waiter_lock:
                self._waiters = max(0, self._waiters - 1)

        raise RequestOverloadedError(
            "The local verification service is busy with another answer. "
            "Please wait for the current request to finish and try again."
        )

    def release(self) -> None:
        self._semaphore.release()

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()
