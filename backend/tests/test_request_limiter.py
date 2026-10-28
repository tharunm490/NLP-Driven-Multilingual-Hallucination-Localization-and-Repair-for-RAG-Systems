"""Tests for bounded local-inference admission control."""

from __future__ import annotations

import threading
import time

import pytest

from app.core.errors import RequestOverloadedError
from app.services.request_limiter import RequestLimiter


def test_limiter_rejects_waiter_when_slot_is_held() -> None:
    limiter = RequestLimiter(max_concurrent=1, wait_seconds=0.0)
    limiter.acquire()
    try:
        with pytest.raises(RequestOverloadedError):
            limiter.acquire()
    finally:
        limiter.release()


def test_limiter_releases_slot_after_context_manager() -> None:
    limiter = RequestLimiter(max_concurrent=1, wait_seconds=0.0)

    with limiter.slot():
        with pytest.raises(RequestOverloadedError):
            limiter.acquire()

    limiter.acquire()
    limiter.release()


def test_limiter_enforces_max_concurrency() -> None:
    limiter = RequestLimiter(max_concurrent=2, wait_seconds=0.0)
    limiter.acquire()
    limiter.acquire()
    try:
        with pytest.raises(RequestOverloadedError):
            limiter.acquire()
    finally:
        limiter.release()
        limiter.release()


def test_limiter_rejects_when_waiter_queue_is_full() -> None:
    limiter = RequestLimiter(
        max_concurrent=1,
        wait_seconds=2.0,
        max_waiters=1,
    )
    acquired = threading.Event()
    release = threading.Event()

    def waiter() -> None:
        limiter.acquire()
        acquired.set()
        release.wait(timeout=5.0)
        limiter.release()

    limiter.acquire()
    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while limiter.waiting == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert limiter.waiting == 1

        with pytest.raises(RequestOverloadedError, match="queue is full"):
            limiter.acquire()
    finally:
        limiter.release()
        acquired.wait(timeout=2.0)
        release.set()
        thread.join(timeout=2.0)

    assert limiter.waiting == 0
