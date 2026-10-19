"""Token counting for chunking.

The tokenizer belonging to the configured embedding model is used when it is
available locally or can be downloaded. If that fails (for example in an
offline test environment), a deterministic character-based estimate is used.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Deterministic fallback used when a model tokenizer is unavailable."""

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii_chars = len(text) - ascii_chars
        estimate = math.ceil(ascii_chars / 4) + math.ceil(non_ascii_chars / 1.5)
        return max(1, estimate)


class HuggingFaceTokenCounter:
    """Counts tokens with the tokenizer of the configured embedding model."""

    def __init__(
        self,
        model_name: str,
        fallback: TokenCounter | None = None,
    ) -> None:
        self._model_name = model_name
        self._fallback = fallback or HeuristicTokenCounter()
        self._tokenizer = None
        self._load_attempted = False
        self._lock = threading.Lock()

    @property
    def using_fallback(self) -> bool:
        return self._tokenizer is None

    def _get_tokenizer(self):
        with self._lock:
            if self._load_attempted:
                return self._tokenizer
            self._load_attempted = True
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name,
                    trust_remote_code=False,
                )
                logger.info(
                    "Loaded tokenizer for %s (chunk sizes measured in model tokens)",
                    self._model_name,
                )
            except Exception as exc:  # noqa: BLE001 - fallback is intentional
                logger.warning(
                    "Could not load tokenizer for %s (%s); using heuristic token counts",
                    self._model_name,
                    exc,
                )
                self._tokenizer = None
            return self._tokenizer

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return self._fallback.count(text)
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tokenizer failed (%s); using heuristic count", exc)
            return self._fallback.count(text)
