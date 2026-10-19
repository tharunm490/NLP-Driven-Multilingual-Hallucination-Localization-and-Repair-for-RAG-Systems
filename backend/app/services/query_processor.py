"""Conservative query preprocessing.

The goal is to remove formatting noise without changing the meaning of a
factual question. Punctuation, names, numbers, dates, casing, and the original
language are all preserved. Stop words are never removed and the query is never
translated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.core.errors import QueryValidationError

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXOTIC_SPACES = str.maketrans({"\u00a0": " ", "\u202f": " ", "\u3000": " ", "\t": " "})
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessedQuery:
    """Original and normalized forms of a user query."""

    original: str
    normalized: str

    @property
    def was_modified(self) -> bool:
        return self.original != self.normalized

    @property
    def length(self) -> int:
        return len(self.normalized)


def preprocess_query(raw: str, max_length: int | None = None) -> PreprocessedQuery:
    """Normalize whitespace and formatting while preserving meaning."""
    original = raw if isinstance(raw, str) else str(raw)

    text = unicodedata.normalize("NFC", original)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = text.replace("\f", " ")
    text = text.translate(_EXOTIC_SPACES)

    # A question is treated as a single logical line: collapse all whitespace
    # runs (including newlines) to one regular space and trim the ends.
    text = _WHITESPACE_RUN.sub(" ", text).strip()

    if not text:
        raise QueryValidationError("The question is empty after preprocessing.")
    if max_length is not None and len(text) > max_length:
        raise QueryValidationError(
            f"The question is too long ({len(text)} characters). "
            f"The configured maximum is {max_length}."
        )

    return PreprocessedQuery(original=original, normalized=text)
