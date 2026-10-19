"""Conservative text cleaning for extracted documents.

This module intentionally avoids stemming, stop-word removal, translation,
punctuation stripping, or any destructive normalisation that could remove
factual information from multilingual text.
"""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """Normalise whitespace and extraction artefacts while preserving meaning.

    Punctuation, numbers, dates, names, casing, and sentence boundaries are
    preserved. Non-breaking and zero-width characters are removed or replaced.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Normalise line endings first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove invisible/control artefacts introduced by PDF extraction.
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\f", "\n")

    # Normalise exotic spaces to a regular space.
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u3000", " ")
    text = text.replace("\t", " ")

    # Collapse runs of horizontal whitespace, but keep line breaks.
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)

    # Strip trailing spaces on lines and leading/trailing blank lines.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    # Collapse two or more blank lines into a single blank line (paragraph break).
    text = _BLANK_LINES.sub("\n\n", text)
    return text


def has_meaningful_text(text: str) -> bool:
    """Return True if the text contains at least one letter or number."""
    return any(char.isalnum() for char in text)
