"""Offset-preserving multilingual text preparation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.schemas.verification import TextSpan

_TERMINATORS = frozenset(".!?।॥")
_BOUNDARY = re.compile(r"[.!?।॥]+[\"'”’)\]]*|\n+")
_WORD_BEFORE_PERIOD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_INNER_PUNCTUATION = frozenset("'’.-")
_LIST_LINE_START = re.compile(
    r"(?:[-*+•‣–—]\s+|\d{1,3}(?:\.\d{1,3})*[.)]\s+|[A-Za-z][.)]\s+|"
    r"\(\d{1,3}\)\s+|\([A-Za-z]\)\s+)"
)
_SECTION_LINE_START = re.compile(r"\d{1,3}(?:\.\d{1,3})*\s+\S")

_ABBREVIATIONS = frozenset(
    {
        "approx",
        "co",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "prof",
        "sr",
        "st",
        "vs",
    }
)


@dataclass(frozen=True)
class CanonicalText:
    original: str
    text: str
    source_starts: tuple[int, ...]
    source_ends: tuple[int, ...]

    def source_span(self, start: int, end: int) -> TextSpan:
        start = max(0, min(start, len(self.text)))
        end = max(start, min(end, len(self.text)))
        if start == end:
            position = (
                self.source_starts[start]
                if start < len(self.source_starts)
                else len(self.original)
            )
            return TextSpan(start=position, end=position, text="")
        source_start = self.source_starts[start]
        source_end = self.source_ends[end - 1]
        return TextSpan(
            start=source_start,
            end=source_end,
            text=self.original[source_start:source_end],
        )

    def trimmed_span(self, start: int, end: int) -> TextSpan | None:
        while start < end and self.text[start].isspace():
            start += 1
        while end > start and self.text[end - 1].isspace():
            end -= 1
        if start >= end:
            return None
        return self.source_span(start, end)


def canonicalize(text: str) -> CanonicalText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char)
        for output in normalized:
            chars.append(output)
            starts.append(index)
            ends.append(index + 1)
    return CanonicalText(
        original=text,
        text="".join(chars),
        source_starts=tuple(starts),
        source_ends=tuple(ends),
    )


def normalized_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def split_sentences(
    text: str, *, join_wrapped_lines: bool = False
) -> list[TextSpan]:
    canonical = canonicalize(text)
    content = canonical.text
    spans: list[TextSpan] = []
    cursor = 0

    for match in _BOUNDARY.finditer(content):
        if _is_abbreviation_period(content, cursor, match.start(), match.end()):
            continue
        if (
            join_wrapped_lines
            and "\n" in match.group()
            and _is_wrapped_line_break(content, match.start(), match.end())
        ):
            continue
        span = canonical.trimmed_span(cursor, match.end())
        if span is not None:
            spans.append(span)
        cursor = match.end()

    tail = canonical.trimmed_span(cursor, len(content))
    if tail is not None:
        spans.append(tail)
    return spans


def _is_wrapped_line_break(text: str, start: int, end: int) -> bool:
    if text[start:end].count("\n") > 1:
        return False
    before = start - 1
    while before >= 0 and text[before].isspace():
        before -= 1
    after = end
    while after < len(text) and text[after].isspace():
        after += 1
    if before < 0 or after >= len(text):
        return False
    if text[before] in _TERMINATORS or text[before] in ":;":
        return False
    line_start = text.rfind("\n", 0, start) + 1
    if _is_heading_line(text[line_start:before+1]):
        return False
    next_line = text[after : after + 160]
    if _LIST_LINE_START.match(next_line) or _SECTION_LINE_START.match(next_line):
        return False
    return True


def _is_heading_line(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 80:
        return False
    words = value.split()
    if len(words) > 8:
        return False
    if _SECTION_LINE_START.match(value):
        return True
    alphabetic = [word for word in words if any(char.isalpha() for char in word)]
    if not alphabetic:
        return False
    if all(word.isupper() for word in alphabetic):
        return True
    return all(word[:1].isupper() for word in alphabetic)


def word_spans(text: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    start: int | None = None
    for index, character in enumerate(text):
        if _is_word_character(character):
            if start is None:
                start = index
            continue
        if (
            start is not None
            and character in _INNER_PUNCTUATION
            and index + 1 < len(text)
            and _is_word_character(text[index + 1])
        ):
            continue
        if start is not None:
            spans.append(TextSpan(start=start, end=index, text=text[start:index]))
            start = None
    if start is not None:
        spans.append(TextSpan(start=start, end=len(text), text=text[start:]))
    return spans


def content_terms(text: str) -> list[str]:
    terms: list[str] = []
    for span in word_spans(text.casefold()):
        token = normalized_text(span.text)
        if any(character.isalnum() for character in token):
            terms.append(token)
    return terms


def _is_word_character(character: str) -> bool:
    if character.isalnum() or character == "_":
        return True
    return unicodedata.category(character).startswith("M")


def _is_abbreviation_period(
    text: str, sentence_start: int, period_start: int, boundary_end: int
) -> bool:
    if text[period_start] != ".":
        return False
    if boundary_end < len(text) and text[boundary_end].isalnum():
        return True
    before = text[period_start - 1] if period_start > 0 else ""
    after = text[boundary_end] if boundary_end < len(text) else ""
    if before.isdigit() and after.isdigit():
        return True
    if period_start > 0 and before.isdigit():
        prefix = text[max(sentence_start, period_start - 2) : period_start]
        if prefix.endswith("."):
            return False
    preceding = text[sentence_start:period_start]
    match = _WORD_BEFORE_PERIOD.search(preceding)
    if not match:
        return False
    word = match.group(1).casefold()
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha() and word.isupper():
        return True
    return False
