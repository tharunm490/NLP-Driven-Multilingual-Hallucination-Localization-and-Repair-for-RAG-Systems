"""Unicode-aware, offset-preserving claim location.

The locator performs only lossless ``formatting'' normalization: Unicode form,
typographic quotes, dashes, whitespace, markdown, HTML entities, and citation
markers. It never uses semantic similarity or fuzzy matching. Every returned
offset still points into the original, unmodified answer string.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

from app.schemas.verification import TextSpan

_HTML_ENTITY = re.compile(
    r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_CITATION = re.compile(
    r"[\[(【〔]\s*(?:source|evidence|स्रोत|साक्ष्य)s?\s*"
    r"\d+(?:\s*[,;–\-]\s*\d+)*\s*[\])】〕]",
    re.IGNORECASE,
)
_CITATION_SUFFIX = re.compile(
    r"(?:[\s,;]*[\[(【〔]\s*(?:source|evidence|स्रोत|साक्ष्य)s?\s*"
    r"\d+(?:\s*[,;–\-]\s*\d+)*\s*[\])】〕])+[\s.!?।॥]*$",
    re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_CODE = re.compile(r"`+([^`]*)`+")
_MARKDOWN_FENCE = re.compile(
    r"(?ms)^[ \t]{0,3}(?:```|~~~).*?^[ \t]{0,3}(?:```|~~~)[ \t]*$"
)
_MARKDOWN_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_MARKDOWN_LIST = re.compile(
    r"(?m)^[ \t]*(?:[-*+•▪◦‣]|\d{1,3}(?:\.\d{1,3})*[.)]|"
    r"[ivxlcdm]{1,5}[.)])[ \t]+",
    re.IGNORECASE,
)
_MARKDOWN_QUOTE = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
_MARKDOWN_UNDERSCORE = re.compile(r"(?<!\w)_+|_+(?!\w)")
_TERMINAL_CHARS = " \t\r\n.!?;:,।॥"
_NO_SPACE_BEFORE = frozenset(".,;:!?।॥%)]}>")

_SPACE_CHARACTERS = frozenset(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u202f\u205f\u3000"
)
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")
_QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}
_DASH_MAP = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2043": "-",
    "\ufe58": "-",
    "\ufe63": "-",
    "\uff0d": "-",
}


@dataclass(frozen=True)
class NormalizedText:
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]


def normalize_for_matching(text: str) -> NormalizedText:
    """Normalize formatting while keeping a per-character source map."""

    mapped = _decode_entities(text)
    mapped = _canonicalize_characters(mapped)
    mapped = _keep_group(mapped, _MARKDOWN_LINK, 1)
    mapped = _remove_matches(mapped, _MARKDOWN_FENCE)
    mapped = _keep_group(mapped, _MARKDOWN_CODE, 1)
    mapped = _remove_matches(mapped, _CITATION)
    mapped = _remove_matches(mapped, _HTML_TAG)
    mapped = _remove_matches(mapped, _MARKDOWN_HEADING)
    mapped = _remove_matches(mapped, _MARKDOWN_LIST)
    mapped = _remove_matches(mapped, _MARKDOWN_QUOTE)
    mapped = _remove_characters(mapped, frozenset("*`~"))
    mapped = _remove_matches(mapped, _MARKDOWN_UNDERSCORE)
    mapped = _collapse_whitespace(mapped)
    return _strip_edges(mapped)


def locate_claim(
    answer: str, claim_text: str, start_at: int = 0
) -> tuple[int, int] | None:
    """Return original answer offsets for a formatting-equivalent claim.

    ``start_at`` is an original-answer character offset. The first match at or
    after that offset is preferred, but an earlier unique match is still
    accepted so claims returned out of order remain traceable.
    """

    if not answer or not claim_text.strip():
        return None
    normalized_answer = normalize_for_matching(answer)
    normalized_claim = normalize_for_matching(claim_text)
    if not normalized_answer.text or not normalized_claim.text:
        return None

    probes = [normalized_claim.text]
    trimmed = normalized_claim.text.rstrip(_TERMINAL_CHARS)
    if trimmed and trimmed != probes[0]:
        probes.append(trimmed)

    for probe in probes:
        pattern = _bounded_pattern(probe)
        matches = list(
            re.finditer(pattern, normalized_answer.text, re.IGNORECASE)
        )
        if not matches:
            continue
        chosen = next(
            (
                match
                for match in matches
                if normalized_answer.starts[match.start()] >= start_at
            ),
            matches[0],
        )
        start = normalized_answer.starts[chosen.start()]
        end = normalized_answer.ends[chosen.end() - 1]
        start, end = _trim_original_whitespace(answer, start, end)
        if end > start:
            return start, end
    return None


def trim_citation_suffix(
    answer: str, start: int, end: int
) -> tuple[int, int]:
    """Remove trailing ``[Source N]`` markers and surrounding whitespace."""

    value = answer[start:end]
    match = _CITATION_SUFFIX.search(value)
    if match is not None:
        end = start + match.start()
    return _trim_original_whitespace(answer, start, end)


def containing_sentence(
    sentences: list[TextSpan], start: int, end: int
) -> tuple[int, TextSpan]:
    """Pick the sentence with the largest overlap, or the nearest one."""

    def overlap(span: TextSpan) -> int:
        return max(0, min(end, span.end) - max(start, span.start))

    candidates = [
        index for index, span in enumerate(sentences) if overlap(span) > 0
    ]
    if candidates:
        index = max(candidates, key=lambda item: overlap(sentences[item]))
        return index, sentences[index]
    index = min(
        range(len(sentences)),
        key=lambda item: min(
            abs(sentences[item].start - end),
            abs(start - sentences[item].end),
        ),
    )
    return index, sentences[index]


@dataclass
class _MappedText:
    chars: list[str]
    starts: list[int]
    ends: list[int]

    @classmethod
    def empty(cls) -> "_MappedText":
        return cls([], [], [])

    def text(self) -> str:
        return "".join(self.chars)


def _decode_entities(text: str) -> _MappedText:
    mapped = _MappedText.empty()
    position = 0
    while position < len(text):
        match = _HTML_ENTITY.match(text, position)
        if match is None:
            mapped.chars.append(text[position])
            mapped.starts.append(position)
            mapped.ends.append(position + 1)
            position += 1
            continue
        decoded = html.unescape(match.group(0))
        for character in decoded:
            mapped.chars.append(character)
            mapped.starts.append(position)
            mapped.ends.append(match.end())
        position = match.end()
    return mapped


def _canonicalize_characters(mapped: _MappedText) -> _MappedText:
    output = _MappedText.empty()
    for character, start, end in zip(
        mapped.chars, mapped.starts, mapped.ends, strict=True
    ):
        for normalized in unicodedata.normalize("NFKC", character):
            if normalized in _ZERO_WIDTH:
                continue
            if normalized in _SPACE_CHARACTERS:
                normalized = " "
            normalized = _QUOTE_MAP.get(normalized, normalized)
            normalized = _DASH_MAP.get(normalized, normalized)
            output.chars.append(normalized)
            output.starts.append(start)
            output.ends.append(end)
    return output


def _remove_matches(
    mapped: _MappedText, pattern: re.Pattern[str]
) -> _MappedText:
    spans = [match.span() for match in pattern.finditer(mapped.text())]
    if not spans:
        return mapped
    removed: set[int] = set()
    for start, end in spans:
        removed.update(range(start, end))
    return _filter(mapped, lambda index: index not in removed)


def _keep_group(
    mapped: _MappedText, pattern: re.Pattern[str], group: int
) -> _MappedText:
    text = mapped.text()
    removed: set[int] = set()
    matched = False
    for match in pattern.finditer(text):
        matched = True
        group_start, group_end = match.span(group)
        for index in range(match.start(), match.end()):
            if not group_start <= index < group_end:
                removed.add(index)
    if not matched:
        return mapped
    return _filter(mapped, lambda index: index not in removed)


def _remove_characters(
    mapped: _MappedText, characters: frozenset[str]
) -> _MappedText:
    return _filter(mapped, lambda index: mapped.chars[index] not in characters)


def _filter(
    mapped: _MappedText, predicate: Callable[[int], bool]
) -> _MappedText:
    output = _MappedText.empty()
    for index, (character, start, end) in enumerate(
        zip(mapped.chars, mapped.starts, mapped.ends, strict=True)
    ):
        if predicate(index):
            output.chars.append(character)
            output.starts.append(start)
            output.ends.append(end)
    return output


def _collapse_whitespace(mapped: _MappedText) -> _MappedText:
    output = _MappedText.empty()
    pending_space: tuple[int, int] | None = None
    for character, start, end in zip(
        mapped.chars, mapped.starts, mapped.ends, strict=True
    ):
        if character.isspace():
            if pending_space is None:
                pending_space = (start, end)
            else:
                pending_space = (pending_space[0], end)
            continue
        if (
            pending_space is not None
            and output.chars
            and character not in _NO_SPACE_BEFORE
        ):
            output.chars.append(" ")
            output.starts.append(pending_space[0])
            output.ends.append(pending_space[1])
        pending_space = None
        output.chars.append(character)
        output.starts.append(start)
        output.ends.append(end)
    return output


def _strip_edges(mapped: _MappedText) -> NormalizedText:
    start = 0
    end = len(mapped.chars)
    while start < end and mapped.chars[start].isspace():
        start += 1
    while end > start and mapped.chars[end - 1].isspace():
        end -= 1
    return NormalizedText(
        text="".join(mapped.chars[start:end]),
        starts=tuple(mapped.starts[start:end]),
        ends=tuple(mapped.ends[start:end]),
    )


def _bounded_pattern(probe: str) -> str:
    pattern = re.escape(probe)
    if probe[:1].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if probe[-1:].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return pattern


def _trim_original_whitespace(
    answer: str, start: int, end: int
) -> tuple[int, int]:
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    return start, end


def normalized_probe(claim_text: str) -> str:
    """Expose the normalized claim text for diagnostics and tests."""

    return normalize_for_matching(claim_text).text


def iter_normalized_terms(text: str) -> Iterable[str]:
    """Yield normalized whitespace-separated terms for conservative checks."""

    for term in normalize_for_matching(text).text.split():
        if term:
            yield term
