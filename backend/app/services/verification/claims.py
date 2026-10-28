"""Local factual claim extraction with offset-preserving sentence spans."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import Settings
from app.schemas.verification import DetectedLanguage, ExtractedClaim
from app.services.verification.text import split_sentences

_LEADING_PREFIX = re.compile(
    r"^\s*(?:#{1,6}\s+|[-*+•▪◦‣]\s+|\d{1,3}(?:\.\d{1,3})*[.)]\s+|[ivxlcdm]{1,5}[.)]\s+)",
    re.IGNORECASE,
)
_LIST_PREFIX = re.compile(
    r"^\s*(?:[-*+•▪◦‣]\s+|\d{1,3}(?:\.\d{1,3})*[.)]\s+|[ivxlcdm]{1,5}[.)]\s+)",
    re.IGNORECASE,
)
_LIST_MARKER_ONLY = re.compile(
    r"^\s*(?:\d{1,3}(?:\.\d{1,3})*[.)]|[ivxlcdm]{1,5}[.)])\s*$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(
    r"^\s{0,3}(?P<hashes>#{1,6})(?:\s+(?P<body>.*?)\s*#*\s*)?$"
)
_BOLD_HEADING = re.compile(
    r"^\s*(?:\*\*|__)(?P<body>[^*_].+?[^.!?。！？])(?:\*\*|__)\s*:?\s*$"
)
_STRUCTURED_LIST_ITEM = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>[-*+•▪◦‣]|\d{1,3}(?:\.\d{1,3})*[.)]|"
    r"[ivxlcdm]{1,5}[.)])"
    r"[ \t]+(?P<body>.+)$",
    re.IGNORECASE,
)
_LIST_LABEL_SEPARATOR = re.compile(
    r"\s+(?:[–—]|-)\s+|\s*[：:]\s+"
)
_SECTION_LABEL = re.compile(
    r"^\s*(?:\*\*|__)?(?P<label>[^*_:]{1,80}?)(?:\*\*|__)?\s*:\s*"
)
_DISCOURSE_PREFIX = re.compile(
    r"^\s*(?:in\s+(?:summary|conclusion|addition)|overall|moreover|"
    r"furthermore|however|therefore|thus|consequently|finally|first(?:ly)?|"
    r"second(?:ly)?|third(?:ly)?|for\s+example|for\s+instance)\s*,?\s+",
    re.IGNORECASE,
)
_META_PREFIX = re.compile(
    r"^\s*(?:sources?|references?|citations?|bibliography|figure|table)\s*:",
    re.IGNORECASE,
)
_TRAILING_CITATION = re.compile(
    r"(?:[\s,;]*\[(?:source|evidence)s?\s*\d+(?:\s*,\s*\d+)*\])+"
    r"(?:\s*[.!?。！？।॥])?\s*$",
    re.IGNORECASE,
)
_EMPHASIS_EDGE = re.compile(r"^\s*(?:[*_`]{1,3})\s*|\s*(?:[*_`]{1,3})\s*$")
_EMPHASIS_WRAPPED = re.compile(r"^(?:\*{2,3}|_{2,3})(.+?)(?:\*{2,3}|_{2,3})$")
_QUESTION_START = re.compile(
    r"^(?:what|why|how|when|where|who|which|whose|do|does|did|is|are|was|were|"
    r"can|could|should|would|will)\b",
    re.IGNORECASE,
)
_TITLE_LEAD = re.compile(
    r"^(?:introduction|overview|summary|conclusion|purpose|scope|objectives?|"
    r"benefits?|key|main|components?|features?|challenges?|risks?|roles?|"
    r"responsibilities?|best practices?|types?|examples?|steps?|process|approach)\b",
    re.IGNORECASE,
)
_TRAILING_LEAD_IN = re.compile(
    r"\b(?:is|are|was|were|includes?|comprises?|consists?\s+of|involves?|"
    r"covers?)\s*$",
    re.IGNORECASE,
)
_INTRO_PATTERNS = (
    re.compile(r"^(?:in|within) this (?:section|answer|response|document|guide)\b", re.IGNORECASE),
    re.compile(
        r"^this (?:section|answer|response|guide|document) "
        r"(?:provides|gives|offers|presents|explains|describes|outlines|covers|"
        r"discusses|summarizes|introduces|is|will|was)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^the following (?:section|sections|list|items?|points?|paragraphs?|"
        r"text|table|figure|steps?|statements?|claims?|guidelines?|examples?|"
        r"benefits?|objectives?|principles?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:here|below|above) (?:is|are|you will find|we (?:provide|describe|"
        r"explain|outline|list))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^as (?:mentioned|noted|described|explained|shown|discussed|stated) "
        r"(?:above|below|earlier|later)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:let us|let's)\b", re.IGNORECASE),
)
_CLAUSE_SEPARATOR = re.compile(
    r"\s*;\s*|\s*,\s*(?:and|but|yet|whereas)\s+|\s+(?:and|but|yet|whereas)\s+",
    re.IGNORECASE,
)
_MODAL_LEAD_IN = re.compile(
    r"\b(?:must|should|shall|can|could|will|may|would|to)\s*$",
    re.IGNORECASE,
)
_ANAPHORIC_SUBJECTS = frozenset(
    {"it", "this", "that", "they", "these", "those", "he", "she", "we", "i", "you"}
)
_PREPOSITIONAL_STARTERS = frozenset(
    {
        "at",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "through",
        "under",
        "upon",
        "via",
        "with",
        "within",
        "without",
    }
)
_CONTINUATION_STARTERS = frozenset(
    {
        "and",
        "but",
        "or",
        "so",
        "yet",
        "while",
        "whereas",
        "which",
        "that",
        "who",
        "whose",
        "whom",
        "to",
        "by",
        "with",
        "through",
        "via",
        "including",
        "such",
        "as",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "upon",
        "at",
        "because",
        "if",
        "when",
        "where",
        "although",
        "though",
        "unless",
        "during",
        "after",
        "before",
        "under",
        "within",
        "without",
    }
)
_DETERMINERS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "my",
        "each",
        "every",
        "all",
        "any",
        "some",
        "no",
    }
)
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_TERMINAL_PUNCTUATION = (".", "!", "?", "。", "！", "？", "।", "॥")
_CLOSING_PUNCTUATION = "\"'”’)]}"
_AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
_IRREGULAR_VERBS = frozenset(
    {
        "became",
        "become",
        "began",
        "begun",
        "built",
        "came",
        "come",
        "found",
        "gave",
        "given",
        "goes",
        "gone",
        "got",
        "gotten",
        "held",
        "kept",
        "led",
        "made",
        "met",
        "ran",
        "said",
        "sought",
        "stood",
        "taken",
        "took",
        "understood",
        "went",
        "written",
        "wrote",
    }
)
_VERB_BASES = frozenset(
    """
    accept achieve act add address adhere adopt advise aim align allow analyse analyze
    apply argue assess assign assist attain audit avoid base become begin belong build
    call carry cause change check choose collaborate combine come communicate compare comply
    comprise conclude conduct confirm conform consider consist contain continue
    contribute control coordinate cover create define deliver demonstrate depend deploy
    describe design detect determine develop differ disclose discuss divide do document
    emphasize enable encourage enforce ensure entail establish evaluate examine exist
    expand explain facilitate focus follow form found function give govern guide handle
    happen have help highlight identify implement improve include incorporate increase
    indicate influence inform integrate introduce involve keep lead limit list maintain
    make manage mandate mean meet mention mitigate monitor need note notify obtain offer
    operate outline participate perform permit plan preserve prevent prioritize produce
    prohibit promote protect provide publish receive recommend record reduce refer
    regulate relate rely remain replace report require respond retain review run
    safeguard say seek set specify state store suggest support supersede take target
    track train treat undergo use validate verify work
    """.split()
)
_ENGLISH_FUNCTION_WORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her his how if in into is it its may might must not of on or our should
    that the their them there these they this those to was we were what when where
    which who will with would you your
    """.split()
)


@dataclass(frozen=True)
class ClaimExtractionResult:
    claims: tuple[ExtractedClaim, ...]
    method: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SentenceCandidate:
    index: int
    text: str
    start: int
    end: int
    is_heading: bool = False
    heading_level: int = 0
    is_list_item: bool = False
    list_indent: int = 0
    list_item_start: int = -1
    subject_hint: str | None = None


@dataclass(frozen=True)
class _CleanedSpan:
    text: str
    start: int
    end: int
    is_list_item: bool = False
    is_heading: bool = False
    heading_level: int = 0
    list_indent: int = 0
    list_item_start: int = -1


@runtime_checkable
class ClaimExtractor(Protocol):
    def extract(self, answer: str) -> ClaimExtractionResult: ...


class HeuristicClaimExtractor:
    method = "heuristic"

    def __init__(self, max_claims: int = 20) -> None:
        self._max_claims = max(1, max_claims)

    def extract(self, answer: str) -> ClaimExtractionResult:
        claims: list[ExtractedClaim] = []
        warnings: list[str] = []
        seen: set[str] = set()
        limit_reached = False
        pending_lead_in: str | None = None
        current_heading: str | None = None
        current_heading_level = 0
        list_contexts: list[tuple[int, str]] = []
        last_claim_sentence_index: int | None = None
        last_list_item_start = -1
        for sentence in _sentences(answer):
            cleaned = _clean_sentence(sentence)
            if cleaned is None:
                continue
            if cleaned.is_heading:
                current_heading = cleaned.text or None
                current_heading_level = cleaned.heading_level
                pending_lead_in = None
                list_contexts.clear()
                continue
            if _is_lead_in(cleaned.text):
                pending_lead_in = cleaned.text.rstrip(":：").strip()
                continue
            if not cleaned.is_list_item and _is_heading_like(
                cleaned.text, cleaned.text
            ):
                current_heading = cleaned.text
                current_heading_level = 0
                pending_lead_in = None
                list_contexts.clear()
                continue

            same_list_item = (
                cleaned.is_list_item
                and cleaned.list_item_start == last_list_item_start
            )
            parent = (
                _nested_parent(list_contexts, cleaned.list_indent)
                if cleaned.is_list_item and not same_list_item
                else None
            )
            if (
                same_list_item
                and claims
                and _starts_with_anaphoric_subject(cleaned.text)
            ):
                parent = claims[-1].hypothesis or claims[-1].claim
            item_label: str | None = None
            semantic_text = cleaned.text
            if cleaned.is_list_item:
                label_body = _split_list_label_body(cleaned.text)
                if label_body is not None:
                    item_label, semantic_text = label_body
            reconstructable = cleaned.is_list_item and (
                _can_reconstruct(
                    semantic_text,
                    lead_in=pending_lead_in,
                    heading=current_heading,
                    parent=parent,
                )
                or (
                    item_label is not None
                    and _has_independent_subject(semantic_text)
                )
            )
            if not _looks_factual(semantic_text) and not reconstructable:
                if not cleaned.is_list_item:
                    pending_lead_in = None
                else:
                    _push_list_context(
                        list_contexts, cleaned.list_indent, semantic_text
                    )
                continue

            if (
                claims
                and not cleaned.is_list_item
                and last_claim_sentence_index is not None
                and sentence.index == last_claim_sentence_index + 1
                and _starts_with_anaphoric_subject(cleaned.text)
            ):
                previous = claims[-1]
                claims[-1] = previous.model_copy(
                    update={
                        "sentence": (
                            f"{previous.sentence.rstrip()} {sentence.text}"
                        ),
                        "sentence_end": sentence.end,
                        "claim": (
                            f"{previous.claim.rstrip()} {cleaned.text}"
                        ),
                        "hypothesis": (
                            f"{previous.hypothesis.rstrip()} {cleaned.text}"
                        ),
                        "claim_end": cleaned.end,
                    }
                )
                last_claim_sentence_index = sentence.index
                pending_lead_in = None
                continue

            span = _SentenceCandidate(
                index=sentence.index,
                text=semantic_text,
                start=cleaned.start,
                end=cleaned.end,
                is_list_item=cleaned.is_list_item,
                list_indent=cleaned.list_indent,
                list_item_start=cleaned.list_item_start,
            )
            lead_in = pending_lead_in if cleaned.is_list_item else None
            heading = current_heading if cleaned.is_list_item else None
            item_claims: list[ExtractedClaim] = []
            fragments = [span] if item_label is not None else _split_compound(span)
            for fragment in fragments:
                if not _looks_factual(fragment.text) and not (
                    (reconstructable or fragment.subject_hint)
                    and (
                        _can_reconstruct(
                            fragment.text,
                            lead_in=lead_in,
                            heading=heading,
                            parent=parent,
                            subject_hint=fragment.subject_hint,
                        )
                        or (
                            item_label is not None
                            and _has_independent_subject(fragment.text)
                        )
                    )
                ):
                    continue
                hypothesis, context, evaluable, incomplete_reason = (
                    _claim_context(
                        fragment.text,
                        lead_in,
                        heading=heading,
                        parent=parent,
                        subject_hint=fragment.subject_hint,
                    )
                )
                if item_label is not None:
                    hypothesis, context, evaluable, incomplete_reason = (
                        _label_body_claim(
                            semantic_text,
                            item_label,
                            lead_in=lead_in,
                            heading=heading,
                            parent=parent,
                        )
                    )
                key = _claim_key(hypothesis)
                if not key or key in seen:
                    continue
                if len(claims) >= self._max_claims:
                    warnings.append(
                        f"Claim extraction stopped at the configured limit of "
                        f"{self._max_claims} claims."
                    )
                    limit_reached = True
                    break
                seen.add(key)
                claim = _make_claim(
                    claim_id=f"c{len(claims) + 1}",
                    sentence=sentence,
                    claim_text=(
                        cleaned.text if item_label is not None else fragment.text
                    ),
                    hypothesis=hypothesis,
                    context=context,
                    evaluable=evaluable,
                    incomplete_reason=incomplete_reason,
                    claim_start=(
                        cleaned.start if item_label is not None else fragment.start
                    ),
                    claim_end=(
                        cleaned.end if item_label is not None else fragment.end
                    ),
                    method=self.method,
                )
                claims.append(claim)
                item_claims.append(claim)
                last_claim_sentence_index = sentence.index
            if cleaned.is_list_item:
                item_context = (
                    semantic_text
                    if _is_noun_phrase(semantic_text)
                    else (
                        item_claims[-1].hypothesis
                        if item_claims
                        else semantic_text
                    )
                )
                _push_list_context(
                    list_contexts, cleaned.list_indent, item_context
                )
                last_list_item_start = cleaned.list_item_start
            else:
                pending_lead_in = None
                list_contexts.clear()
                last_list_item_start = -1
            if limit_reached:
                break
        return ClaimExtractionResult(
            claims=tuple(claims), method=self.method, warnings=tuple(warnings)
        )


def _claim_key(text: str) -> str:
    value = _normalize_markdown(_strip_emphasis_edges(text))
    value = " ".join(value.split()).casefold()
    return _without_closers(value).rstrip("".join(_TERMINAL_PUNCTUATION)).strip()


def _sentences(answer: str) -> list[_SentenceCandidate]:
    lines = answer.splitlines(keepends=True)
    if not lines:
        return []
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    candidates: list[_SentenceCandidate] = []
    index = 1
    paragraph_start: int | None = None
    paragraph_end: int | None = None

    def append_spans(
        start: int,
        end: int,
        *,
        is_heading: bool = False,
        heading_level: int = 0,
        is_list_item: bool = False,
        list_indent: int = 0,
    ) -> None:
        nonlocal index
        block = answer[start:end]
        if not block.strip():
            return
        if is_heading:
            text = block.strip()
            candidates.append(
                _SentenceCandidate(
                    index=index,
                    text=text,
                    start=start,
                    end=end,
                    is_heading=True,
                    heading_level=heading_level,
                    is_list_item=is_list_item,
                    list_indent=list_indent,
                    list_item_start=start if is_list_item else -1,
                )
            )
            index += 1
            return
        for span in split_sentences(block, join_wrapped_lines=True):
            if not span.text.strip():
                continue
            candidates.append(
                _SentenceCandidate(
                    index=index,
                    text=span.text,
                    start=start + span.start,
                    end=start + span.end,
                    is_list_item=is_list_item,
                    list_indent=list_indent,
                    list_item_start=start if is_list_item else -1,
                )
            )
            index += 1

    def flush_paragraph() -> None:
        nonlocal paragraph_start, paragraph_end
        if paragraph_start is not None and paragraph_end is not None:
            append_spans(paragraph_start, paragraph_end)
        paragraph_start = None
        paragraph_end = None

    position = 0
    while position < len(lines):
        line = lines[position]
        content = line.rstrip("\r\n")
        line_start = line_starts[position]
        line_end = line_start + len(line)
        if not content.strip():
            flush_paragraph()
            position += 1
            continue

        markdown_heading = _MARKDOWN_HEADING.match(content)
        bold_heading = _BOLD_HEADING.match(content)
        if markdown_heading is not None or bold_heading is not None:
            flush_paragraph()
            level = (
                len(markdown_heading.group("hashes"))
                if markdown_heading is not None
                else 0
            )
            append_spans(
                line_start,
                line_end,
                is_heading=True,
                heading_level=level,
            )
            position += 1
            continue

        list_match = _STRUCTURED_LIST_ITEM.match(content)
        if list_match is not None:
            flush_paragraph()
            indent = len(list_match.group("indent").expandtabs(4))
            block_end = line_end
            next_position = position + 1
            while next_position < len(lines):
                continuation = lines[next_position]
                continuation_content = continuation.rstrip("\r\n")
                if not continuation_content.strip():
                    break
                if (
                    _MARKDOWN_HEADING.match(continuation_content)
                    or _BOLD_HEADING.match(continuation_content)
                    or _STRUCTURED_LIST_ITEM.match(continuation_content)
                ):
                    break
                if not continuation_content[:1].isspace():
                    break
                block_end = (
                    line_starts[next_position] + len(continuation)
                )
                next_position += 1
            append_spans(
                line_start,
                block_end,
                is_list_item=True,
                list_indent=indent,
            )
            position = next_position
            continue

        if _is_plain_heading_line(
            content, lines, position, starts_paragraph=paragraph_start is None
        ):
            flush_paragraph()
            append_spans(line_start, line_end, is_heading=True)
            position += 1
            continue

        if paragraph_start is None:
            paragraph_start = line_start
        paragraph_end = line_end
        position += 1

    flush_paragraph()
    return candidates


def _is_plain_heading_line(
    content: str,
    lines: list[str],
    position: int,
    *,
    starts_paragraph: bool,
) -> bool:
    if not starts_paragraph:
        return False
    stripped = content.strip()
    if not stripped or _has_terminal_punctuation(stripped):
        return False
    if (
        _MARKDOWN_HEADING.match(content)
        or _BOLD_HEADING.match(content)
        or _STRUCTURED_LIST_ITEM.match(content)
    ):
        return False
    words = _word_tokens(stripped)
    if not 2 <= len(words) <= 18:
        return False
    if not _is_latin_text(stripped):
        return False

    next_content = ""
    for next_line in lines[position + 1 :]:
        next_content = next_line.rstrip("\r\n")
        if next_content.strip():
            break
    if next_content and _STRUCTURED_LIST_ITEM.match(next_content):
        return True
    if len(words) <= 10 and not _has_finite_verb(stripped):
        return True
    if stripped.endswith((":", "：")):
        return True
    return _is_heading_like(stripped, stripped)


def _clean_sentence(sentence: _SentenceCandidate) -> _CleanedSpan | None:
    base = sentence.text
    offset = 0
    is_list_item = sentence.is_list_item
    for _ in range(3):
        if _LIST_PREFIX.match(base):
            is_list_item = True
        prefix = _LEADING_PREFIX.match(base)
        if prefix is None:
            break
        offset += prefix.end()
        base = base[prefix.end() :]

    discourse = _DISCOURSE_PREFIX.match(base)
    had_discourse_prefix = discourse is not None
    if discourse is not None:
        offset += discourse.end()
        base = base[discourse.end() :]

    if _META_PREFIX.match(base):
        return None

    base_start = offset
    base_end = len(sentence.text)
    trailing = _TRAILING_CITATION.search(sentence.text, base_start)
    if trailing:
        base_end = trailing.start()
    while base_end > base_start and sentence.text[base_end - 1].isspace():
        base_end -= 1
    if base_end <= base_start:
        return None

    raw = sentence.text[base_start:base_end]
    label = _SECTION_LABEL.match(raw)
    if label is not None and _is_section_label(
        label.group("label"), raw[label.end() :]
    ):
        base_start += label.end()
        raw = sentence.text[base_start:base_end]
    text = _strip_emphasis_edges(_normalize_markdown(raw)).strip()
    if had_discourse_prefix and text:
        text = text[0].upper() + text[1:]
    if not text:
        return None
    return _CleanedSpan(
        text=text,
        start=sentence.start + base_start,
        end=sentence.start + base_end,
        is_list_item=is_list_item,
        is_heading=sentence.is_heading,
        heading_level=sentence.heading_level,
        list_indent=sentence.list_indent,
        list_item_start=sentence.list_item_start,
    )


def _is_section_label(value: str, following: str) -> bool:
    if not following.strip():
        return False
    words = _word_tokens(value)
    if not words or len(words) > 6:
        return False
    if any(character.isdigit() for character in value) or "/" in value:
        return False
    if words[0].casefold() in _DETERMINERS:
        return False
    if value.rstrip().endswith((".", "!", "?")):
        return False
    if _has_independent_subject(following):
        return True
    return _first_finite_verb_index(words) is None


def _normalize_markdown(text: str) -> str:
    value = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.DOTALL)
    value = re.sub(r"(?<!\w)\*(?!\*)(.+?)(?<!\*)\*(?!\w)", r"\1", value)
    value = re.sub(r"(?<!\w)\*+|\*+(?!\w)", "", value)
    return value.replace("`", "")


def _is_lead_in(text: str) -> bool:
    value = text.strip()
    if not value.endswith((":", "：")):
        return False
    body = value[:-1].strip()
    if len(_word_tokens(body)) < 2:
        return False
    if _MODAL_LEAD_IN.search(body) or _TRAILING_LEAD_IN.search(body):
        return True
    if re.search(r"\b(?:the following|as follows|including|such as)\s*$", body, re.IGNORECASE):
        return True
    if any(pattern.match(body) for pattern in _INTRO_PATTERNS):
        return True
    return bool(re.search(r"\b(?:lists?|describes?|defines?|covers?)\b", body, re.IGNORECASE))


def _split_list_label_body(text: str) -> tuple[str, str] | None:
    for match in _LIST_LABEL_SEPARATOR.finditer(text):
        label = text[: match.start()].strip()
        body = text[match.end() :].strip()
        if not label or not body:
            continue
        label_words = _word_tokens(label)
        if not 1 <= len(label_words) <= 10:
            continue
        if label.casefold() in {"source", "sources", "reference", "references"}:
            continue
        if _ends_with_question(label):
            continue
        if not _has_independent_subject(body) and not _looks_like_label_body(body):
            continue
        return label, body
    return None


def _looks_like_label_body(text: str) -> bool:
    words = _word_tokens(text)
    if len(words) < 3:
        return False
    original_first = words[0]
    first = original_first.casefold()
    if first in _ANAPHORIC_SUBJECTS or _is_adverb(first):
        return False
    if first in _DETERMINERS or first in _AUXILIARIES:
        return True
    return original_first[:1].isupper() and not _is_verb_like(first)


def _label_body_claim(
    body: str,
    label: str,
    *,
    lead_in: str | None,
    heading: str | None,
    parent: str | None,
) -> tuple[str, str | None, bool, str | None]:
    if _has_independent_subject(body):
        return body, label, True, None
    if _is_noun_phrase(body) or _starts_with_verb(body):
        hypothesis = _hypothesis_from_heading(body, label)
        if hypothesis is not None:
            return hypothesis, label, True, None
    reason = (
        "The label and explanation do not form an independently evaluable "
        "proposition."
    )
    return body, label, False, reason


def _split_compound(sentence: _SentenceCandidate) -> list[_SentenceCandidate]:
    if not _is_latin_text(sentence.text):
        return [sentence]

    for match in _CLAUSE_SEPARATOR.finditer(sentence.text):
        if _inside_dash_parenthetical(sentence.text, match.start()):
            continue
        left = _sub_sentence(sentence, 0, match.start())
        right = _sub_sentence(sentence, match.end(), len(sentence.text))
        if left is None or right is None:
            continue

        separator = sentence.text[match.start() : match.end()]
        if ";" not in separator and "," not in separator:
            if "," in left.text or "," in right.text:
                continue
        if not _is_independent_clause(left.text):
            continue
        if not _is_independent_clause(right.text):
            continue
        if not _looks_factual(left.text) or not _looks_factual(right.text):
            continue
        return [*_split_compound(left), *_split_compound(right)]

    predicate_fragments = _split_predicate_coordination(sentence)
    if predicate_fragments is not None:
        return predicate_fragments
    return [sentence]


def _dash_parenthetical_spans(text: str) -> list[tuple[int, int]]:
    opening: int | None = None
    spans: list[tuple[int, int]] = []
    for position, character in enumerate(text):
        if character not in "—–":
            continue
        if opening is None:
            opening = position
        else:
            spans.append((opening, position))
            opening = None
    return spans


def _inside_dash_parenthetical(text: str, position: int) -> bool:
    return any(
        start < position < end
        for start, end in _dash_parenthetical_spans(text)
    )


def _split_predicate_coordination(
    sentence: _SentenceCandidate,
) -> list[_SentenceCandidate] | None:
    text = sentence.text
    if "," not in text or _dash_parenthetical_spans(text):
        return None
    segments = _comma_segments(text)
    if len(segments) < 2:
        return None
    first_clause = text[segments[0][0] : segments[0][1]]
    if re.search(r"\bto\s+\w+$", first_clause, re.IGNORECASE):
        return None

    first_start, first_end = segments[0]
    subject_span = _subject_predicate_span(text[first_start:first_end])
    if subject_span is None:
        return None
    subject_end, _predicate_start = subject_span
    subject = text[first_start : first_start + subject_end].strip()
    if not subject:
        return None

    groups = [segments[0]]
    for start, end in segments[1:]:
        segment = text[start:end].strip()
        if not segment:
            return None
        if _starts_with_verb(segment):
            groups.append((start, end))
        else:
            previous_start, _previous_end = groups[-1]
            groups[-1] = (previous_start, end)
    if len(groups) < 2:
        return None

    first = _sub_sentence(sentence, groups[0][0], groups[0][1])
    if first is None:
        return None
    fragments = [first]
    for start, end in groups[1:]:
        fragment = _sub_sentence(sentence, start, end)
        if fragment is None or not _starts_with_verb(fragment.text):
            return None
        fragments.append(
            _SentenceCandidate(
                index=fragment.index,
                text=fragment.text,
                start=fragment.start,
                end=fragment.end,
                is_list_item=fragment.is_list_item,
                list_indent=fragment.list_indent,
                list_item_start=fragment.list_item_start,
                subject_hint=subject,
            )
        )
    return fragments


def _comma_segments(text: str) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    depth = 0
    for position, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            segments.append(_trim_segment(text, start, position, strip_and=False))
            start = position + 1
    segments.append(_trim_segment(text, start, len(text), strip_and=False))
    if any(start >= end for start, end in segments):
        return []
    normalized: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(segments):
        if index > 0:
            start, end = _trim_segment(text, start, end, strip_and=True)
        if start >= end:
            return []
        normalized.append((start, end))
    return normalized


def _trim_segment(
    text: str, start: int, end: int, *, strip_and: bool
) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if strip_and:
        match = re.match(r"(?:and|but|yet)\s+", text[start:end], re.IGNORECASE)
        if match is not None:
            start += match.end()
    return start, end


def _subject_predicate_span(text: str) -> tuple[int, int] | None:
    words = list(_WORD.finditer(text))
    for index, word in enumerate(words):
        if not _is_finite_verb(word.group(0)):
            continue
        if index == 0:
            return None
        subject_words = [item.group(0) for item in words[:index]]
        if not any(
            value.casefold() not in _DETERMINERS
            and value.casefold() not in _AUXILIARIES
            and value.casefold() not in _CONTINUATION_STARTERS
            for value in subject_words
        ):
            return None
        return word.start(), word.start()
    return None


def _sub_sentence(
    sentence: _SentenceCandidate, start: int, end: int
) -> _SentenceCandidate | None:
    while start < end and (
        sentence.text[start].isspace() or sentence.text[start] in ",;:"
    ):
        start += 1
    while end > start and (
        sentence.text[end - 1].isspace() or sentence.text[end - 1] in ",;:"
    ):
        end -= 1
    if end <= start:
        return None
    return _SentenceCandidate(
        index=sentence.index,
        text=sentence.text[start:end],
        start=sentence.start + start,
        end=sentence.start + end,
        is_list_item=sentence.is_list_item,
        list_indent=sentence.list_indent,
        list_item_start=sentence.list_item_start,
    )


def _is_independent_clause(text: str) -> bool:
    words = _word_tokens(text)
    return len(words) >= 3 and _has_independent_subject(text)


def _claim_context(
    text: str,
    lead_in: str | None,
    *,
    heading: str | None = None,
    parent: str | None = None,
    subject_hint: str | None = None,
) -> tuple[str, str | None, bool, str | None]:
    value = text.strip()
    if not value or not _is_latin_text(value):
        return value, None, True, None
    if subject_hint and _starts_with_verb(value):
        hypothesis = (
            f"{_capitalize(subject_hint)} "
            f"{_lower_initial(_strip_terminal_sentence(value))}."
        )
        return hypothesis, subject_hint, True, None
    if _starts_with_anaphoric_subject(value):
        resolved = _resolve_anaphora(value, parent)
        if resolved is not None:
            context = _sentence_subject(parent) if parent else None
            return resolved, context or parent, True, None
        return (
            value,
            None,
            False,
            (
                "The claim starts with a pronoun or demonstrative whose "
                "referent is outside the claim, so it cannot be evaluated as "
                "an independent proposition."
            ),
        )
    if _has_independent_subject(value):
        return value, None, True, None
    reconstructed = _contextual_hypothesis(
        value, lead_in=lead_in, heading=heading, parent=parent
    )
    if reconstructed is not None:
        hypothesis, context = reconstructed
        return hypothesis, context, True, None
    return (
        value,
        None,
        False,
        (
            "The claim has no explicit independent subject or finite "
            "predicate, so it cannot be evaluated as a standalone proposition."
        ),
    )


def _can_reconstruct(
    text: str,
    *,
    lead_in: str | None,
    heading: str | None,
    parent: str | None,
    subject_hint: str | None = None,
) -> bool:
    value = text.strip()
    if _ends_with_question(value):
        return False
    words = _word_tokens(value)
    if len(words) < 2:
        return False
    if subject_hint and _starts_with_verb(value):
        return True
    if _starts_with_anaphoric_subject(value):
        return parent is not None
    return bool(lead_in or heading or parent)


def _nested_parent(
    contexts: list[tuple[int, str]], indent: int
) -> str | None:
    while contexts and contexts[-1][0] >= indent:
        contexts.pop()
    return contexts[-1][1] if contexts else None


def _push_list_context(
    contexts: list[tuple[int, str]], indent: int, text: str
) -> None:
    while contexts and contexts[-1][0] >= indent:
        contexts.pop()
    contexts.append((indent, text.strip()))


def _resolve_anaphora(text: str, parent: str | None) -> str | None:
    if not parent:
        return None
    words = _word_tokens(text)
    if not words:
        return None
    first = words[0].casefold()
    if first not in _ANAPHORIC_SUBJECTS:
        return None
    remainder = text.strip()[len(words[0]) :].strip()
    if not remainder:
        return None
    if not (
        _is_finite_verb(remainder.split(maxsplit=1)[0])
        or _is_adverb(remainder.split(maxsplit=1)[0])
    ):
        return None
    base = _strip_terminal_sentence(parent)
    subject = _sentence_subject(base)
    if subject is not None:
        return f"{_capitalize(subject)} {_lower_initial(remainder)}"
    return f"{base} {remainder}"


def _sentence_subject(text: str) -> str | None:
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    for start, end in reversed(_comma_segments(cleaned)):
        segment = cleaned[start:end].strip()
        if not segment:
            continue
        span = _subject_predicate_span(segment)
        if span is None:
            continue
        subject = segment[: span[0]].strip(" ,;:-—–")
        if not subject:
            continue
        first = _word_tokens(subject)[0].casefold()
        if (
            first in _CONTINUATION_STARTERS
            and first not in _PREPOSITIONAL_STARTERS
        ) or _is_adverb(first):
            continue
        return " ".join(subject.split())
    return None


def _contextual_hypothesis(
    value: str,
    *,
    lead_in: str | None,
    heading: str | None,
    parent: str | None,
) -> tuple[str, str] | None:
    if parent:
        hypothesis = _hypothesis_from_parent(value, parent)
        if hypothesis is not None:
            return hypothesis, parent
    if lead_in:
        hypothesis = _hypothesis_from_lead_in(value, lead_in)
        if hypothesis is not None:
            return hypothesis, lead_in
    if heading:
        hypothesis = _hypothesis_from_heading(value, heading)
        if hypothesis is not None:
            return hypothesis, heading
    return None


def _hypothesis_from_parent(value: str, parent: str) -> str | None:
    value = _strip_terminal_sentence(value)
    base = _strip_terminal_sentence(parent)
    if not _is_noun_phrase(base):
        return None
    if _is_noun_phrase(value):
        return f"{base} includes {_lower_initial(value)}."
    if _starts_with_verb(value):
        return f"{base} includes {_gerundize(value)}."
    return None


def _hypothesis_from_lead_in(value: str, lead_in: str) -> str | None:
    value = _strip_terminal_sentence(value)
    context = lead_in.strip().rstrip(":：").strip()
    modal = _lead_subject_modal(context)
    if modal is not None:
        return f"{modal} {_lower_initial(value)}."

    category = re.match(
        r"^(?:the following|these|they)\s+are\s+(?P<category>.+)$",
        context,
        re.IGNORECASE,
    )
    if category is not None and _is_noun_phrase(value):
        category_text = _strip_lead_determiner(
            category.group("category").strip()
        )
        singular = _singular(category_text)
        article = _article(singular)
        return f"{article} {singular} is {_lower_initial(value)}."
    if category is not None and _starts_with_verb(value):
        category_text = _strip_lead_determiner(
            category.group("category").strip()
        )
        singular = _singular(category_text)
        article = _article(singular)
        return f"{article} {singular} is to {_lower_initial(value)}."

    if re.search(r"\b(?:are|is)\s*$", context, re.IGNORECASE):
        if _starts_with_verb(value):
            return f"{context} to {_lower_initial(value)}."
        if _is_noun_phrase(value):
            return f"{context} {_lower_initial(value)}."

    if re.search(
        r"\b(?:includes?|comprises?|consists?\s+of|covers?|involves?)\s*$",
        context,
        re.IGNORECASE,
    ):
        if _starts_with_verb(value):
            return f"{context} {_gerundize(value)}."
        if _is_noun_phrase(value):
            return f"{context} {_lower_initial(value)}."

    including = re.match(
        r"^(?P<prefix>.+?)\b(?:including|such as)\s*$",
        context,
        re.IGNORECASE,
    )
    if including is not None:
        prefix = including.group("prefix").strip()
        if _starts_with_verb(value):
            return f"{prefix}, including {_gerundize(value)}."
        return f"{prefix}, including {_lower_initial(value)}."

    # A colon can introduce a simple noun phrase, but only turn it into a
    # claim when the lead clearly supplies a category for the item.
    subject = _lead_subject(context)
    if subject is not None and _starts_with_verb(value):
        return f"{subject} {_lower_initial(value)}."
    return None


def _hypothesis_from_heading(value: str, heading: str) -> str | None:
    value = _strip_terminal_sentence(value)
    context = heading.strip().rstrip(":：").strip()
    modal = _lead_subject_modal(context)
    if modal is not None:
        return f"{modal} {_lower_initial(value)}."

    how = re.match(
        r"^how\s+(?P<subject>.+?)\s+"
        r"(?P<verb>helps?|supports?|enables?|allows?|assists?|"
        r"contributes?\s+to)\s+"
        r"(?P<object>.+)$",
        context,
        re.IGNORECASE,
    )
    if how is not None and (
        _is_noun_phrase(value) or _starts_with_verb(value)
    ):
        subject = _capitalize(how.group("subject"))
        verb = how.group("verb").casefold()
        object_text = how.group("object")
        contribution = (
            _gerundize(value)
            if _starts_with_verb(value)
            else _lower_initial(value)
        )
        return (
            f"{subject} {verb} {object_text} through "
            f"{contribution}."
        )

    category = re.match(
        r"^(?:the\s+)?(?:main\s+|key\s+|several\s+)?"
        r"(?P<category>objectives?|goals?|benefits?|components?|"
        r"elements?|types?|ways?|steps?|roles?|responsibilities?|"
        r"risks?|threats?)\s+(?:of|for)\s+(?P<subject>.+)$",
        context,
        re.IGNORECASE,
    )
    if category is not None:
        category_name = category.group("category").casefold()
        subject = category.group("subject").strip()
        if category_name in {"objectives", "objective", "goals", "goal"}:
            if _starts_with_verb(value):
                return (
                    f"An objective of {subject} is to "
                    f"{_lower_initial(value)}."
                )
            if _is_noun_phrase(value):
                return f"An objective of {subject} is {_lower_initial(value)}."
        if category_name in {"benefits", "benefit"} and _is_noun_phrase(value):
            return f"{_capitalize(subject)} provides {_lower_initial(value)}."
        if category_name in {
            "components",
            "component",
            "elements",
            "element",
            "types",
            "type",
        } and _is_noun_phrase(value):
            return f"{_capitalize(subject)} includes {_lower_initial(value)}."
        if category_name in {"ways", "way", "steps", "step"}:
            if _is_noun_phrase(value):
                return f"A way {_capitalize(subject)} works is {_lower_initial(value)}."
        if category_name in {"risks", "risk", "threats", "threat"} and _is_noun_phrase(value):
            return f"{_capitalize(value)} is a {_singular(category_name)} to {subject}."

    if _is_noun_phrase(context):
        if _is_noun_phrase(value):
            if len(_word_tokens(context)) >= 2:
                return (
                    f"{_capitalize(context)} includes "
                    f"{_lower_initial(value)}."
                )
        if _starts_with_verb(value):
            return (
                f"{_capitalize(context)} includes "
                f"{_gerundize(value)}."
            )
    return None


def _lead_subject_modal(context: str) -> str | None:
    match = re.search(
        r"\b(?P<modal>must|shall|should|can|could|will|may|would)\b",
        context,
        re.IGNORECASE,
    )
    if match is None:
        return None
    prefix = context[: match.start()].strip()
    prefix = re.split(
        r"\b(?:that|which|who|specifies?|requires?|states?|notes?|"
        r"explains?|indicates?|defines?)\b|[;,]",
        prefix,
        flags=re.IGNORECASE,
    )[-1].strip()
    prefix = re.sub(r"^(?:and|or)\s+", "", prefix, flags=re.IGNORECASE)
    if not prefix or prefix.casefold() in {"it", "this", "that"}:
        return None
    if len(_word_tokens(prefix)) > 8:
        return None
    return f"{_capitalize(prefix)} {match.group('modal').casefold()}"


def _lead_subject(context: str) -> str | None:
    match = re.match(
        r"^(?P<subject>(?:the|an?|each|every|all|this|these)\s+.+?)"
        r"\s+(?:lists?|describes?|defines?|covers?|includes?|contains?)\b",
        context,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return _capitalize(match.group("subject"))


def _strip_lead_determiner(text: str) -> str:
    return re.sub(
        r"^(?:the\s+)?(?:main\s+|key\s+|several\s+)?",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _is_noun_phrase(text: str) -> bool:
    value = text.strip()
    if not value or _ends_with_question(value):
        return False
    words = _word_tokens(value)
    if len(words) < 2:
        return False
    if _has_finite_verb(value):
        return False
    first = words[0].casefold()
    if first in _CONTINUATION_STARTERS or _is_adverb(first):
        return False
    if _is_verb_like(first) and not _is_participial_start(first):
        return False
    return True


def _starts_with_verb(text: str) -> bool:
    words = _word_tokens(text)
    if not words:
        return False
    first = words[0].casefold()
    return _is_verb_like(first) and not _is_participial_start(first)


def _gerundize(text: str) -> str:
    words = text.split()
    if not words:
        return text
    first = words[0]
    lower = first.casefold()
    if lower.endswith("ie"):
        changed = f"{first[:-2]}ying"
    elif lower.endswith("e") and len(first) > 2:
        changed = f"{first[:-1]}ing"
    elif (
        len(first) >= 3
        and first[-1] not in "aeiou"
        and first[-2] in "aeiou"
        and first[-3] not in "aeiou"
    ):
        changed = f"{first}{first[-1]}ing"
    else:
        changed = f"{first}ing"
    return " ".join([changed, *words[1:]])


def _capitalize(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    return value[0].upper() + value[1:]


def _singular(text: str) -> str:
    value = text.strip()
    if value.casefold().endswith("ies") and len(value) > 3:
        return f"{value[:-3]}y"
    if value.casefold().endswith(("ses", "xes", "zes", "ches", "shes")):
        return value[:-2]
    if value.casefold().endswith("s") and not value.casefold().endswith("ss"):
        return value[:-1]
    return value


def _article(word: str) -> str:
    return "an" if word[:1].casefold() in "aeiou" else "a"


def _has_independent_subject(text: str) -> bool:
    value = text.strip()
    words = _word_tokens(value)
    if not words:
        return False
    first = words[0].casefold()
    if _is_adverb(first):
        return False
    if first in _CONTINUATION_STARTERS and first not in _PREPOSITIONAL_STARTERS:
        comma = value.find(",")
        if 0 < comma < len(value) - 1:
            remainder = value[comma + 1 :].strip()
            if remainder:
                return _has_independent_subject(remainder)
        return False
    if _is_verb_like(first) and not _is_participial_start(first):
        return False
    text = value
    for finite_index, word in enumerate(words):
        if not _is_finite_verb(word):
            continue
        if finite_index == 0 and _is_participial_start(first):
            continue
        if finite_index < 1:
            continue
        if words[finite_index - 1].casefold() in {
            "as",
            "thereby",
            "including",
            "according",
        }:
            continue
        before = words[:finite_index]
        for candidate in before:
            value = candidate.casefold()
            if value in _DETERMINERS or value in _AUXILIARIES:
                continue
            if value in _CONTINUATION_STARTERS or _is_adverb(value):
                continue
            return True
    return False


def _first_finite_verb_index(words: list[str]) -> int | None:
    for index, word in enumerate(words):
        if _is_finite_verb(word):
            return index
    return None


def _is_participial_start(word: str) -> bool:
    value = word.casefold().replace("’", "'")
    return value.endswith(("ed", "ing")) or value in {
        "based",
        "given",
        "derived",
        "defined",
        "documented",
        "established",
        "implemented",
        "integrated",
        "maintained",
        "managed",
        "measured",
        "monitored",
        "planned",
        "reviewed",
    }


def _starts_with_anaphoric_subject(text: str) -> bool:
    words = _word_tokens(text)
    if not words:
        return False
    first = words[0].casefold()
    if first not in _ANAPHORIC_SUBJECTS:
        return False
    if len(words) == 1:
        return True
    second = words[1].casefold()
    if _is_finite_verb(second) or _is_adverb(second):
        return True
    return second in {"also", "then", "still", "only", "just", "not"}


def _nli_lead_in(context: str) -> str:
    if re.search(r"\b(?:must|shall|should)\b", context, re.IGNORECASE):
        return "It is required to"
    return context


def _lower_initial(text: str) -> str:
    if not text:
        return text
    first = text.split(maxsplit=1)[0]
    if len(first) > 1 and first.isupper():
        return text
    return text[0].lower() + text[1:]


def _looks_factual(claim: str) -> bool:
    text = claim.strip()
    if not text:
        return False
    normalized = _strip_emphasis_edges(text)
    if not normalized:
        return False
    if _ends_with_question(normalized):
        return False
    if any(pattern.match(normalized) for pattern in _INTRO_PATTERNS):
        return False
    if _is_heading_like(text, normalized):
        return False

    letters = sum(character.isalpha() for character in normalized)
    if letters < 4:
        return False
    words = _word_tokens(normalized)
    if len(words) < 2 and not any(character.isdigit() for character in normalized):
        return False

    if _is_latin_text(normalized):
        if _is_likely_english(normalized) and not _has_finite_verb(normalized):
            return False
        if not _has_terminal_punctuation(normalized) and not _has_finite_verb(
            normalized
        ):
            return False
    return True


def _is_heading_like(raw: str, normalized: str) -> bool:
    if normalized.endswith((":", "：")):
        return True
    if _QUESTION_START.match(normalized):
        return True
    if _EMPHASIS_WRAPPED.match(raw.strip()):
        return True
    if _TRAILING_LEAD_IN.search(normalized):
        return True
    if _is_latin_text(normalized) and not _has_finite_verb(normalized):
        if len(_word_tokens(normalized)) <= 8:
            return True
        if _TITLE_LEAD.match(normalized):
            return True
        if _is_likely_english(normalized):
            return True
    return False


def _strip_emphasis_edges(text: str) -> str:
    previous = None
    value = text.strip()
    while value != previous:
        previous = value
        value = _EMPHASIS_EDGE.sub("", value).strip()
    return value


def _ends_with_question(text: str) -> bool:
    return _without_closers(text).endswith(("?", "？"))


def _has_terminal_punctuation(text: str) -> bool:
    return _without_closers(text).endswith(_TERMINAL_PUNCTUATION)


def _without_closers(text: str) -> str:
    value = text.rstrip()
    while value and value[-1] in _CLOSING_PUNCTUATION:
        value = value[:-1].rstrip()
    return value


def _strip_terminal_sentence(text: str) -> str:
    return (
        _without_closers(text)
        .rstrip("".join(_TERMINAL_PUNCTUATION))
        .strip()
    )


def _word_tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _is_latin_text(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    latin = sum(
        1
        for character in letters
        if "LATIN" in unicodedata.name(character, "")
    )
    return latin / len(letters) >= 0.6


def _is_likely_english(text: str) -> bool:
    words = [word.casefold() for word in _word_tokens(text)]
    if not words:
        return False
    matches = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
    return matches >= 2 or (len(words) <= 4 and matches >= 1)


def _has_finite_verb(text: str) -> bool:
    return any(_is_finite_verb(word) for word in _word_tokens(text))


def _is_finite_verb(word: str) -> bool:
    value = word.casefold().replace("’", "'")
    if value in {"be", "been", "being"}:
        return False
    if value in _AUXILIARIES:
        return True
    if value.endswith("ing"):
        return False
    return _is_verb_like(word)


def _is_adverb(word: str) -> bool:
    value = word.casefold()
    return value.endswith("ly") or value in {
        "also",
        "already",
        "always",
        "even",
        "further",
        "however",
        "just",
        "never",
        "not",
        "often",
        "only",
        "otherwise",
        "still",
        "then",
        "thereby",
        "therefore",
        "thus",
        "too",
    }


def _is_verb_like(word: str) -> bool:
    value = word.casefold().replace("’", "'")
    if value in _AUXILIARIES or value in _IRREGULAR_VERBS or value in _VERB_BASES:
        return True

    candidates: list[str] = []
    if len(value) > 3 and value.endswith("ies"):
        candidates.append(f"{value[:-3]}y")
    if len(value) > 3 and value.endswith("ied"):
        candidates.append(f"{value[:-3]}y")
    if len(value) > 3 and value.endswith("es"):
        candidates.append(value[:-2])
    if len(value) > 2 and value.endswith("s") and not value.endswith("ss"):
        candidates.append(value[:-1])
    if len(value) > 3 and value.endswith("ed"):
        candidates.extend((value[:-2], value[:-1]))
    if len(value) > 4 and value.endswith("ing"):
        candidates.extend((value[:-3], f"{value[:-3]}e"))
        if len(value) > 5 and value[-4] == value[-5]:
            candidates.append(value[:-4])
    return any(candidate in _VERB_BASES for candidate in candidates)


def _make_claim(
    claim_id: str,
    sentence: _SentenceCandidate,
    claim_text: str,
    hypothesis: str,
    context: str | None,
    evaluable: bool,
    incomplete_reason: str | None,
    claim_start: int,
    claim_end: int,
    method: str,
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=claim_id,
        sentence_index=sentence.index,
        sentence=sentence.text,
        sentence_start=sentence.start,
        sentence_end=sentence.end,
        claim=claim_text,
        claim_start=claim_start,
        claim_end=claim_end,
        hypothesis=hypothesis,
        context=context,
        complete=evaluable,
        evaluable=evaluable,
        incomplete_reason=incomplete_reason,
        language=DetectedLanguage(method="not_detected"),
        extraction_method=method,
    )


def build_claim_extractor(
    settings: Settings, generator: object | None = None
) -> ClaimExtractor:
    return HeuristicClaimExtractor(
        max_claims=settings.verification_max_extracted_claims
    )
