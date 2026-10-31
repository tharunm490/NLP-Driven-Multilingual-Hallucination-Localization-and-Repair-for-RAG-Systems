"""Hindi-specific rule-based claim extraction.

This extractor never translates, paraphrases, or reconstructs a sentence. A
claim is accepted only when the original text already contains a complete
predicate; headings, labels, questions, and fragments are excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.verification import TextSpan
from app.services.verification.claim_common import (
    MAX_CLAIMS,
    claim_key,
    make_extracted_claim,
    select_claims,
)
from app.services.verification.claim_location import normalize_for_matching
from app.services.verification.claims import (
    ClaimExtractionResult,
    SkippedClaim,
)
from app.services.verification.text import split_sentences, word_spans

_HEADING = re.compile(r"^[ \t]{0,3}(?P<hashes>#{1,6})[ \t]+(?P<body>.+?)\s*$")
_BOLD_ONLY = re.compile(
    r"^\s*(?:\*\*|__)(?P<body>.+?)(?:\*\*|__)\s*:?\s*$"
)
_LIST_ITEM = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-*+•▪◦‣]|\d{1,3}(?:\.\d{1,3})*[.)]|"
    r"[ivxlcdm]{1,5}[.)])[ \t]+(?P<body>.+)$",
    re.IGNORECASE,
)
_LABEL_BODY = re.compile(
    r"^\s*(?P<label>[^:：]{1,60}?)[ \t]*[:：][ \t]*(?P<body>.+)$"
)
_QUESTION_START = re.compile(
    r"^(?:क्या|क्यों|कैसे|कब|कहाँ|कहां|कौन|किसने|किसको|किसका|किसकी|"
    r"कितना|कितने|कितनी)(?:\s|$)"
)
_GREETING = re.compile(
    r"^(?:नमस्ते|नमस्कार|धन्यवाद|शुभ\s+प्रभात|आपका\s+स्वागत)"
)
_FILLER = re.compile(
    r"^(?:यहाँ|यहां|इस\s+(?:उत्तर|अनुभाग|खंड|प्रतिक्रिया)\s+में|"
    r"निम्नलिखित\s+(?:बिंदु|बिंदुओं|सूची)|नीचे\s+दिए\s+गए|"
    r"ऊपर\s+दिए\s+गए)"
)
_OPINION = re.compile(
    r"^(?:मुझे\s+लगता\s+है|मेरी\s+राय|मेरे\s+विचार|शायद)"
)
_COMPOUND_SEPARATOR = re.compile(
    r"\s*;\s*|\s*,\s*(?:और|तथा|एवं)\s+|\s+(?:और|तथा|एवं)\s+"
)
_TERMINAL = ("।", "॥", ".", "!", "?", "؟")

_HINDI_FINITE_WORDS = frozenset(
    """
    है हैं था थी थे हूँ हूं हों हो होगा होगी होंगे होता होती होते
    चाहिए चाहिये सकता सकती सकते सकें सका सकी सके
    करता करती करते करें करेगा करेगी करेंगे किया करना करनी करने
    दिया दी दिए लिया ली लिए गया गई गए रहा रही रहे
    रखता रखती रखते बनता बनती बनते मिलता मिलती मिलते
    आता आती आते जाता जाती जाते पाता पाती पाते
    देता देती देते लेता लेती लेते कहलाता कहलाती कहलाते
    निर्धारित करता करती करते स्थापित सुनिश्चित लागू
    आवश्यक अनिवार्य जरूरी ज़रूरी जिम्मेदार उत्तरदायी
    शामिल सम्मिलित आधारित युक्त संपन्न स्थित सक्षम
    """.split()
)
_HINDI_VERB_FORM = re.compile(
    r"^(?:कर|हो|जा|आ|दे|ले|रख|बन|मिल|पा|जान|कह|मान|समझ|निर्धार|"
    r"सुनिश्चित|स्थापित|लागू|विकसित|प्रबंध|संचालित|मूल्यांकन|विश्लेषण|"
    r"पहचान|दस्तावेज|अपना|बढ़ा|घटा|रोक|सुधार|प्रदान|समर्थन|सुरक्षित|"
    r"संग्रह|निगरानी|जाँच|परीक्षण|समीक्षा|अद्यतन)"
    r"(?:ता|ती|ते|या|यी|ये|ना|नी|ने|ें|ेंगे|ेगा|ेगी|कर|"
    r"करता|करती|करते|किया|करना|करनी|करने|करें|करेंगे|"
    r"रहा|रही|रहे|दिया|दी|दिए|लिया|ली|लिए|सकता|सकती|सकते)$"
)


@dataclass(frozen=True)
class _Unit:
    text: str
    start: int
    end: int
    is_heading: bool = False
    is_list_item: bool = False


@dataclass(frozen=True)
class _Candidate:
    text: str
    start: int
    end: int


class HindiClaimExtractor:
    method = "hindi_heuristic"

    def __init__(self, max_claims: int = MAX_CLAIMS) -> None:
        self._max_claims = max(1, min(max_claims, MAX_CLAIMS))

    def extract(self, answer: str) -> ClaimExtractionResult:
        text = answer or ""
        if not text.strip():
            return ClaimExtractionResult(
                claims=(),
                method=self.method,
                warnings=("The answer was empty; no claims were extracted.",),
            )

        claims = []
        skipped: list[SkippedClaim] = []
        seen: set[str] = set()
        sentences = split_sentences(text, join_wrapped_lines=True)
        for unit in _units(text):
            if unit.is_heading:
                skipped.append(
                    SkippedClaim(
                        text=unit.text,
                        start=unit.start,
                        end=unit.end,
                        reason=(
                            "A heading or section label does not assert a "
                            "factual proposition."
                        ),
                    )
                )
                continue
            for candidate in _unit_candidates(answer, unit):
                if not _is_complete_hindi(candidate.text):
                    skipped.append(
                        SkippedClaim(
                            text=candidate.text,
                            start=candidate.start,
                            end=candidate.end,
                            reason=_hindi_non_claim_reason(candidate.text),
                        )
                    )
                    continue
                key = claim_key(candidate.text)
                if not key or key in seen:
                    if key:
                        skipped.append(
                            SkippedClaim(
                                text=candidate.text,
                                start=candidate.start,
                                end=candidate.end,
                                reason=(
                                    "The candidate duplicates an earlier "
                                    "claim."
                                ),
                            )
                        )
                    continue
                seen.add(key)
                claims.append(
                    make_extracted_claim(
                        answer=answer,
                        claim_id=f"c{len(claims) + 1}",
                        claim_text=candidate.text,
                        start=candidate.start,
                        end=candidate.end,
                        method=self.method,
                        language_code="hi",
                        sentences=sentences,
                    )
                )

        selected, warnings = select_claims(claims, self._max_claims)
        return ClaimExtractionResult(
            claims=tuple(selected),
            method=self.method,
            warnings=tuple(warnings),
            skipped=tuple(skipped),
        )


def _units(answer: str) -> list[_Unit]:
    lines = answer.splitlines(keepends=True)
    if not lines:
        return []
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    units: list[_Unit] = []
    paragraph_start: int | None = None
    paragraph_end: int | None = None

    def flush() -> None:
        nonlocal paragraph_start, paragraph_end
        if paragraph_start is not None and paragraph_end is not None:
            value, start, end = _trim_span(
                answer, paragraph_start, paragraph_end
            )
            if value:
                units.extend(_paragraph_units(answer, start, end))
        paragraph_start = None
        paragraph_end = None

    position = 0
    while position < len(lines):
        line = lines[position]
        content = line.rstrip("\r\n")
        line_start = starts[position]
        line_end = line_start + len(line)
        if not content.strip():
            flush()
            position += 1
            continue

        heading = _HEADING.match(content)
        if heading is not None:
            flush()
            body = heading.group("body")
            body_start = line_start + heading.start("body")
            units.append(
                _Unit(
                    text=body.strip(),
                    start=body_start,
                    end=body_start + len(body),
                    is_heading=True,
                )
            )
            position += 1
            continue

        bold = _BOLD_ONLY.match(content)
        if bold is not None:
            body = bold.group("body").strip()
            if not _has_terminal(body) and not _has_hindi_predicate(body):
                flush()
                body_start = line_start + bold.start("body")
                units.append(
                    _Unit(
                        text=body,
                        start=body_start,
                        end=body_start + len(bold.group("body")),
                        is_heading=True,
                    )
                )
                position += 1
                continue

        list_match = _LIST_ITEM.match(content)
        if list_match is not None:
            flush()
            body_start = line_start + list_match.start("body")
            block_end = line_end
            next_position = position + 1
            while next_position < len(lines):
                continuation = lines[next_position]
                continuation_content = continuation.rstrip("\r\n")
                if not continuation_content.strip():
                    break
                if (
                    _HEADING.match(continuation_content)
                    or _BOLD_ONLY.match(continuation_content)
                    or _LIST_ITEM.match(continuation_content)
                ):
                    break
                if not continuation_content[:1].isspace():
                    break
                block_end = starts[next_position] + len(continuation)
                next_position += 1
            value, start, end = _trim_span(answer, body_start, block_end)
            if value:
                units.extend(_paragraph_units(answer, start, end, list_item=True))
            position = next_position
            continue

        if paragraph_start is None and _looks_like_heading_line(content):
            flush()
            value, start, end = _trim_span(answer, line_start, line_end)
            if value:
                units.append(
                    _Unit(
                        text=value,
                        start=start,
                        end=end,
                        is_heading=True,
                    )
                )
            position += 1
            continue

        if paragraph_start is None:
            paragraph_start = line_start
        paragraph_end = line_end
        position += 1

    flush()
    return units


def _paragraph_units(
    answer: str, start: int, end: int, *, list_item: bool = False
) -> list[_Unit]:
    block = answer[start:end]
    units: list[_Unit] = []
    for span in split_sentences(block, join_wrapped_lines=True):
        value, span_start, span_end = _trim_span(
            answer, start + span.start, start + span.end
        )
        if value:
            units.append(
                _Unit(
                    text=value,
                    start=span_start,
                    end=span_end,
                    is_list_item=list_item,
                )
            )
    return units


def _unit_candidates(answer: str, unit: _Unit) -> list[_Candidate]:
    label_body = _label_body_candidate(answer, unit)
    if label_body is not None:
        return _split_compound(label_body)
    candidate = _Candidate(unit.text, unit.start, unit.end)
    return _split_compound(candidate)


def _label_body_candidate(
    answer: str, unit: _Unit
) -> _Candidate | None:
    match = _LABEL_BODY.match(unit.text)
    if match is None:
        return None
    label = match.group("label").strip()
    if not _is_short_label(label):
        return None
    body = match.group("body").strip()
    if not body:
        return None
    body_start = unit.start + match.start("body")
    value, start, end = _trim_span(
        answer, body_start, unit.start + match.end("body")
    )
    if not value:
        return None
    return _Candidate(value, start, end)


def _is_short_label(value: str) -> bool:
    tokens = [span.text for span in word_spans(value)]
    return (
        1 <= len(tokens) <= 8
        and not _has_hindi_predicate(value)
        and not _has_terminal(value)
    )


def _split_compound(candidate: _Candidate) -> list[_Candidate]:
    for match in _COMPOUND_SEPARATOR.finditer(candidate.text):
        left_value, left_start, left_end = _trim_span(
            candidate.text, 0, match.start()
        )
        right_value, right_start, right_end = _trim_span(
            candidate.text, match.end(), len(candidate.text)
        )
        if not left_value or not right_value:
            continue
        left = _Candidate(
            left_value,
            candidate.start + left_start,
            candidate.start + left_end,
        )
        right = _Candidate(
            right_value,
            candidate.start + right_start,
            candidate.start + right_end,
        )
        if not _is_complete_hindi(left.text):
            continue
        if not _is_complete_hindi(right.text):
            continue
        if _token_count(left.text) < 4 or _token_count(right.text) < 4:
            continue
        return [*_split_compound(left), *_split_compound(right)]
    return [candidate]


def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while start < end and text[start] in "*_`":
        start += 1
    while end > start and text[end - 1] in "*_`":
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[start:end], start, end


def _looks_like_heading_line(content: str) -> bool:
    value = normalize_for_matching(content).text.strip()
    if not value:
        return False
    if value.endswith((":", "：")):
        return True
    if _has_terminal(value):
        return False
    return len(value) <= 80 and not _has_hindi_predicate(value)


def _is_complete_hindi(text: str) -> bool:
    value = normalize_for_matching(text).text.strip()
    if not value:
        return False
    if value.endswith((":", "：")):
        return False
    if _QUESTION_START.match(value) or value.endswith(("?", "？")):
        return False
    if _GREETING.match(value) or _FILLER.match(value) or _OPINION.match(value):
        return False
    tokens = [span.text for span in word_spans(value)]
    if len(tokens) < 3:
        return False
    if not _has_hindi_predicate(value):
        return False
    if not _has_terminal(text) and len(tokens) < 5:
        return False
    return True


def _has_hindi_predicate(text: str) -> bool:
    for token in (span.text for span in word_spans(text)):
        if token in _HINDI_FINITE_WORDS:
            return True
        if _HINDI_VERB_FORM.match(token):
            return True
    return False


def _has_terminal(text: str) -> bool:
    value = text.rstrip()
    while value and value[-1] in "\"'”’)]}":
        value = value[:-1].rstrip()
    return value.endswith(_TERMINAL)


def _token_count(text: str) -> int:
    return len(word_spans(normalize_for_matching(text).text))


def _hindi_non_claim_reason(text: str) -> str:
    value = normalize_for_matching(text).text.strip()
    if not value:
        return "The fragment is empty."
    if value.endswith(("?", "？")) or _QUESTION_START.match(value):
        return "A question is not a factual claim."
    if value.endswith((":", "：")):
        return (
            "A heading or section label does not assert a factual "
            "proposition."
        )
    if _GREETING.match(value) or _FILLER.match(value) or _OPINION.match(value):
        return "The fragment is conversational filler or opinion."
    if not _has_hindi_predicate(value):
        return (
            "The fragment has no complete Hindi predicate and cannot be "
            "evaluated as a standalone proposition."
        )
    return (
        "The fragment is not a complete, independently verifiable Hindi "
        "factual sentence."
    )


__all__ = ["HindiClaimExtractor"]
