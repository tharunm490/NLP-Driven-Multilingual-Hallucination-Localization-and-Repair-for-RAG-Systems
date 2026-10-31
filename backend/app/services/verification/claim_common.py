"""Shared claim construction, deduplication, and hard-limit selection."""

from __future__ import annotations

import re
import unicodedata

from app.schemas.verification import DetectedLanguage, ExtractedClaim, TextSpan
from app.services.verification.claim_location import (
    containing_sentence,
    normalize_for_matching,
)
from app.services.verification.text import split_sentences, word_spans

MAX_CLAIMS = 30

_IMPORTANT_PATTERNS = (
    re.compile(r"\b\d{1,4}(?:[.,]\d+)?%?\b"),
    re.compile(
        r"\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not|no|never|none|neither|nor|without|cannot|can't|"
        r"must\s+not|shall\s+not|does\s+not|do\s+not|is\s+not|are\s+not)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all|any|each|every|only|at\s+least|at\s+most|more\s+than|"
        r"fewer\s+than|less\s+than|exactly|approximately|about|up\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:नहीं|कोई\s+नहीं|सभी|प्रत्येक|अधिकतम|न्यूनतम|केवल)"),
)


def claim_key(text: str) -> str:
    """Deterministic duplicate key independent of formatting/punctuation."""

    normalized = unicodedata.normalize(
        "NFKC", normalize_for_matching(text).text
    ).casefold()
    return " ".join(span.text for span in word_spans(normalized))


def importance_score(text: str) -> int:
    return sum(1 for pattern in _IMPORTANT_PATTERNS if pattern.search(text))


def select_claims(
    claims: list[ExtractedClaim],
    max_claims: int,
) -> tuple[list[ExtractedClaim], list[str]]:
    """Cap claims at the hard limit, retaining important claims in order."""

    limit = max(1, min(max_claims, MAX_CLAIMS))
    ordered = sorted(claims, key=lambda claim: (claim.claim_start, claim.claim_end))
    if len(ordered) <= limit:
        return ordered, []
    ranked = sorted(
        enumerate(ordered),
        key=lambda item: (-importance_score(item[1].claim), item[0]),
    )
    retained = {index for index, _ in ranked[:limit]}
    selected = [
        claim for index, claim in enumerate(ordered) if index in retained
    ]
    return selected, [
        f"Claim extraction found more than the hard limit of {limit} claims; "
        f"the {limit} most important claims were retained in source order."
    ]


def make_extracted_claim(
    *,
    answer: str,
    claim_id: str,
    claim_text: str,
    start: int,
    end: int,
    method: str,
    language_code: str | None,
    hypothesis: str | None = None,
    context: str | None = None,
    evaluable: bool = True,
    incomplete_reason: str | None = None,
    sentences: list[TextSpan] | None = None,
) -> ExtractedClaim:
    if sentences is None:
        sentences = split_sentences(answer, join_wrapped_lines=True)
    if not sentences:
        sentences = [TextSpan(start=0, end=len(answer), text=answer)]
    sentence_index, sentence = containing_sentence(sentences, start, end)
    return ExtractedClaim(
        claim_id=claim_id,
        sentence_index=sentence_index,
        sentence=sentence.text,
        sentence_start=sentence.start,
        sentence_end=sentence.end,
        claim=claim_text,
        claim_start=start,
        claim_end=end,
        hypothesis=hypothesis or "",
        context=context,
        complete=evaluable,
        evaluable=evaluable,
        incomplete_reason=incomplete_reason,
        language=DetectedLanguage(
            code=language_code,
            confidence=None,
            reliable=language_code is not None,
            method="claim_extractor_route" if language_code else "not_detected",
        ),
        extraction_method=method,
        location_status="verified",
    )


__all__ = [
    "MAX_CLAIMS",
    "claim_key",
    "importance_score",
    "make_extracted_claim",
    "select_claims",
]
