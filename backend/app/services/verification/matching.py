"""Claim-to-evidence ranking with a deterministic lexical fallback."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from app.services.verification.text import content_terms

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "का",
        "की",
        "के",
        "को",
        "और",
        "में",
        "से",
        "है",
        "हैं",
    }
)


@dataclass(frozen=True)
class EvidenceMatch:
    evidence_index: int
    similarity: float


@dataclass(frozen=True)
class MatchingResult:
    matches: tuple[tuple[EvidenceMatch, ...], ...]
    method: str
    warnings: tuple[str, ...] = ()


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class EvidenceMatcher(Protocol):
    def match_many(
        self, claims: Sequence[str], evidence: Sequence[str]
    ) -> MatchingResult: ...


class SemanticEvidenceMatcher:
    method = "embedding_cosine"

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def match_many(
        self, claims: Sequence[str], evidence: Sequence[str]
    ) -> MatchingResult:
        if not claims:
            return MatchingResult(matches=(), method=self.method)
        if not evidence:
            return MatchingResult(
                matches=tuple(() for _ in claims), method=self.method
            )
        evidence_vectors = self._embedder.embed_documents(list(evidence))
        claim_vectors = [self._embedder.embed_query(claim) for claim in claims]
        matches: list[tuple[EvidenceMatch, ...]] = []
        for claim_vector in claim_vectors:
            scored = [
                EvidenceMatch(
                    evidence_index=index,
                    similarity=_cosine(claim_vector, evidence_vector),
                )
                for index, evidence_vector in enumerate(evidence_vectors)
            ]
            scored.sort(key=lambda item: (-item.similarity, item.evidence_index))
            matches.append(tuple(scored))
        return MatchingResult(matches=tuple(matches), method=self.method)


class LexicalEvidenceMatcher:
    method = "lexical_overlap"

    def match_many(
        self, claims: Sequence[str], evidence: Sequence[str]
    ) -> MatchingResult:
        if not claims:
            return MatchingResult(matches=(), method=self.method)
        evidence_terms = [_terms(text) for text in evidence]
        matches: list[tuple[EvidenceMatch, ...]] = []
        for claim in claims:
            query_terms = _terms(claim)
            scored: list[EvidenceMatch] = []
            for index, terms in enumerate(evidence_terms):
                score = _overlap(query_terms, terms)
                scored.append(
                    EvidenceMatch(evidence_index=index, similarity=score)
                )
            scored.sort(key=lambda item: (-item.similarity, item.evidence_index))
            matches.append(tuple(scored))
        return MatchingResult(matches=tuple(matches), method=self.method)


class FallbackEvidenceMatcher:
    def __init__(self, primary: EvidenceMatcher, fallback: EvidenceMatcher) -> None:
        self._primary = primary
        self._fallback = fallback

    def match_many(
        self, claims: Sequence[str], evidence: Sequence[str]
    ) -> MatchingResult:
        try:
            return self._primary.match_many(claims, evidence)
        except Exception as exc:
            logger.warning("Semantic evidence matching failed; using lexical fallback: %s", exc)
            result = self._fallback.match_many(claims, evidence)
            return MatchingResult(
                matches=result.matches,
                method="lexical_fallback",
                warnings=(*result.warnings, f"Semantic matching failed: {exc}"),
            )


def build_evidence_matcher(embedder: Embedder) -> EvidenceMatcher:
    return FallbackEvidenceMatcher(
        SemanticEvidenceMatcher(embedder), LexicalEvidenceMatcher()
    )


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        term
        for term in content_terms(text)
        if term not in _STOPWORDS and (len(term) > 1 or term.isdigit())
    )


def _overlap(query: frozenset[str], evidence: frozenset[str]) -> float:
    if not query or not evidence:
        return 0.0
    return len(query & evidence) / len(query)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    value = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))
