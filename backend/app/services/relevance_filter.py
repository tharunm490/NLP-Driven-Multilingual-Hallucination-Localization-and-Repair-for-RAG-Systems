"""Retrieval relevance filtering.

ChromaDB is configured with cosine space, so its ``distance`` is
``1 - cosine_similarity``: lower is more similar. It is a ranking signal, not a
probability, confidence, or factual-verification result.

The filter is deliberately per-chunk. A retrieved top-K list is a candidate
set, not evidence. Chunks above the configured distance threshold are rejected
before any evidence is shown or sent to the LLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.core.config import Settings
from app.schemas.retrieval import RetrievedChunk

DISTANCE_METRIC = "cosine_distance"
SCORE_INTERPRETATION = (
    "ChromaDB cosine distance (1 - cosine similarity): lower is more similar. "
    "It is not a probability, confidence score, or proof that a chunk answers "
    "the question."
)


@dataclass(frozen=True)
class RejectedChunk:
    """A retrieved candidate that did not pass the relevance policy."""

    chunk: RetrievedChunk
    reason: str


@dataclass(frozen=True)
class RelevanceFilterResult:
    """Per-chunk relevance decisions for one retrieval call."""

    candidates: tuple[RetrievedChunk, ...]
    kept: tuple[RetrievedChunk, ...]
    rejected: tuple[RejectedChunk, ...]
    filter_enabled: bool
    threshold: float | None
    metric: str = DISTANCE_METRIC
    score_interpretation: str = SCORE_INTERPRETATION


def filter_relevant_chunks(
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
) -> RelevanceFilterResult:
    """Normalize scores and keep only chunks that satisfy the relevance policy.

    Invalid scores (NaN/inf) are always rejected: they cannot be interpreted
    safely. When ``RAG_DISTANCE_FILTER_ENABLED=false`` the absolute-distance
    gate is intentionally bypassed, but scores are still normalized and ranked.
    This switch exists only for controlled experiments/backward compatibility;
    the production default is enabled.
    """

    threshold = settings.rag_max_distance if settings.rag_distance_filter_enabled else None
    kept: list[RetrievedChunk] = []
    rejected: list[RejectedChunk] = []

    for chunk in sorted(chunks, key=lambda item: _normalized_distance(item.distance)):
        distance = _normalized_distance(chunk.distance)
        if not math.isfinite(distance):
            rejected.append(RejectedChunk(chunk=_with_normalized_score(chunk, distance), reason="invalid_score"))
            continue
        if not settings.rag_distance_filter_enabled:
            kept.append(_with_normalized_score(chunk, distance))
            continue
        assert threshold is not None
        if distance <= threshold:
            kept.append(_with_normalized_score(chunk, distance))
        else:
            rejected.append(
                RejectedChunk(
                    chunk=_with_normalized_score(chunk, distance),
                    reason="distance_above_threshold",
                )
            )

    return RelevanceFilterResult(
        candidates=tuple(chunks),
        kept=tuple(kept),
        rejected=tuple(rejected),
        filter_enabled=settings.rag_distance_filter_enabled,
        threshold=threshold,
    )


def _normalized_distance(value: float) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(distance):
        return math.nan
    # Cosine distance is 0..2 in Chroma. Small negative values can occur from
    # floating-point rounding; clamp them to zero rather than treating them as
    # an error. Values above 2 are left as-is so they fail a 0..2 threshold.
    return max(0.0, distance)


def _with_normalized_score(chunk: RetrievedChunk, distance: float) -> RetrievedChunk:
    similarity = 1.0 - distance if math.isfinite(distance) else math.nan
    similarity = min(1.0, max(-1.0, similarity)) if math.isfinite(similarity) else math.nan
    return chunk.model_copy(update={"distance": distance, "similarity": similarity})
