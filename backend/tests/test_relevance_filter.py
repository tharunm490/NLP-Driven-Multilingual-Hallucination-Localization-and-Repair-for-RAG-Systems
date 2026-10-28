"""Unit tests for per-chunk cosine-distance relevance filtering."""

from __future__ import annotations

import math

from app.core.config import Settings
from app.services.relevance_filter import filter_relevant_chunks
from tests.rag_support import chunk_by_id


def test_threshold_is_inclusive_and_rejections_are_reported(settings: Settings):
    narrow = settings.model_copy(update={"rag_max_distance": 0.55})
    result = filter_relevant_chunks(
        [
            chunk_by_id("iso-cia", 0.55),
            chunk_by_id("iso-risk", 0.56),
            chunk_by_id("iso-audit", 0.20),
        ],
        narrow,
    )

    assert [chunk.chunk_id for chunk in result.kept] == ["iso-audit", "iso-cia"]
    assert len(result.rejected) == 1
    assert result.rejected[0].chunk.chunk_id == "iso-risk"
    assert result.rejected[0].reason == "distance_above_threshold"
    assert result.threshold == 0.55
    assert result.metric == "cosine_distance"
    assert result.filter_enabled is True


def test_invalid_scores_are_rejected_even_when_filter_is_disabled(settings: Settings):
    disabled = settings.model_copy(update={"rag_distance_filter_enabled": False})
    result = filter_relevant_chunks(
        [
            chunk_by_id("iso-cia", math.nan),
            chunk_by_id("iso-risk", math.inf),
            chunk_by_id("iso-scope", 0.40),
        ],
        disabled,
    )

    assert [chunk.chunk_id for chunk in result.kept] == ["iso-scope"]
    assert {item.chunk.chunk_id for item in result.rejected} == {"iso-cia", "iso-risk"}
    assert {item.reason for item in result.rejected} == {"invalid_score"}
    assert result.threshold is None
    assert result.filter_enabled is False


def test_negative_rounding_noise_is_clamped_and_similarity_is_derived(settings: Settings):
    result = filter_relevant_chunks([chunk_by_id("iso-cia", -0.0001)], settings)
    kept = result.kept[0]

    assert kept.distance == 0.0
    assert kept.similarity == 1.0


def test_filter_does_not_mutate_input_chunks(settings: Settings):
    original = chunk_by_id("iso-cia", 0.30)
    result = filter_relevant_chunks([original], settings)

    assert original.distance == 0.30
    assert result.kept[0] is not original
