"""Tests for the query language detection service."""

from __future__ import annotations

import pytest

from app.services.query_language_detection import (
    LanguageScore,
    QueryLanguageDetector,
)


class StaticBackend:
    name = "static"

    def __init__(self, scores: list[LanguageScore]) -> None:
        self._scores = scores

    def is_available(self) -> bool:
        return True

    def detect(self, text: str) -> list[LanguageScore]:
        return list(self._scores)


class UnavailableBackend:
    name = "unavailable"

    def is_available(self) -> bool:
        return False

    def detect(self, text: str) -> list[LanguageScore]:  # pragma: no cover
        return []


def make_detector(scores, **kwargs) -> QueryLanguageDetector:
    return QueryLanguageDetector(
        enabled=True,
        backend=StaticBackend(scores),
        fallback_backend=UnavailableBackend(),
        **kwargs,
    )


def test_detects_english():
    detector = make_detector([LanguageScore("en", 0.97)])
    result = detector.detect("When was the company founded?")
    assert result.code == "en"
    assert result.reliable is True
    assert result.confidence == pytest.approx(0.97)


def test_detects_hindi():
    detector = make_detector([LanguageScore("hi", 0.95)])
    result = detector.detect("कंपनी की स्थापना कब हुई?")
    assert result.code == "hi"
    assert result.reliable is True


def test_numbers_only_returns_no_language():
    detector = make_detector([LanguageScore("en", 0.99)])
    result = detector.detect("2024 42 3.14")
    assert result.code is None
    assert result.reliable is False
    assert result.reason == "no_letters"


def test_short_query_is_not_confidently_assigned():
    detector = make_detector([LanguageScore("en", 0.99)])
    result = detector.detect("hi")
    assert result.reliable is False
    assert result.reason == "too_short"


def test_disabled_detector_returns_nothing():
    detector = QueryLanguageDetector(enabled=False)
    result = detector.detect("When was the company founded?")
    assert result.code is None
    assert result.reliable is False
    assert result.method == "disabled"


def test_low_confidence_is_not_reliable():
    detector = make_detector([LanguageScore("en", 0.30), LanguageScore("fr", 0.20)])
    result = detector.detect("When was the company founded?")
    assert result.reliable is False
    assert result.reason == "low_confidence"
    assert result.code == "en"  # best-effort guess, but flagged unreliable


def test_mixed_script_query_is_flagged():
    detector = make_detector([LanguageScore("en", 0.99)])
    result = detector.detect("What is the capital of भारत?")
    assert result.mixed is True
    assert result.reliable is False
    assert result.reason == "mixed_language"


def test_close_runner_up_is_treated_as_mixed():
    detector = make_detector(
        [LanguageScore("en", 0.55), LanguageScore("fr", 0.52)], min_margin=0.15
    )
    result = detector.detect("Where is this place located now?")
    assert result.mixed is True
    assert result.reliable is False


def test_fallback_language_is_used_when_unreliable():
    detector = make_detector([LanguageScore("en", 0.20)], fallback_language="en")
    result = detector.detect("zzzz qqqq")
    assert result.code == "en"
    assert result.reliable is False
    assert result.method.endswith("+fallback")


def test_unavailable_backends_return_no_language():
    detector = QueryLanguageDetector(
        enabled=True,
        backend=UnavailableBackend(),
        fallback_backend=UnavailableBackend(),
    )
    result = detector.detect("When was the company founded?")
    assert result.code is None
    assert result.reliable is False
    assert result.reason == "no_candidates"


def test_real_backend_detects_english_and_hindi():
    detector = QueryLanguageDetector(enabled=True)
    if not detector.available():
        pytest.skip("no real language-detection backend installed")
    english = detector.detect("When was the company founded?")
    assert english.code == "en"
    assert english.reliable is True
    hindi = detector.detect("कंपनी की स्थापना कब हुई?")
    assert hindi.code == "hi"
    assert hindi.reliable is True
