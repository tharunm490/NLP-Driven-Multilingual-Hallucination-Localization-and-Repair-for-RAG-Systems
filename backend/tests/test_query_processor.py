"""Tests for conservative query preprocessing."""

from __future__ import annotations

import pytest

from app.core.errors import QueryValidationError
from app.services.query_processor import preprocess_query


def test_trims_and_collapses_whitespace():
    result = preprocess_query("   When   was\t the \n company founded?  ")
    assert result.normalized == "When was the company founded?"
    assert result.was_modified is True


def test_preserves_punctuation_numbers_and_case():
    result = preprocess_query("Did Acme Corp. reach $1.5M by Q3 2023?")
    assert result.normalized == "Did Acme Corp. reach $1.5M by Q3 2023?"


def test_preserves_hindi_text_without_translation():
    original = " कंपनी की स्थापना कब हुई? "
    result = preprocess_query(original)
    assert result.normalized == "कंपनी की स्थापना कब हुई?"
    assert "कंपनी" in result.normalized


def test_removes_zero_width_and_control_characters():
    result = preprocess_query("Eiff\u200b el\u200d Tower\x00?")
    assert "\u200b" not in result.normalized
    assert "\x00" not in result.normalized
    assert result.normalized == "Eiff el Tower ?"


def test_normalizes_non_breaking_spaces():
    result = preprocess_query("New\u00a0Delhi\uff1f")
    assert result.normalized == "New Delhi\uff1f"


def test_empty_query_raises():
    with pytest.raises(QueryValidationError):
        preprocess_query("   \n\t  ")


def test_overlong_query_raises():
    with pytest.raises(QueryValidationError):
        preprocess_query("a" * 50, max_length=10)


def test_original_is_kept_for_logging():
    original = "  Where is Paris?  "
    result = preprocess_query(original)
    assert result.original == original
    assert result.normalized == "Where is Paris?"
