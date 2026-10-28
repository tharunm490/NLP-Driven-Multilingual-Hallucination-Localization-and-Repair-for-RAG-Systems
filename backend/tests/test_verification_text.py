"""Tests for offset-preserving verification text utilities."""

from __future__ import annotations

from app.services.verification.text import (
    canonicalize,
    content_terms,
    split_sentences,
    word_spans,
)


def test_canonicalize_preserves_source_offsets_for_expanding_normalization():
    source = "The \ufb01le is ready."
    canonical = canonicalize(source)

    assert canonical.text == "The file is ready."
    span = canonical.source_span(4, 8)
    assert span.text == "\ufb01le"
    assert source[span.start : span.end] == span.text


def test_sentence_segmentation_handles_english_and_hindi_terminators():
    text = "First sentence. दूसरा वाक्य। Third one! Final?"
    spans = split_sentences(text)

    assert [span.text for span in spans] == [
        "First sentence.",
        "दूसरा वाक्य।",
        "Third one!",
        "Final?",
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_sentence_segmentation_does_not_split_decimals_or_common_abbreviations():
    text = "Version 3.5 is current. Dr. Rao approved it."
    spans = split_sentences(text)

    assert [span.text for span in spans] == [
        "Version 3.5 is current.",
        "Dr. Rao approved it.",
    ]


def test_sentence_segmentation_keeps_urls_and_domains_intact():
    text = "See www.iso.org/directives or https://patents.iec.ch. Thank you."
    spans = split_sentences(text)

    assert [span.text for span in spans] == [
        "See www.iso.org/directives or https://patents.iec.ch.",
        "Thank you.",
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_sentence_spans_remain_valid_after_unicode_normalization_changes_length():
    text = "\ufb01rst claim. second claim."
    spans = split_sentences(text)

    assert spans
    assert spans[0].text == "\ufb01rst claim."
    assert text[spans[0].start : spans[0].end] == spans[0].text
    assert text[spans[1].start : spans[1].end] == spans[1].text


def test_word_spans_and_terms_include_multilingual_words_and_clause_numbers():
    text = "Clause 6.1.2 में जोखिम है."

    assert [span.text for span in word_spans(text)][:4] == [
        "Clause",
        "6.1.2",
        "में",
        "जोखिम",
    ]
    terms = content_terms(text)
    assert "6.1.2" in terms
    assert "जोखिम" in terms

def test_join_wrapped_lines_reassembles_pdf_propositions_and_preserves_offsets():
    text = (
        "The organization shall monitor and measure\n"
        "security controls and evaluate\n"
        "the results. Next sentence."
    )

    spans = split_sentences(text, join_wrapped_lines=True)

    assert [span.text for span in spans] == [
        (
            "The organization shall monitor and measure\n"
            "security controls and evaluate\n"
            "the results."
        ),
        "Next sentence.",
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_join_wrapped_lines_keeps_headings_and_paragraphs_separate():
    text = (
        "Introduction\n"
        "The organization shall monitor and measure\n"
        "security controls and evaluate the results.\n\n"
        "Next paragraph."
    )

    spans = split_sentences(text, join_wrapped_lines=True)

    assert [span.text for span in spans] == [
        "Introduction",
        (
            "The organization shall monitor and measure\n"
            "security controls and evaluate the results."
        ),
        "Next paragraph.",
    ]


def test_default_segmentation_still_splits_soft_line_breaks():
    text = "The organization shall monitor and measure\nsecurity controls."

    assert [span.text for span in split_sentences(text)] == [
        "The organization shall monitor and measure",
        "security controls.",
    ]
