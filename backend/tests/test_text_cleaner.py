from app.services.text_cleaner import clean_text, has_meaningful_text


def test_normalises_line_endings_and_whitespace():
    raw = "Hello   world\r\n\r\n\r\nSecond\tparagraph  \r\n"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\n\nSecond paragraph"


def test_preserves_punctuation_numbers_dates_and_names():
    raw = "Dr. A. Sharma met Asha on 12/03/2024 at 09:30. Cost: $1,250.75!"
    cleaned = clean_text(raw)
    assert cleaned == raw


def test_preserves_multilingual_text_and_danda():
    raw = "यह एक परीक्षण है। मूल्य ₹1,200 है।\n\nதமிழ் சோதனை 2024."
    cleaned = clean_text(raw)
    assert "परीक्षण" in cleaned
    assert "।" in cleaned
    assert "₹1,200" in cleaned
    assert "தமிழ்" in cleaned


def test_removes_zero_width_and_soft_hyphen():
    raw = "invisi\u200bble\u00ad text\ufeff"
    assert clean_text(raw) == "invisible text"


def test_collapses_repeated_blank_lines_only():
    raw = "a\n\n\n\nb"
    assert clean_text(raw) == "a\n\nb"


def test_handles_empty_input():
    assert clean_text("") == ""
    assert not has_meaningful_text("   \n  ")
    assert has_meaningful_text("ok 123")
