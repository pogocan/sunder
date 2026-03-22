"""Tests for text extraction and cleaning."""

from sunder.extract import clean_text


def test_dehyphenation():
    assert clean_text("knowl-\nedge base") == "knowledge base"
    assert clean_text("multi-\n  line") == "multiline"


def test_legitimate_hyphens_preserved():
    assert "self-contained" in clean_text("self-contained")
    assert "well-known" in clean_text("well-known method")


def test_whitespace_normalization():
    assert clean_text("too   many   spaces") == "too many spaces"
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_page_breaks_preserved():
    assert "--- Page Break ---" in clean_text("text\n\n--- Page Break ---\n\nmore")
