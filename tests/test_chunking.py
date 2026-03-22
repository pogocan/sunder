"""Tests for chunking."""

from sunder.chunking import chunk_text
from sunder.types import SunderConfig


def _make_text(n_paragraphs=8):
    """Generate test text with enough words to pass min_chunk_size."""
    paras = [
        f"Paragraph {i} discusses topic number {i} with enough detail and context "
        f"to ensure the word count exceeds the minimum chunk size threshold of fifty "
        f"words which is the default value. This text helps meet that requirement."
        for i in range(n_paragraphs)
    ]
    return "\n\n".join(paras)


def test_chunk_text_returns_chunks():
    text = _make_text()
    chunks = chunk_text(text, doc_id="test")
    assert len(chunks) > 0
    assert all(c.doc_id == "test" for c in chunks)


def test_deterministic_ids():
    text = _make_text()
    c1 = chunk_text(text, doc_id="stable")
    c2 = chunk_text(text, doc_id="stable")
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]


def test_chunk_id_format():
    text = _make_text()
    chunks = chunk_text(text, doc_id="mydoc")
    for c in chunks:
        assert c.chunk_id.startswith("mydoc_chunk_")


def test_flat_chunks_have_no_topic():
    text = _make_text()
    chunks = chunk_text(text, doc_id="test")
    for c in chunks:
        assert c.topic_title is None
        assert c.topic_index is None


def test_min_chunk_size_filtering():
    text = "Short."
    cfg = SunderConfig(chunk_size=200, min_chunk_size=50)
    chunks = chunk_text(text, doc_id="test", config=cfg)
    assert len(chunks) == 0


def test_custom_config():
    text = _make_text()
    cfg = SunderConfig(chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    chunks = chunk_text(text, doc_id="test", config=cfg)
    assert len(chunks) > 2
