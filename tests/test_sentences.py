"""Tests for sentence segmentation."""

from sunder.chunking import chunk_text
from sunder.sentences import segment_sentences
from sunder.types import Chunk, SunderConfig


def _make_chunk(text, chunk_id="test_chunk_0000", doc_id="test"):
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text,
        token_count=len(text) // 4, chunk_index=0,
        page=1, start_char=0, end_char=len(text),
    )


def test_bullets_atomic():
    c = _make_chunk("- Install component\n- Configure DataMover")
    segment_sentences([c])
    assert len(c.sentences) == 2
    assert c.sentences[0].text == "- Install component"


def test_abbreviations_not_split():
    text = (
        "Install e.g. the CICS component first and then configure the "
        "DataMover service for production deployment and verify all "
        "settings are correct before proceeding to the next step."
    )
    c = _make_chunk(text)
    segment_sentences([c])
    assert len(c.sentences) == 1


def test_short_lines_atomic():
    c = _make_chunk("Step 1: Prerequisites")
    segment_sentences([c])
    assert len(c.sentences) == 1


def test_ids_globally_unique():
    chunks = [
        _make_chunk("First sentence.", chunk_id="test_chunk_0000"),
        _make_chunk("Second sentence.", chunk_id="test_chunk_0001"),
    ]
    segment_sentences(chunks)
    all_ids = [s.sentence_id for c in chunks for s in c.sentences]
    assert len(all_ids) == len(set(all_ids))


def test_in_place_mutation():
    chunks = [_make_chunk("Hello world.")]
    result = segment_sentences(chunks)
    assert result is chunks


def test_empty_chunk():
    c = _make_chunk("")
    segment_sentences([c])
    assert c.sentences == []


def test_counter_resets_per_doc():
    chunks = [
        _make_chunk("A.", chunk_id="a_chunk_0000", doc_id="docA"),
        _make_chunk("B.", chunk_id="b_chunk_0000", doc_id="docB"),
    ]
    segment_sentences(chunks)
    assert chunks[0].sentences[0].sentence_id == "docA_sent_000000"
    assert chunks[1].sentences[0].sentence_id == "docB_sent_000000"
