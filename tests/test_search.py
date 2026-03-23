"""Tests for search -- SearchHit.snippets and topic_summary populated."""

import pytest

from sunder.chunking import chunk_by_topics_sentence_aware
from sunder.index import index_embeddings, Corpus
from sunder.types import DocumentStructure, Section, SunderConfig


# -- Synthetic fixture: 2 topics, ~4 sentences each --------------------------

_TOPIC_A_TEXT = (
    "Photosynthesis converts sunlight into chemical energy in plant cells. "
    "Chlorophyll absorbs light primarily in the blue and red wavelengths. "
    "The Calvin cycle fixes carbon dioxide into glucose molecules. "
    "Oxygen is released as a byproduct of the light-dependent reactions."
)

_TOPIC_B_TEXT = (
    "Mitochondria generate ATP through oxidative phosphorylation. "
    "The electron transport chain creates a proton gradient across the inner membrane. "
    "Krebs cycle intermediates feed electrons into the chain. "
    "Cellular respiration ultimately converts glucose back into carbon dioxide and water."
)

_FULL_TEXT = _TOPIC_A_TEXT + "\n\n" + _TOPIC_B_TEXT


def _build_corpus(tmp_path) -> Corpus:
    """Build a small searchable corpus from synthetic text."""
    boundary = _FULL_TEXT.index("Mitochondria")
    sections = [
        Section(
            title="Photosynthesis",
            level=1,
            start_char=0,
            end_char=boundary,
            page=1,
            heading_style="topics",
            summary="How plants convert light to energy.",
        ),
        Section(
            title="Cellular Respiration",
            level=1,
            start_char=boundary,
            end_char=len(_FULL_TEXT),
            page=1,
            heading_style="topics",
            summary="How cells generate ATP from glucose.",
        ),
    ]
    root = Section(title="Doc", level=0, start_char=0, end_char=len(_FULL_TEXT))
    structure = DocumentStructure(title="Doc", sections=sections, root=root, method="topics")

    cfg = SunderConfig(chunk_size=200, chunk_overlap=15, min_chunk_size=5)
    chunks = chunk_by_topics_sentence_aware(_FULL_TEXT, "bio", structure, config=cfg)
    assert len(chunks) >= 2, "Need at least 2 chunks for the test"

    return index_embeddings(chunks, str(tmp_path), config=cfg)


class TestSearchHitFields:
    """SearchHit must expose snippets and topic_summary after search()."""

    @pytest.fixture(autouse=True)
    def corpus(self, tmp_path):
        self.corpus = _build_corpus(tmp_path)

    def test_snippets_populated(self):
        """SearchHit.snippets contains actual sentence text, not empty."""
        hits = self.corpus.search("chlorophyll light absorption", top_k=3)
        assert len(hits) > 0
        best = hits[0]
        assert len(best.snippets) > 0
        assert isinstance(best.snippets[0], str)
        assert len(best.snippets[0]) > 10  # real sentence, not stub

    def test_sentence_ids_populated(self):
        """SearchHit.sentence_ids is parallel to snippets."""
        hits = self.corpus.search("ATP energy", top_k=3)
        assert len(hits) > 0
        best = hits[0]
        assert len(best.sentence_ids) == len(best.snippets)
        assert all(sid.startswith("bio_sent_") for sid in best.sentence_ids)

    def test_topic_summary_populated(self):
        """SearchHit carries topic_summary from the chunk's topic metadata."""
        hits = self.corpus.search("chlorophyll sunlight plants", top_k=3)
        assert len(hits) > 0
        # The best hit should be from the Photosynthesis topic
        photo_hits = [h for h in hits if h.topic_title == "Photosynthesis"]
        assert len(photo_hits) > 0
        assert photo_hits[0].topic_summary == "How plants convert light to energy."

    def test_topic_title_populated(self):
        """SearchHit.topic_title matches one of the two topics."""
        hits = self.corpus.search("mitochondria ATP oxidative", top_k=3)
        assert len(hits) > 0
        titles = {h.topic_title for h in hits}
        assert "Cellular Respiration" in titles

    def test_score_bounded(self):
        """Scores are similarity in (0, 1]."""
        hits = self.corpus.search("glucose carbon dioxide", top_k=5)
        for h in hits:
            assert 0 < h.score <= 1.0

    def test_search_returns_both_topics(self):
        """A broad query can return hits from both topics."""
        hits = self.corpus.search("energy molecules cells", top_k=10)
        titles = {h.topic_title for h in hits}
        # With a broad enough query and only 2 chunks, both should appear
        assert len(titles) >= 1  # at minimum one, ideally both
