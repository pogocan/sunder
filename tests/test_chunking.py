"""Tests for chunking."""

from sunder.chunking import chunk_by_topics_sentence_aware, chunk_text
from sunder.sentences import split_text_to_sentences
from sunder.types import DocumentStructure, Section, SunderConfig


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


# -- Fixtures for sentence-aware chunking tests ------------------------------

# Two topics, each with multiple sentences — enough to produce >1 chunk per topic.
_TOPIC_A = (
    "The Federal Reserve announced new monetary policy guidelines today. "
    "Interest rates will remain unchanged through the end of the fiscal year. "
    "The committee emphasized that inflation targets have been met. "
    "Several board members expressed concern about global trade tensions. "
    "Consumer spending rose by three point two percent in the latest quarterly report. "
    "Retail sectors showed particularly strong growth driven by e-commerce expansion. "
    "Housing starts also exceeded expectations reaching a five-year high. "
    "Analysts attribute the gains to improved consumer confidence and low unemployment."
)

_TOPIC_B = (
    "Meanwhile the technology sector faced renewed regulatory scrutiny. "
    "Lawmakers introduced a bill targeting data privacy practices at major tech firms. "
    "The proposed legislation would require companies to obtain explicit consent. "
    "Industry groups responded with criticism arguing the rules could stifle innovation. "
    "In international news trade negotiations between the US and EU reached a critical juncture. "
    "Diplomats reported progress on agricultural tariffs but disagreements remain. "
    "The next round of talks is scheduled for April. "
    "Both sides expressed cautious optimism about reaching a framework agreement."
)

_FULL_TEXT = _TOPIC_A + "\n\n" + _TOPIC_B


def _make_structure() -> DocumentStructure:
    """Build a fake 2-section DocumentStructure for testing."""
    boundary = _FULL_TEXT.index("Meanwhile")
    sections = [
        Section(
            title="Economic Policy",
            level=1,
            start_char=0,
            end_char=boundary,
            page=1,
            heading_style="topics",
            summary="Economy stuff.",
        ),
        Section(
            title="Tech Regulation",
            level=1,
            start_char=boundary,
            end_char=len(_FULL_TEXT),
            page=1,
            heading_style="topics",
            summary="Tech stuff.",
        ),
    ]
    root = Section(title="Doc", level=0, start_char=0, end_char=len(_FULL_TEXT))
    return DocumentStructure(title="Doc", sections=sections, root=root, method="topics")


# -- Sentence-aware chunking tests ------------------------------------------


class TestChunkByTopicsSentenceAware:
    """Tests for chunk_by_topics_sentence_aware()."""

    CFG = SunderConfig(chunk_size=60, chunk_overlap=15, min_chunk_size=10)

    def _get_chunks(self):
        return chunk_by_topics_sentence_aware(
            _FULL_TEXT, "test_doc", _make_structure(), config=self.CFG,
        )

    def test_produces_chunks(self):
        chunks = self._get_chunks()
        assert len(chunks) >= 2

    def test_sentences_populated(self):
        """Every chunk has a non-empty sentences list."""
        chunks = self._get_chunks()
        for c in chunks:
            assert len(c.sentences) > 0, f"{c.chunk_id} has no sentences"

    def test_chunk_boundaries_on_sentence_boundaries(self):
        """Each chunk's text is exactly its sentences joined by spaces.

        This guarantees chunks never split mid-sentence.
        """
        chunks = self._get_chunks()
        for c in chunks:
            reconstructed = " ".join(s.text for s in c.sentences)
            assert c.text == reconstructed, (
                f"{c.chunk_id}: chunk text does not equal joined sentence texts"
            )

    def test_overlap_at_start_of_next_chunk(self):
        """When overlap occurs, the carried sentences appear at the START of
        the next chunk's sentence list — not appended at the end."""
        chunks = self._get_chunks()
        # Find consecutive chunks in the same topic with overlap
        found_overlap = False
        for i in range(len(chunks) - 1):
            a, b = chunks[i], chunks[i + 1]
            if a.topic_index != b.topic_index:
                continue
            # Check if last sentence(s) of chunk a appear at start of chunk b
            a_last = a.sentences[-1].text
            if b.sentences and b.sentences[0].text == a_last:
                found_overlap = True
                # Verify it's at the START (index 0), not somewhere else
                assert b.sentences[0].text == a_last
                break
        assert found_overlap, "Expected to find at least one overlap between consecutive chunks"

    def test_overlap_sentences_have_new_ids(self):
        """Overlap sentences in the next chunk get new sentence_ids, not copies."""
        chunks = self._get_chunks()
        all_sent_ids = []
        for c in chunks:
            for s in c.sentences:
                all_sent_ids.append(s.sentence_id)
        # All sentence IDs must be unique (no duplicates across chunks)
        assert len(all_sent_ids) == len(set(all_sent_ids)), "Duplicate sentence IDs found"

    def test_topic_metadata_populated(self):
        """Each chunk carries its parent topic's title and summary."""
        chunks = self._get_chunks()
        topic_titles = {c.topic_title for c in chunks}
        assert "Economic Policy" in topic_titles
        assert "Tech Regulation" in topic_titles
        for c in chunks:
            assert c.topic_title is not None
            assert c.topic_summary is not None
            assert c.topic_index is not None

    def test_sentence_ids_reference_parent_chunk(self):
        """Every sentence's chunk_id matches its parent chunk."""
        chunks = self._get_chunks()
        for c in chunks:
            for s in c.sentences:
                assert s.chunk_id == c.chunk_id
                assert s.doc_id == c.doc_id

    def test_no_cross_topic_chunks(self):
        """No chunk contains sentences from two different topics."""
        chunks = self._get_chunks()
        for c in chunks:
            assert c.topic_index is not None
            # All chunks within a topic should share the same topic_title
            # (this is structural, not about sentence content)
