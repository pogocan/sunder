"""Tests for sunder.agent -- Agent class, keyword search, hit merging, run logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sunder.agent import Agent, keyword_search, merge_hits
from sunder.agent_types import AgentMemory, AgentResult, AgentTool
from sunder.chunking import chunk_by_topics_sentence_aware, chunk_text
from sunder.index import Corpus, index_embeddings
from sunder.llm import LLMProvider, ToolResult
from sunder.types import (
    AgentConfig,
    DocumentStructure,
    SearchHit,
    Section,
    SunderConfig,
)


# -- Synthetic corpus fixtures ------------------------------------------------

_TOPIC_A = (
    "Photosynthesis converts sunlight into chemical energy in plant cells. "
    "Chlorophyll absorbs light primarily in the blue and red wavelengths. "
    "The Calvin cycle fixes carbon dioxide into glucose molecules. "
    "Oxygen is released as a byproduct of the light-dependent reactions."
)

_TOPIC_B = (
    "Mitochondria generate ATP through oxidative phosphorylation. "
    "The electron transport chain creates a proton gradient across the inner membrane. "
    "Krebs cycle intermediates feed electrons into the chain. "
    "Cellular respiration ultimately converts glucose back into carbon dioxide and water."
)

_FULL_TEXT = _TOPIC_A + "\n\n" + _TOPIC_B


def _build_topic_corpus(tmp_path: Path) -> Corpus:
    """Build a small corpus with 2 topics."""
    boundary = _FULL_TEXT.index("Mitochondria")
    sections = [
        Section(title="Photosynthesis", level=1, start_char=0,
                end_char=boundary, page=1, heading_style="topics",
                summary="How plants convert light to energy."),
        Section(title="Cellular Respiration", level=1, start_char=boundary,
                end_char=len(_FULL_TEXT), page=1, heading_style="topics",
                summary="How cells generate ATP from glucose."),
    ]
    root = Section(title="Doc", level=0, start_char=0, end_char=len(_FULL_TEXT))
    structure = DocumentStructure(title="Doc", sections=sections, root=root, method="topics")
    cfg = SunderConfig(chunk_size=200, chunk_overlap=15, min_chunk_size=5)
    chunks = chunk_by_topics_sentence_aware(_FULL_TEXT, "bio", structure, config=cfg)
    return index_embeddings(chunks, str(tmp_path / "corpus"), config=cfg)


def _build_flat_corpus(tmp_path: Path) -> Corpus:
    """Build a corpus with flat chunking (no topic fields)."""
    cfg = SunderConfig(chunking_mode="flat", chunk_size=200, chunk_overlap=15, min_chunk_size=5)
    chunks = chunk_text(_FULL_TEXT, "bio", config=cfg)
    # Manually segment sentences for flat chunks
    from sunder.sentences import segment_sentences
    segment_sentences(chunks, config=cfg)
    return index_embeddings(chunks, str(tmp_path / "corpus_flat"), config=cfg)


# -- Mock LLM provider -------------------------------------------------------

class MockProvider(LLMProvider):
    """A mock LLM that follows a scripted action sequence.

    actions: list of dicts, each representing what complete_with_tool should return.
    Falls through to an answer action if the script runs out.
    """

    name = "mock"

    def __init__(self, actions: list[dict] | None = None):
        self._actions = list(actions or [])
        self._call_count = 0

    def complete(self, messages, system="", max_tokens=4096, temperature=0.0):
        return "Forced answer: based on the available context."

    def complete_with_tool(self, messages, tools, tool_choice,
                           system="", max_tokens=4096, temperature=0.0):
        if self._call_count < len(self._actions):
            action = self._actions[self._call_count]
        else:
            # Default: produce answer
            action = {
                "action": "answer",
                "answer_text": "Default mock answer.",
                "cited_chunk_ids": [],
                "follow_up_questions": ["What else?"],
            }
        self._call_count += 1
        return ToolResult(tool_name=tool_choice, input=action)


def _make_answer_action(text="Mock answer.", cited=None, follow_ups=None):
    return {
        "action": "answer",
        "answer_text": text,
        "cited_chunk_ids": cited or [],
        "follow_up_questions": follow_ups or ["Follow up 1?", "Follow up 2?"],
    }


def _make_read_action(chunk_id):
    return {"action": "read_chunk", "chunk_id": chunk_id}


def _make_tool_action(tool_name, query):
    return {"action": "call_tool", "tool_name": tool_name, "tool_query": query}


# -- Tests --------------------------------------------------------------------

class TestAgentConstruction:

    def test_constructs_with_corpus_only(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus)
        assert agent.corpus is corpus
        assert agent.config.max_steps == 10
        assert agent.config.max_reads == 4
        assert agent.memory is None
        assert len(agent.tools) == 0

    def test_constructs_with_all_options(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        tool = AgentTool(name="lookup", description="Look up a term", func=lambda q: "result")
        memory = AgentMemory(knowledge_sheet={"key": "val"})
        cfg = AgentConfig(max_steps=5, max_reads=2)
        agent = Agent(corpus, config=cfg, tools=[tool], memory=memory)
        assert agent.config.max_steps == 5
        assert "lookup" in agent.tools
        assert agent.memory is memory


class TestAgentAsk:

    def test_returns_agent_result(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus, config=AgentConfig(max_steps=3, max_reads=2))
        agent._provider = MockProvider([_make_answer_action("Test answer.")])
        result = agent.ask("What is photosynthesis?")
        assert isinstance(result, AgentResult)
        assert result.answer == "Test answer."
        assert result.run_id  # non-empty
        assert result.steps == 1

    def test_answer_has_follow_ups(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus, config=AgentConfig(max_steps=3))
        agent._provider = MockProvider([
            _make_answer_action(follow_ups=["Q1?", "Q2?", "Q3?"]),
        ])
        result = agent.ask("question")
        assert len(result.follow_ups) <= 3
        assert len(result.follow_ups) >= 1

    def test_read_chunk_then_answer(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        # Get a real chunk_id from the corpus
        chunk_id = list(corpus._chunks_data.keys())[0]
        agent = Agent(corpus, config=AgentConfig(max_steps=5, max_reads=2))
        agent._provider = MockProvider([
            _make_read_action(chunk_id),
            _make_answer_action("Answer after reading.", cited=[chunk_id]),
        ])
        result = agent.ask("question")
        assert chunk_id in result.chunks_read
        assert result.steps == 2


class TestMaxLimits:

    def test_chunks_read_never_exceeds_max_reads(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        chunk_ids = list(corpus._chunks_data.keys())
        # Script: try to read 5 chunks, but max_reads=2
        actions = [_make_read_action(cid) for cid in chunk_ids[:5]]
        actions.append(_make_answer_action())
        agent = Agent(corpus, config=AgentConfig(max_steps=10, max_reads=2))
        agent._provider = MockProvider(actions)
        result = agent.ask("question")
        assert len(result.chunks_read) <= 2

    def test_steps_never_exceeds_max_steps(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        chunk_ids = list(corpus._chunks_data.keys())
        # Script: many reads, never answer — should be cut off by max_steps
        actions = [_make_read_action(chunk_ids[i % len(chunk_ids)]) for i in range(20)]
        agent = Agent(corpus, config=AgentConfig(max_steps=3, max_reads=10))
        agent._provider = MockProvider(actions)
        result = agent.ask("question")
        assert result.steps <= 3
        assert result.answer  # should have a forced answer


class TestNoDuplicateReads:

    def test_chunk_not_read_twice(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        chunk_id = list(corpus._chunks_data.keys())[0]
        # Script: read same chunk twice, then answer
        actions = [
            _make_read_action(chunk_id),
            _make_read_action(chunk_id),  # duplicate — should be skipped
            _make_answer_action(cited=[chunk_id]),
        ]
        agent = Agent(corpus, config=AgentConfig(max_steps=10, max_reads=5))
        agent._provider = MockProvider(actions)
        result = agent.ask("question")
        assert result.chunks_read.count(chunk_id) == 1


class TestToolCalls:

    def test_tool_gets_called(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        call_log = []

        def my_tool(query: str) -> str:
            call_log.append(query)
            return f"Tool says: {query}"

        tool = AgentTool(name="lookup", description="Look up", func=my_tool)
        agent = Agent(corpus, config=AgentConfig(max_steps=5), tools=[tool])
        agent._provider = MockProvider([
            _make_tool_action("lookup", "ATP"),
            _make_answer_action("ATP is energy."),
        ])
        result = agent.ask("What is ATP?")
        assert len(call_log) == 1
        assert call_log[0] == "ATP"
        assert result.steps == 2

    def test_unknown_tool_handled(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus, config=AgentConfig(max_steps=5))
        agent._provider = MockProvider([
            _make_tool_action("nonexistent", "query"),
            _make_answer_action(),
        ])
        result = agent.ask("question")
        # Should not crash, should produce answer
        assert result.answer


class TestRunLog:

    def test_run_log_created(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus, config=AgentConfig(max_steps=3))
        agent._provider = MockProvider([_make_answer_action()])
        result = agent.ask("question")

        runs_dir = Path(corpus.output_dir) / "agent_runs"
        assert runs_dir.exists()

        log_path = runs_dir / f"run_{result.run_id}.json"
        assert log_path.exists()

        log = json.loads(log_path.read_text(encoding="utf-8"))
        assert log["run_id"] == result.run_id
        assert log["question"] == "question"
        assert isinstance(log["steps"], list)
        assert log["answer"] == result.answer
        assert "timestamp" in log


class TestMemoryOptional:

    def test_works_without_memory(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        agent = Agent(corpus, memory=None)
        agent._provider = MockProvider([_make_answer_action()])
        result = agent.ask("question")
        assert result.answer

    def test_memory_cache_updated(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        memory = AgentMemory()
        agent = Agent(corpus, memory=memory)
        agent._provider = MockProvider([_make_answer_action("Cached answer.")])
        agent.ask("What is X?")
        assert len(memory.answer_cache) == 1
        assert memory.answer_cache[0]["question"] == "What is X?"
        assert memory.answer_cache[0]["answer"] == "Cached answer."


class TestFlatCorpus:

    def test_works_with_flat_chunking(self, tmp_path):
        """Agent works when corpus has no topic fields (chunking_mode=flat)."""
        corpus = _build_flat_corpus(tmp_path)
        agent = Agent(corpus, config=AgentConfig(max_steps=3))
        agent._provider = MockProvider([_make_answer_action("Flat answer.")])
        result = agent.ask("question")
        assert result.answer == "Flat answer."

    def test_flat_corpus_search_returns_hits(self, tmp_path):
        corpus = _build_flat_corpus(tmp_path)
        hits = corpus.search("photosynthesis", top_k=5)
        assert len(hits) > 0
        # topic_title should be None for flat chunks
        for h in hits:
            assert h.topic_title is None


class TestKeywordSearch:

    def test_keyword_search_returns_hits(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        hits = keyword_search("photosynthesis chlorophyll sunlight", corpus, top_k=5)
        assert len(hits) > 0
        assert all(h.score > 0 for h in hits)

    def test_keyword_search_empty_query(self, tmp_path):
        corpus = _build_topic_corpus(tmp_path)
        hits = keyword_search("", corpus)
        assert len(hits) == 0


class TestMergeHits:

    def test_merge_deduplicates(self):
        h1 = SearchHit(chunk_id="a", doc_id="d", score=0.8,
                        snippets=["s1"], sentence_ids=["sid1"])
        h2 = SearchHit(chunk_id="a", doc_id="d", score=0.6,
                        snippets=["s2"], sentence_ids=["sid2"])
        h3 = SearchHit(chunk_id="b", doc_id="d", score=0.5,
                        snippets=["s3"], sentence_ids=["sid3"])
        merged = merge_hits([h1], [h2, h3])
        ids = [h.chunk_id for h in merged]
        assert ids.count("a") == 1
        assert "b" in ids

    def test_merge_weighted_score(self):
        sem = SearchHit(chunk_id="x", doc_id="d", score=1.0,
                         snippets=["s"], sentence_ids=["sid"])
        kw = SearchHit(chunk_id="x", doc_id="d", score=1.0,
                        snippets=["s"], sentence_ids=["sid"])
        merged = merge_hits([sem], [kw], semantic_weight=0.6, keyword_weight=0.4)
        assert len(merged) == 1
        assert abs(merged[0].score - 1.0) < 0.01

    def test_merge_empty_inputs(self):
        assert merge_hits([], []) == []
