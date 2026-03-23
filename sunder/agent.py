"""
sunder.agent -- Domain-agnostic retrieval agent.

Takes a Corpus and answers questions by progressively retrieving and
reading chunks. Uses an LLM to decide which chunks to open, when to
call tools, and when to produce a final answer.

Usage:
    agent = Agent(corpus)
    result = agent.ask("What is this document about?")
    print(result.answer)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_types import AgentMemory, AgentResult, AgentTool
from .index import Corpus
from .llm import LLMProvider, get_provider
from .types import AgentConfig, SearchHit


# -- Keyword search -----------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation."""
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]


def keyword_search(query: str, corpus: Corpus, top_k: int = 10) -> list[SearchHit]:
    """Score every chunk by token overlap with the query.

    Returns synthetic SearchHit objects with score = matching_tokens / query_tokens.
    Only returns chunks with score > 0.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for chunk_id, cd in corpus._chunks_data.items():
        chunk_tokens = set(_tokenize(cd.get("text", "")))
        overlap = len(query_tokens & chunk_tokens)
        if overlap > 0:
            score = overlap / len(query_tokens)
            scored.append((score, chunk_id))

    scored.sort(key=lambda x: x[0], reverse=True)

    hits = []
    for score, chunk_id in scored[:top_k]:
        cd = corpus._chunks_data.get(chunk_id, {})
        hits.append(SearchHit(
            chunk_id=chunk_id,
            doc_id=cd.get("doc_id", ""),
            score=score,
            snippets=[cd.get("text", "")[:200]],
            sentence_ids=[],
            topic_title=cd.get("topic_title"),
            topic_summary=cd.get("topic_summary"),
        ))
    return hits


def merge_hits(
    semantic: list[SearchHit],
    keyword: list[SearchHit],
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[SearchHit]:
    """Merge semantic and keyword hits by weighted score.

    Deduplicates by chunk_id. The SearchHit with richer metadata wins (semantic
    preferred since it has real snippets).
    """
    by_id: dict[str, dict[str, Any]] = {}

    for hit in semantic:
        by_id[hit.chunk_id] = {
            "hit": hit,
            "semantic": hit.score,
            "keyword": 0.0,
        }

    for hit in keyword:
        if hit.chunk_id in by_id:
            by_id[hit.chunk_id]["keyword"] = hit.score
        else:
            by_id[hit.chunk_id] = {
                "hit": hit,
                "semantic": 0.0,
                "keyword": hit.score,
            }

    merged: list[tuple[float, SearchHit]] = []
    for entry in by_id.values():
        final_score = (
            semantic_weight * entry["semantic"]
            + keyword_weight * entry["keyword"]
        )
        hit = entry["hit"]
        hit.score = final_score
        merged.append((final_score, hit))

    merged.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in merged]


# -- Tool schema for LLM -----------------------------------------------------

_AGENT_TOOL = {
    "name": "agent_action",
    "description": (
        "Choose one action: read a chunk, call a tool, rewrite the search query, "
        "decompose into sub-queries, or produce a final answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "read_chunk", "call_tool", "answer",
                    "rewrite_search", "decompose",
                ],
                "description": "The action to take.",
            },
            "chunk_id": {
                "type": "string",
                "description": "For read_chunk: the chunk_id to open.",
            },
            "tool_name": {
                "type": "string",
                "description": "For call_tool: which tool to invoke.",
            },
            "tool_query": {
                "type": "string",
                "description": "For call_tool: the query to pass to the tool.",
            },
            "new_query": {
                "type": "string",
                "description": "For rewrite_search: the rewritten query to search with.",
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For decompose: list of sub-queries to search independently and merge.",
            },
            "answer_text": {
                "type": "string",
                "description": "For answer: the final answer text.",
            },
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For answer: chunk_ids that support the answer.",
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For answer: 2-3 suggested follow-up questions.",
            },
        },
        "required": ["action"],
    },
}


# -- System prompt ------------------------------------------------------------

_SYSTEM_PROMPT = """You are a document analysis agent. You answer questions by reading chunks from an indexed document corpus.

You will be shown search results (snippets only — not full text). To see the complete text of a chunk, use read_chunk with its chunk_id. You have a limited number of reads, so choose wisely.

Available actions:
- read_chunk: Open the full text of a chunk. Use this when a snippet looks relevant but you need more context.
- call_tool: Call an external tool. Only available if tools were provided.
- rewrite_search: Re-run the search with a better query. Use when the initial results don't match what you need. Provide new_query. Limited number of rewrites allowed.
- decompose: Break the question into sub-queries and search for each. Provide sub_queries (list of strings). Each sub-query is searched independently and results are merged. Counts as one rewrite.
- answer: Produce your final answer with citations.

Rules:
- Always cite the chunk_ids that support your answer.
- If the search results and any chunks you've read don't contain enough information to answer, say so honestly.
- Be concise and direct. Answer the question, don't summarize the entire document.
- Suggest 2-3 follow-up questions the user might want to ask.
- Each read_chunk counts toward your read limit — don't read chunks unless the snippet suggests they're relevant.
- Use rewrite_search or decompose when the current results seem off-topic or incomplete."""


# -- Agent --------------------------------------------------------------------

class Agent:
    """Domain-agnostic retrieval agent over a sunder Corpus."""

    def __init__(
        self,
        corpus: Corpus,
        config: AgentConfig | None = None,
        tools: list[AgentTool] | None = None,
        memory: AgentMemory | None = None,
    ):
        self.corpus = corpus
        self.config = config or AgentConfig()
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory
        self._provider: LLMProvider | None = None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self.config)
        return self._provider

    def _run_search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Run semantic + keyword search and merge results."""
        semantic_hits = self.corpus.search(query, top_k=top_k)
        kw_hits = keyword_search(query, self.corpus, top_k=top_k)
        return merge_hits(semantic_hits, kw_hits)

    @staticmethod
    def _merge_hit_pools(
        existing: list[SearchHit],
        new: list[SearchHit],
    ) -> list[SearchHit]:
        """Merge two hit pools, keeping the best score per chunk_id."""
        by_id: dict[str, SearchHit] = {}
        for hit in existing:
            if hit.chunk_id not in by_id or hit.score > by_id[hit.chunk_id].score:
                by_id[hit.chunk_id] = hit
        for hit in new:
            if hit.chunk_id not in by_id or hit.score > by_id[hit.chunk_id].score:
                by_id[hit.chunk_id] = hit
        merged = sorted(by_id.values(), key=lambda h: h.score, reverse=True)
        return merged

    def ask(self, question: str) -> AgentResult:
        """Ask a question and get an answer with citations."""
        run_id = uuid.uuid4().hex
        steps_log: list[dict] = []
        chunks_read: set[str] = set()
        step_count = 0
        read_count = 0
        rewrite_count = 0

        max_steps = self.config.max_steps
        max_reads = self.config.max_reads
        max_rewrites = self.config.max_rewrites

        # -- Step 1: Pre-load memory context --
        memory_context = self._build_memory_context(question)

        # -- Step 2: Search --
        merged = self._run_search(question)

        steps_log.append({
            "type": "search",
            "query": question,
            "hits": len(merged),
        })

        # Build snippet context for LLM
        snippet_context = self._format_snippets(merged[:10])

        # Build tool descriptions
        tool_desc = ""
        if self.tools:
            tool_lines = []
            for name, tool in self.tools.items():
                tool_lines.append(f"- {name}: {tool.description}")
            tool_desc = "\nAvailable tools:\n" + "\n".join(tool_lines) + "\n"

        # -- Step 3: Agent loop --
        messages: list[dict] = []

        user_content = f"Question: {question}\n\n"
        if memory_context:
            user_content += memory_context + "\n\n"
        user_content += f"Search results:\n{snippet_context}"
        if tool_desc:
            user_content += f"\n{tool_desc}"
        user_content += f"\n\nYou may read up to {max_reads} chunks. Choose your actions."

        messages.append({"role": "user", "content": user_content})

        answer_text = ""
        cited_ids: list[str] = []
        follow_ups: list[str] = []

        while step_count < max_steps:
            result = self.provider.complete_with_tool(
                messages=messages,
                tools=[_AGENT_TOOL],
                tool_choice="agent_action",
                system=_SYSTEM_PROMPT,
                max_tokens=4096,
                temperature=0.0,
            )

            action = result.input.get("action", "answer")
            step_count += 1

            if action == "read_chunk":
                chunk_id = result.input.get("chunk_id", "")

                if chunk_id in chunks_read:
                    # Already read — tell the LLM and continue
                    messages.append({"role": "assistant", "content": f"[Already read {chunk_id}]"})
                    messages.append({
                        "role": "user",
                        "content": f"You already read {chunk_id}. Choose a different action.",
                    })
                    continue

                if read_count >= max_reads:
                    # Limit hit — force answer
                    messages.append({"role": "assistant", "content": "[Read limit reached]"})
                    messages.append({
                        "role": "user",
                        "content": "Read limit reached. Produce your best answer now from what you've read.",
                    })
                    continue

                chunk_data = self.corpus.get_chunk(chunk_id)
                if chunk_data is None:
                    messages.append({"role": "assistant", "content": f"[Chunk {chunk_id} not found]"})
                    messages.append({
                        "role": "user",
                        "content": f"Chunk {chunk_id} not found. Try a different chunk or answer.",
                    })
                    continue

                chunks_read.add(chunk_id)
                read_count += 1

                chunk_text = chunk_data.get("text", "")
                word_count = len(chunk_text.split())

                steps_log.append({
                    "type": "read_chunk",
                    "chunk_id": chunk_id,
                    "words": word_count,
                })

                topic_info = ""
                tt = chunk_data.get("topic_title")
                if tt:
                    topic_info = f" [Topic: {tt}]"

                messages.append({
                    "role": "assistant",
                    "content": f"[Reading chunk {chunk_id}]",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Chunk {chunk_id}{topic_info} ({word_count} words):\n\n"
                        f"{chunk_text}\n\n"
                        f"Reads remaining: {max_reads - read_count}. "
                        f"Continue reading or produce your answer."
                    ),
                })

            elif action == "call_tool":
                tool_name = result.input.get("tool_name", "")
                tool_query = result.input.get("tool_query", "")

                if tool_name not in self.tools:
                    messages.append({"role": "assistant", "content": f"[Tool {tool_name} not found]"})
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' is not available. Choose a different action.",
                    })
                    continue

                tool_result = self.tools[tool_name].func(tool_query)

                steps_log.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "query": tool_query,
                })

                messages.append({
                    "role": "assistant",
                    "content": f"[Calling tool {tool_name}]",
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' result:\n{tool_result}\n\nContinue or produce your answer.",
                })

            elif action == "rewrite_search":
                new_query = result.input.get("new_query", "")

                if rewrite_count >= max_rewrites:
                    messages.append({"role": "assistant", "content": "[Rewrite limit reached]"})
                    messages.append({
                        "role": "user",
                        "content": "Rewrite limit reached. Work with the current results or produce your answer.",
                    })
                    continue

                new_hits = self._run_search(new_query)
                merged = self._merge_hit_pools(merged, new_hits)
                rewrite_count += 1

                steps_log.append({
                    "type": "rewrite_search",
                    "original": question,
                    "rewritten": new_query,
                    "new_hits": len(new_hits),
                })

                snippet_context = self._format_snippets(merged[:10])
                messages.append({
                    "role": "assistant",
                    "content": f"[Rewriting search: {new_query}]",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Updated search results:\n{snippet_context}\n\n"
                        f"Rewrites remaining: {max_rewrites - rewrite_count}. "
                        f"Continue reading or produce your answer."
                    ),
                })

            elif action == "decompose":
                sub_queries = result.input.get("sub_queries", [])

                if rewrite_count >= max_rewrites:
                    messages.append({"role": "assistant", "content": "[Rewrite limit reached]"})
                    messages.append({
                        "role": "user",
                        "content": "Rewrite limit reached. Work with the current results or produce your answer.",
                    })
                    continue

                all_new: list[SearchHit] = []
                for sq in sub_queries:
                    all_new.extend(self._run_search(sq))
                merged = self._merge_hit_pools(merged, all_new)
                rewrite_count += 1

                steps_log.append({
                    "type": "decompose",
                    "sub_queries": sub_queries,
                    "new_hits": len(all_new),
                })

                snippet_context = self._format_snippets(merged[:10])
                messages.append({
                    "role": "assistant",
                    "content": f"[Decomposed into {len(sub_queries)} sub-queries]",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"Updated search results:\n{snippet_context}\n\n"
                        f"Rewrites remaining: {max_rewrites - rewrite_count}. "
                        f"Continue reading or produce your answer."
                    ),
                })

            elif action == "answer":
                answer_text = result.input.get("answer_text", "")
                cited_ids = result.input.get("cited_chunk_ids", [])
                follow_ups = result.input.get("follow_up_questions", [])

                steps_log.append({
                    "type": "answer",
                    "cited": cited_ids,
                })
                break

            else:
                # Unknown action — force answer
                break

        # If loop exhausted without an answer, produce a fallback
        if not answer_text:
            answer_text = self._force_answer(messages)
            steps_log.append({"type": "answer", "cited": cited_ids})

        # -- Step 4: Post-answer --
        if self.memory and self.memory.answer_cache is not None:
            self.memory.answer_cache.append({
                "question": question,
                "answer": answer_text,
                "chunk_ids": cited_ids,
            })

        agent_result = AgentResult(
            answer=answer_text,
            citations=cited_ids,
            chunks_read=sorted(chunks_read),
            follow_ups=follow_ups[:3],
            steps=step_count,
            run_id=run_id,
        )

        # Write run log
        self._write_run_log(run_id, question, steps_log, agent_result)

        return agent_result

    # -- Internal helpers -----------------------------------------------------

    def _build_memory_context(self, question: str) -> str:
        """Build context string from memory, if provided."""
        if not self.memory:
            return ""

        parts = []

        # Knowledge sheet
        if self.memory.knowledge_sheet:
            lines = [f"- {k}: {v}" for k, v in self.memory.knowledge_sheet.items()]
            parts.append("Known facts:\n" + "\n".join(lines))

        # Past answers (simple text match — embedding-based matching deferred
        # to when the agent has access to the embedding model at ask() time)
        if self.memory.answer_cache:
            q_tokens = set(_tokenize(question))
            scored = []
            for entry in self.memory.answer_cache:
                past_tokens = set(_tokenize(entry.get("question", "")))
                if q_tokens and past_tokens:
                    overlap = len(q_tokens & past_tokens) / max(len(q_tokens), len(past_tokens))
                    if overlap > 0.3:
                        scored.append((overlap, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                cache_lines = []
                for _, entry in scored[:3]:
                    cache_lines.append(
                        f"Q: {entry['question']}\n"
                        f"A: {entry['answer'][:300]}"
                    )
                parts.append("Relevant past answers:\n" + "\n---\n".join(cache_lines))

        # User notes
        if self.memory.user_notes:
            parts.append(f"User notes:\n{self.memory.user_notes}")

        return "\n\n".join(parts)

    def _format_snippets(self, hits: list[SearchHit]) -> str:
        """Format merged search hits as snippet context for the LLM."""
        lines = []
        for i, hit in enumerate(hits):
            snippet = hit.snippets[0] if hit.snippets else "(no snippet)"
            topic = f" [Topic: {hit.topic_title}]" if hit.topic_title else ""
            summary = f"\n  Summary: {hit.topic_summary}" if hit.topic_summary else ""
            lines.append(
                f"{i+1}. [{hit.chunk_id}]{topic} (score: {hit.score:.3f})\n"
                f"  \"{snippet}\"{summary}"
            )
        return "\n".join(lines) if lines else "(no search results)"

    def _force_answer(self, messages: list[dict]) -> str:
        """Force an answer when the loop exhausts without one."""
        messages.append({
            "role": "user",
            "content": (
                "You have run out of steps. Based on everything you've seen so far, "
                "produce your best answer now. Be honest about limitations."
            ),
        })
        return self.provider.complete(
            messages=messages,
            system=_SYSTEM_PROMPT,
            max_tokens=2048,
        )

    def _write_run_log(
        self,
        run_id: str,
        question: str,
        steps_log: list[dict],
        result: AgentResult,
    ) -> None:
        """Write run log to {corpus.output_dir}/agent_runs/run_{uuid}.json."""
        runs_dir = Path(self.corpus.output_dir) / "agent_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        log = {
            "run_id": run_id,
            "question": question,
            "steps": steps_log,
            "answer": result.answer,
            "citations": result.citations,
            "chunks_read": result.chunks_read,
            "follow_ups": result.follow_ups,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log_path = runs_dir / f"run_{run_id}.json"
        log_path.write_text(
            json.dumps(log, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
