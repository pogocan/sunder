"""
sunder.types -- Core data types for document analysis pipeline.

Three-tier retrieval model:
  Topic  -> structural metadata (which concept/section)
  Chunk  -> readable context unit (what a caller opens and reads)
  Sentence -> retrieval unit (what gets embedded and searched)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# -- Pipeline Configuration ---------------------------------------------------

@dataclass
class SunderConfig:
    """Configuration governing all pipeline stages.

    Passed through the pipeline so every stage uses consistent parameters.
    Persisted alongside the corpus so you know exactly what produced it.
    """
    # Chunking
    chunking_mode: str = "topic"    # "topic" (LLM-guided) or "flat" (no LLM)
    chunk_size: int = 200           # target words per chunk
    chunk_overlap: int = 30         # overlap in words between chunks
    min_chunk_size: int = 50        # discard chunks smaller than this (words)

    # Sentence segmentation
    atomic_line_length: int = 150   # lines shorter than this are never split

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384        # must match model output
    embedding_batch_size: int = 32

    # Graph extraction
    graph_max_chunks: int | None = None  # limit chunks for KG extraction (None = all)

    # Search
    default_top_k: int = 10
    search_oversample: int = 3      # fetch top_k * this, then group by chunk


# -- Document Structure -------------------------------------------------------

@dataclass
class Section:
    """A section/topic in a document."""
    title: str
    level: int              # 0 = root, 1 = part/chapter, 2 = item/section, 3 = subsection
    start_char: int         # character offset in the full text
    end_char: int = -1      # filled in after all sections are detected
    page: int | None = None
    children: list[Section] = field(default_factory=list)
    heading_style: str = ""  # "allcaps", "numbered", "llm", "topics", etc.
    summary: str = ""       # LLM-generated summary of the section content

    @property
    def char_range(self) -> tuple[int, int]:
        return (self.start_char, self.end_char)


@dataclass
class DocumentStructure:
    """Detected structure of a document: sections, tree, and lookup methods."""
    title: str
    sections: list[Section]     # flat list, all sections
    root: Section               # tree root with children
    method: str = "regex"       # "regex", "llm", or "topics"

    def find_section_for_offset(self, char_offset: int) -> Section | None:
        """Find the deepest section containing this character offset."""
        best = None
        for s in self.sections:
            if s.start_char <= char_offset < s.end_char:
                if best is None or s.level > best.level:
                    best = s
        return best

    def find_section_for_chunk(self, chunk_text: str, full_text: str) -> Section | None:
        """Find the section a chunk belongs to by locating it in the full text."""
        search = chunk_text[:80]
        idx = full_text.find(search)
        if idx == -1:
            normalized_full = re.sub(r'\s+', ' ', full_text)
            normalized_search = re.sub(r'\s+', ' ', search)
            idx = normalized_full.find(normalized_search)
        if idx == -1:
            return None
        return self.find_section_for_offset(idx)


# -- Sentence (retrieval unit) -----------------------------------------------

@dataclass
class Sentence:
    """A sentence within a chunk -- the retrieval unit that gets embedded."""
    sentence_id: str            # {doc_id}_sent_{counter:06d}
    chunk_id: str               # parent chunk
    doc_id: str                 # source document
    text: str
    sent_index_in_chunk: int    # position within parent chunk (0, 1, 2...)


# -- Chunk (readable context unit) -------------------------------------------

@dataclass
class Chunk:
    """A passage-level chunk -- the readable context unit.

    Both flat chunking and topic chunking produce this type.
    Topic fields are None when flat chunking is used.
    Sentences are populated in-place by sentence segmentation.
    """
    chunk_id: str               # {doc_id}_chunk_{index:04d}
    doc_id: str
    text: str
    token_count: int            # len(text) // 4
    chunk_index: int            # global index across whole document
    page: int | None
    start_char: int
    end_char: int
    # Topic metadata -- None when flat chunking is used
    topic_title: str | None = None
    topic_summary: str | None = None
    topic_index: int | None = None
    chunk_index_in_topic: int | None = None
    # Populated in-place by sentence segmentation
    sentences: list[Sentence] = field(default_factory=list)


# -- Search Results -----------------------------------------------------------

@dataclass
class SearchHit:
    """A search result grouped by chunk.

    Scores are similarity scores: higher = more similar (1.0 = identical).
    Computed as 1 / (1 + L2_distance) so they are bounded in (0, 1].
    """
    chunk_id: str
    doc_id: str
    score: float                # best sentence similarity for this chunk (higher = better)
    snippets: list[str]         # up to 3 matching sentence texts, best first
    sentence_ids: list[str]     # corresponding sentence_ids
    # Optional: topic metadata carried through from chunk
    topic_title: str | None = None
    topic_summary: str | None = None


# -- Knowledge Graph ----------------------------------------------------------

@dataclass
class Triple:
    """A knowledge graph triple: (subject) --[relation]--> (object)."""
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    triple_id: str              # {chunk_id}_triple_{index:03d}
    doc_id: str
    chunk_id: str
    chunk_index: int | None = None    # kept for convenience, derived from chunk
    sentence_id: str | None = None    # future: sentence-level provenance


@dataclass
class Ontology:
    """Seed entity and relation types. Expandable as new types are discovered."""
    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)

    def expand(self, new_entity_types: set[str], new_relation_types: set[str]):
        """Add newly discovered types to the ontology."""
        for t in new_entity_types:
            if t and t not in self.entity_types:
                self.entity_types.append(t)
        for t in new_relation_types:
            if t and t not in self.relation_types:
                self.relation_types.append(t)


@dataclass
class ExtractionResult:
    """Result of KG extraction from a set of chunks."""
    triples: list[Triple]
    elapsed: float
    api_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    entity_types_discovered: set[str] = field(default_factory=set)
    relation_types_discovered: set[str] = field(default_factory=set)


@runtime_checkable
class KGExtractor(Protocol):
    """Protocol for KG extractors. Implement this to add new extraction backends."""
    name: str

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        ontology: Ontology,
        max_triples_per_chunk: int = 15,
    ) -> ExtractionResult: ...
