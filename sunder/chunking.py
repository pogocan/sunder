"""
sunder.chunking -- Text chunking utilities.

Two modes:
  - chunk_text(): flat paragraph-boundary chunking
  - chunk_by_topics(): chunk within topic boundaries so chunks never cross topics

Both return list[Chunk] with the unified schema.
"""

from __future__ import annotations

import re

from .constants import PAGE_BREAK
from .types import Chunk, DocumentStructure, SunderConfig


def _split_paragraphs(text: str) -> list[tuple[str, int]]:
    """Split text into paragraphs, returning (paragraph_text, start_char) pairs.

    Splits on double-newline boundaries. Single newlines within a paragraph
    are preserved (they may be meaningful formatting).
    """
    paragraphs = []
    for m in re.finditer(r'[^\n](?:[^\n]|\n(?!\n))*', text):
        para = m.group().strip()
        if para:
            paragraphs.append((para, m.start()))
    return paragraphs


def _word_count(text: str) -> int:
    """Estimate word count."""
    return len(text.split())


def _estimate_page(text: str, char_offset: int) -> int:
    """Estimate page number from page break markers."""
    return text[:char_offset].count(PAGE_BREAK) + 1


def _chunk_paragraphs(
    text: str, chunk_size: int, overlap: int,
) -> list[tuple[str, int]]:
    """Core paragraph-boundary chunking, returning (chunk_text, start_char) pairs.

    Algorithm:
      1. Split text into paragraphs on double-newline boundaries
      2. Greedily pack paragraphs until adding the next would exceed chunk_size
      3. Oversized paragraphs (> chunk_size words) become their own chunk
      4. Overlap is achieved by carrying the last paragraph of the previous chunk
         into the next chunk (if it fits within overlap budget)
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[tuple[str, int]] = []
    current_paras: list[tuple[str, int]] = []  # (text, start_char)
    current_words = 0

    for para_text, para_start in paragraphs:
        para_words = _word_count(para_text)

        if current_words + para_words > chunk_size and current_paras:
            # Flush current chunk
            chunk_str = "\n\n".join(p for p, _ in current_paras)
            chunk_start = current_paras[0][1]
            chunks.append((chunk_str, chunk_start))

            # Overlap: carry last paragraph forward if it's within overlap budget
            last_para, last_start = current_paras[-1]
            last_words = _word_count(last_para)
            if last_words <= overlap and len(current_paras) > 1:
                current_paras = [(last_para, last_start)]
                current_words = last_words
            else:
                current_paras = []
                current_words = 0

        current_paras.append((para_text, para_start))
        current_words += para_words

    # Flush remaining
    if current_paras:
        chunk_str = "\n\n".join(p for p, _ in current_paras)
        chunk_start = current_paras[0][1]
        chunks.append((chunk_str, chunk_start))

    return chunks


def chunk_text(
    text: str, doc_id: str, config: SunderConfig | None = None,
) -> list[Chunk]:
    """Flat paragraph-boundary chunking. Topic fields are None.

    Args:
        text: The full document text.
        doc_id: Document identifier (used for deterministic chunk IDs).
        config: Pipeline config. Uses SunderConfig() defaults if None.

    Returns:
        List of Chunk objects with topic fields set to None.
    """
    cfg = config or SunderConfig()
    raw_chunks = _chunk_paragraphs(text, cfg.chunk_size, cfg.chunk_overlap)
    chunks = []
    for i, (chunk_str, start_char) in enumerate(raw_chunks):
        word_count = _word_count(chunk_str)
        if word_count < cfg.min_chunk_size:
            continue
        chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=chunk_str,
            token_count=len(chunk_str) // 4,
            chunk_index=len(chunks),
            page=_estimate_page(text, start_char),
            start_char=start_char,
            end_char=start_char + len(chunk_str),
        ))
    return chunks


def chunk_by_topics(
    full_text: str,
    doc_id: str,
    structure: DocumentStructure,
    config: SunderConfig | None = None,
) -> list[Chunk]:
    """Chunk a document using topic boundaries.

    Instead of blindly slicing every N words, this chunks *within* each topic
    section so chunks never cross topic boundaries. Each chunk carries its
    parent topic's title, summary, and page.

    Args:
        full_text: The full document text.
        doc_id: Document identifier (used for deterministic chunk IDs).
        structure: DocumentStructure from detect_structure().
        config: Pipeline config. Uses SunderConfig() defaults if None.

    Returns:
        List of Chunk objects with topic fields populated.
    """
    cfg = config or SunderConfig()
    sections = structure.sections
    if not sections:
        # No sections detected -- fall back to flat chunking
        return chunk_text(full_text, doc_id, cfg)

    all_chunks: list[Chunk] = []
    global_idx = 0

    for topic_idx, section in enumerate(sections):
        section_text = full_text[section.start_char:section.end_char]
        if not section_text.strip():
            continue

        # Chunk within this section
        section_chunks = _chunk_paragraphs(section_text, cfg.chunk_size, cfg.chunk_overlap)

        for local_idx, (chunk_str, rel_pos) in enumerate(section_chunks):
            word_count = _word_count(chunk_str)
            if word_count < cfg.min_chunk_size:
                continue
            chunk_start = section.start_char + rel_pos
            chunk_id = f"{doc_id}_chunk_{global_idx:04d}"

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_str,
                token_count=len(chunk_str) // 4,
                chunk_index=global_idx,
                page=_estimate_page(full_text, chunk_start),
                start_char=chunk_start,
                end_char=min(chunk_start + len(chunk_str), section.end_char),
                topic_title=section.title,
                topic_summary=section.summary or None,
                topic_index=topic_idx,
                chunk_index_in_topic=local_idx,
            ))
            global_idx += 1

    return all_chunks
