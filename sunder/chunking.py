"""
sunder.chunking -- Text chunking utilities.

Three modes:
  - chunk_text(): flat paragraph-boundary chunking
  - chunk_by_topics(): chunk within topic boundaries at paragraph boundaries
  - chunk_by_topics_sentence_aware(): chunk within topic boundaries at sentence boundaries

All return list[Chunk] with the unified schema.
"""

from __future__ import annotations

import re

from .constants import PAGE_BREAK
from .sentences import split_text_to_sentences
from .types import Chunk, DocumentStructure, Sentence, SunderConfig


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


def chunk_by_topics_sentence_aware(
    full_text: str,
    doc_id: str,
    structure: DocumentStructure,
    config: SunderConfig | None = None,
) -> list[Chunk]:
    """Chunk a document at sentence boundaries within topic spans.

    Unlike chunk_by_topics (which chunks at paragraph boundaries then segments
    sentences afterward), this function segments sentences *first* and uses
    sentences as the atomic packing unit. Chunk boundaries always fall on
    sentence boundaries.

    Overlap behavior: when a chunk is flushed, the last N sentences (fitting
    within chunk_overlap words) are carried to the START of the next chunk's
    sentence list. Carried sentences get new Sentence objects with IDs belonging
    to the new chunk — they are not duplicated in storage, but the agent sees
    overlap context when reading chunk text.

    Chunks come back with .sentences already populated — no need to call
    segment_sentences() afterward.

    Args:
        full_text: The full document text.
        doc_id: Document identifier (used for deterministic chunk IDs).
        structure: DocumentStructure from detect_structure().
        config: Pipeline config. Uses SunderConfig() defaults if None.

    Returns:
        List of Chunk objects with topic fields and sentences populated.
    """
    cfg = config or SunderConfig()
    sections = structure.sections
    if not sections:
        return chunk_text(full_text, doc_id, cfg)

    all_chunks: list[Chunk] = []
    global_chunk_idx = 0
    sent_counter = 0  # globally unique sentence counter for this doc

    for topic_idx, section in enumerate(sections):
        section_text = full_text[section.start_char:section.end_char]
        if not section_text.strip():
            continue

        # Segment the entire topic span into sentences
        sent_texts = split_text_to_sentences(
            section_text, atomic_line_length=cfg.atomic_line_length,
        )
        if not sent_texts:
            continue

        # Greedy sentence packing within this topic
        local_chunk_idx = 0
        current_sents: list[str] = []   # sentence texts in current chunk
        current_words = 0
        overlap_sents: list[str] = []   # sentences to carry into next chunk

        def _flush_chunk() -> None:
            nonlocal global_chunk_idx, local_chunk_idx, sent_counter
            nonlocal current_sents, current_words, overlap_sents

            chunk_str = " ".join(current_sents)
            word_count = _word_count(chunk_str)

            if word_count < cfg.min_chunk_size:
                # Too small — don't emit, but don't carry overlap either
                current_sents = []
                current_words = 0
                overlap_sents = []
                return

            chunk_id = f"{doc_id}_chunk_{global_chunk_idx:04d}"

            # Build Sentence objects for this chunk
            sentences: list[Sentence] = []
            for si, st in enumerate(current_sents):
                sentences.append(Sentence(
                    sentence_id=f"{doc_id}_sent_{sent_counter:06d}",
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=st,
                    sent_index_in_chunk=si,
                ))
                sent_counter += 1

            # Approximate start_char: find first sentence text in section
            first_sent = current_sents[0]
            rel_start = section_text.find(first_sent)
            if rel_start == -1:
                rel_start = 0
            chunk_start = section.start_char + rel_start

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_str,
                token_count=len(chunk_str) // 4,
                chunk_index=global_chunk_idx,
                page=_estimate_page(full_text, chunk_start),
                start_char=chunk_start,
                end_char=min(chunk_start + len(chunk_str), section.end_char),
                topic_title=section.title,
                topic_summary=section.summary or None,
                topic_index=topic_idx,
                chunk_index_in_topic=local_chunk_idx,
                sentences=sentences,
            ))
            global_chunk_idx += 1
            local_chunk_idx += 1

            # Compute overlap: walk backward through current_sents collecting
            # sentences that fit within chunk_overlap words
            overlap_sents = []
            overlap_words = 0
            for s in reversed(current_sents):
                sw = _word_count(s)
                if overlap_words + sw > cfg.chunk_overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_words += sw
            # Don't carry the entire chunk as overlap
            if len(overlap_sents) == len(current_sents):
                overlap_sents = overlap_sents[1:] if len(overlap_sents) > 1 else []

            current_sents = []
            current_words = 0

        # Seed with overlap from previous chunk (empty for first chunk in topic)
        for sent_text in sent_texts:
            sent_words = _word_count(sent_text)

            if current_words + sent_words > cfg.chunk_size and current_sents:
                _flush_chunk()
                # Carry overlap sentences into the new chunk
                if overlap_sents:
                    current_sents = list(overlap_sents)
                    current_words = sum(_word_count(s) for s in overlap_sents)
                    overlap_sents = []

            current_sents.append(sent_text)
            current_words += sent_words

        # Flush remaining sentences in this topic
        if current_sents:
            _flush_chunk()

    return all_chunks
