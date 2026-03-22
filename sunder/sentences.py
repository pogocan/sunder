"""
sunder.sentences -- Sentence segmentation for chunks.

Splits chunk text into individual sentences for fine-grained retrieval.
Each sentence becomes a Sentence object attached to its parent Chunk.
"""

from __future__ import annotations

import re

from .types import Chunk, Sentence, SunderConfig

# Abbreviations that should not trigger sentence splits.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "e.g", "i.e", "etc", "vs", "fig", "no",
    "vol", "dept", "approx", "inc", "ltd", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

# Regex that matches sentence-ending punctuation followed by whitespace.
# Abbreviation protection is handled in code for clarity.
_RE_SENT_END = re.compile(r'([.!?])\s+')


def _split_line_into_sentences(line: str) -> list[str]:
    """Split a long line on sentence-ending punctuation, protecting abbreviations."""
    sentences: list[str] = []
    current_pos = 0

    for m in _RE_SENT_END.finditer(line):
        end_pos = m.end()
        candidate = line[current_pos:m.start() + 1].strip()

        if not candidate:
            continue

        # Check if this period follows a known abbreviation
        # Look at the last word before the punctuation
        if m.group(1) == '.':
            before_dot = line[current_pos:m.start()].rstrip()
            last_word = before_dot.split()[-1].lower().rstrip('.') if before_dot.split() else ""
            # Check both "e.g" and "eg" forms
            bare_word = last_word.replace('.', '')
            if last_word in _ABBREVIATIONS or bare_word in _ABBREVIATIONS:
                continue

        sentences.append(candidate)
        current_pos = end_pos

    # Remainder after last split
    remainder = line[current_pos:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences if sentences else [line.strip()]


def segment_sentences(
    chunks: list[Chunk], config: SunderConfig | None = None,
) -> list[Chunk]:
    """Populate chunk.sentences in-place for each chunk.

    Segmentation heuristic (proven on technical PDFs):
      1. Split chunk text on newlines
      2. Lines under atomic_line_length chars -> atomic sentence
      3. Longer lines -> split on sentence-ending punctuation (. ! ?)
      4. Protect common abbreviations from false splits
      5. Strip empty sentences

    Sentence IDs are globally unique across the whole batch:
      {doc_id}_sent_{counter:06d}
    Counter resets per doc_id, not per chunk.

    Args:
        chunks: List of Chunk objects to segment.
        config: Pipeline config. Uses SunderConfig() defaults if None.

    Returns the same list[Chunk] -- mutates and passes through.
    """
    cfg = config or SunderConfig()
    atomic_max = cfg.atomic_line_length

    # Track counter per doc_id for globally unique sentence IDs
    doc_counters: dict[str, int] = {}

    for chunk in chunks:
        doc_id = chunk.doc_id

        if doc_id not in doc_counters:
            doc_counters[doc_id] = 0

        chunk.sentences = []

        if not chunk.text or not chunk.text.strip():
            continue

        # Split on newlines first
        lines = chunk.text.split('\n')
        sent_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if len(line) <= atomic_max:
                # Short line -> atomic sentence
                sentence = Sentence(
                    sentence_id=f"{doc_id}_sent_{doc_counters[doc_id]:06d}",
                    chunk_id=chunk.chunk_id,
                    doc_id=doc_id,
                    text=line,
                    sent_index_in_chunk=sent_index,
                )
                chunk.sentences.append(sentence)
                doc_counters[doc_id] += 1
                sent_index += 1
            else:
                # Long line -> split on sentence boundaries
                parts = _split_line_into_sentences(line)
                for part in parts:
                    if not part:
                        continue
                    sentence = Sentence(
                        sentence_id=f"{doc_id}_sent_{doc_counters[doc_id]:06d}",
                        chunk_id=chunk.chunk_id,
                        doc_id=doc_id,
                        text=part,
                        sent_index_in_chunk=sent_index,
                    )
                    chunk.sentences.append(sentence)
                    doc_counters[doc_id] += 1
                    sent_index += 1

    return chunks
