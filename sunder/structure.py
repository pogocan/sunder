"""
sunder.structure -- Document structure detection.

Three detection methods:
  - "topics" (default): Content-based topic detection. Works on any document.
  - "llm": Heading-based detection via LLM. Good for structured docs.
  - "regex": Pattern-based. Free and fast.

All methods return a DocumentStructure with sections, tree, and summaries.
"""

from __future__ import annotations

import os
import re

from .chunking import _chunk_paragraphs
from .constants import (
    ANTHROPIC_MODEL,
    LLM_STRUCTURE_TOOL,
    PAGE_BREAK,
    RE_ALLCAPS,
    RE_CHAPTER,
    RE_NUMBERED,
    RE_PART,
    SKIP_NUMBERED,
    SUMMARY_TOOL,
    TOPIC_TOOL,
)
from .types import DocumentStructure, Section


# -- Running header detection and stripping ------------------------------------

def detect_running_headers(text: str, min_frequency: float = 0.3) -> list[str]:
    """Detect phrases that appear on many pages (running headers/footers).

    Looks at the first and last line around each page break. If the same line
    appears on more than min_frequency of pages, it's a running header.
    """
    breaks = [m.end() for m in re.finditer(re.escape(PAGE_BREAK), text)]
    if not breaks:
        return []

    total_pages = len(breaks)

    first_lines: dict[str, int] = {}
    for bp in breaks:
        after = text[bp:bp + 300].lstrip()
        line = after.split('\n')[0].strip()
        line = _clean_title(line)
        if line and len(line) > 5:
            first_lines[line] = first_lines.get(line, 0) + 1

    break_starts = [m.start() for m in re.finditer(re.escape(PAGE_BREAK), text)]
    for bp in break_starts:
        before = text[max(0, bp - 300):bp].rstrip()
        line = before.split('\n')[-1].strip()
        line = _clean_title(line)
        if line and len(line) > 5:
            first_lines[line] = first_lines.get(line, 0) + 1

    threshold = total_pages * min_frequency
    return [line for line, count in first_lines.items() if count >= threshold]


def strip_running_headers(text: str, headers: list[str]) -> str:
    """Remove running headers from text, preserving page breaks and structure."""
    if not headers:
        return text

    for header in headers:
        pattern = re.compile(
            r'(?:^|\n)\s*' + re.escape(header) + r'\s*(?:\n|$)',
            re.MULTILINE,
        )
        text = pattern.sub('\n', text)

    return text


# -- Internal helpers ---------------------------------------------------------

def _clean_title(title: str) -> str:
    """Normalize tab-separated text into clean title."""
    title = re.sub(r'\t+', ' ', title)
    title = re.sub(r' {2,}', ' ', title)
    return title.strip()


def _estimate_page(text: str, char_offset: int) -> int:
    """Estimate page number from page break markers."""
    return text[:char_offset].count(PAGE_BREAK) + 1


# -- Regex heading detection --------------------------------------------------

def detect_headings_regex(text: str) -> list[Section]:
    """Detect section headings using regex patterns."""
    headings = []
    seen_positions = set()

    def _add(match, level, style):
        pos = match.start()
        if pos in seen_positions:
            return
        title = _clean_title(match.group(1))
        if not title or len(title) < 3:
            return
        if re.match(r'^[\d\.\s]+$', title):
            return
        seen_positions.add(pos)
        headings.append(Section(
            title=title, level=level, start_char=pos,
            page=_estimate_page(text, pos), heading_style=style,
        ))

    for m in RE_PART.finditer(text):
        _add(m, 1, "allcaps_part")

    for m in RE_CHAPTER.finditer(text):
        _add(m, 1, "chapter")

    for m in RE_NUMBERED.finditer(text):
        title = _clean_title(m.group(1))
        text_part = re.sub(r'^\d+\.\d+\.?\s*', '', title)
        if len(text_part) < 5:
            continue
        if any(text_part.startswith(w) for w in SKIP_NUMBERED):
            continue
        _add(m, 2, "numbered")

    for m in RE_ALLCAPS.finditer(text):
        title = _clean_title(m.group(1))
        if len(title) < 10:
            continue
        if any(abs(m.start() - s.start_char) < 5 for s in headings):
            continue
        digits = sum(1 for c in title if c.isdigit())
        if digits > len(title) * 0.3:
            continue
        _add(m, 2, "allcaps")

    headings.sort(key=lambda s: s.start_char)
    headings = _filter_toc_duplicates(headings)
    return headings


def _filter_toc_duplicates(headings: list[Section]) -> list[Section]:
    """If headings appear twice (TOC + body), keep only the body versions."""
    if len(headings) < 4:
        return headings

    by_title: dict[str, list[Section]] = {}
    for h in headings:
        key = re.sub(r'\s+', ' ', h.title.strip().upper())
        by_title.setdefault(key, []).append(h)

    duplicated = [t for t, entries in by_title.items() if len(entries) == 2]
    if len(duplicated) >= 3:
        toc_positions = set()
        for title in duplicated:
            entries = sorted(by_title[title], key=lambda s: s.start_char)
            toc_positions.add(entries[0].start_char)
        headings = [h for h in headings if h.start_char not in toc_positions]

    return headings


# -- LLM heading detection ----------------------------------------------------

def detect_headings_llm(text: str, running_headers: list[str] | None = None) -> list[Section]:
    """Use Claude to detect section headings from the document text."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    sample_text = text
    if running_headers:
        sample_text = strip_running_headers(text, running_headers)
    sample = _build_text_sample(sample_text)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=(
            "You are a document structure analyzer. Given a sample of a document's text, "
            "identify ALL section headings and their hierarchy levels.\n\n"
            "Rules:\n"
            "- Level 1: Top-level divisions (Part I, Chapter 3, etc.)\n"
            "- Level 2: Major sections (Item 1, Section 3.1, etc.)\n"
            "- Level 3: Subsections within sections\n"
            "- Clean up the title (remove tabs, extra spaces) but keep it recognizable\n"
            "- The heading_text_exact field must contain text that appears verbatim in the document "
            "so we can find its position. Include enough characters to be unique.\n"
            "- Do NOT include table of contents entries -- only the actual section headings in the body\n"
            "- Do NOT include figure labels, table headers, or exhibit numbers\n"
            "- Do NOT include page numbers or footers\n"
            "- Include ALL sections you can find, not just the first few"
        ),
        tools=[LLM_STRUCTURE_TOOL],
        tool_choice={"type": "tool", "name": "report_document_structure"},
        messages=[{
            "role": "user",
            "content": f"Identify all section headings in this document:\n\n{sample}",
        }],
    )

    raw_headings = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_document_structure":
            for s in block.input.get("sections", []):
                title = _clean_title(s["title"])
                level = s.get("level", 2)
                exact = s.get("heading_text_exact", title)
                raw_headings.append((title, level, exact))

    headings = []
    for title, level, exact in raw_headings:
        pos = _find_heading_position(text, exact, title)
        if pos == -1:
            continue
        headings.append(Section(
            title=title, level=level, start_char=pos,
            page=_estimate_page(text, pos), heading_style="llm",
        ))

    headings = _resolve_toc_to_body(headings, text)
    headings = _validate_heading_order(headings, text)

    headings.sort(key=lambda s: s.start_char)
    deduped = []
    for h in headings:
        if not deduped or h.start_char - deduped[-1].start_char > 50:
            deduped.append(h)

    return deduped


def _validate_heading_order(headings: list[Section], text: str) -> list[Section]:
    """Ensure heading positions follow the order the LLM reported them."""
    if len(headings) < 2:
        return headings

    validated = [headings[0]]

    for h in headings[1:]:
        prev_pos = validated[-1].start_char

        if h.start_char > prev_pos:
            validated.append(h)
        else:
            new_pos = _find_heading_after(text, h.title, prev_pos)
            if new_pos >= 0:
                h.start_char = new_pos
                h.page = _estimate_page(text, new_pos)
                validated.append(h)

    return validated


def _find_heading_after(text: str, title: str, after_pos: int) -> int:
    """Find a heading title in text after a given position."""
    search_text = text[after_pos:]

    pos = search_text.find(title)
    if pos >= 0:
        return after_pos + pos

    pattern = re.escape(title).replace(r'\ ', r'\s+')
    m = re.search(pattern, search_text, re.IGNORECASE)
    if m:
        return after_pos + m.start()

    core = re.sub(r'^(Chapter|Part|Section|Item)\s+[\dIVX]+[:.]\s*', '', title, flags=re.IGNORECASE).strip()
    if core and len(core) > 10:
        pos = search_text.find(core)
        if pos >= 0:
            return after_pos + pos

    return -1


def _resolve_toc_to_body(headings: list[Section], text: str) -> list[Section]:
    """For headings found in the TOC area, find their actual body position."""
    if not headings:
        return headings

    toc_boundary = int(len(text) * 0.10)
    resolved = []

    for h in headings:
        if h.start_char >= toc_boundary:
            resolved.append(h)
            continue

        body_pos = _find_heading_in_body(text, h.title, toc_boundary)
        if body_pos >= 0:
            resolved.append(Section(
                title=h.title,
                level=h.level,
                start_char=body_pos,
                page=_estimate_page(text, body_pos),
                heading_style=h.heading_style,
            ))
        else:
            resolved.append(h)

    return resolved


def _find_heading_in_body(text: str, title: str, search_from: int) -> int:
    """Search for a heading title in the body text (after the TOC area)."""
    body = text[search_from:]

    pos = body.find(title)
    if pos >= 0:
        return search_from + pos

    pattern = re.escape(title).replace(r'\ ', r'\s+')
    m = re.search(pattern, body, re.IGNORECASE)
    if m:
        return search_from + m.start()

    core = re.sub(r'^(Chapter|Part|Section|Item)\s+[\dIVX]+[:.]\s*', '', title, flags=re.IGNORECASE).strip()
    if core and len(core) > 10:
        occurrences = list(re.finditer(re.escape(core), body))
        total_pages = body.count(PAGE_BREAK) + 1
        if len(occurrences) > total_pages * 0.3:
            return -1

        for occ in occurrences:
            pos = search_from + occ.start()
            before = text[max(0, pos - 60):pos]
            if PAGE_BREAK in before:
                continue
            return pos

    return -1


def _build_text_sample(text: str, max_chars: int = 30000) -> str:
    """Build a representative sample of the document for LLM analysis."""
    if len(text) <= max_chars:
        return text

    parts = []
    chars_used = 0

    parts.append(text[:4000])
    parts.append("\n\n[...]\n\n")
    chars_used += 4006

    break_positions = [m.start() for m in re.finditer(re.escape(PAGE_BREAK), text)]

    if break_positions:
        available = max_chars - 6000
        snippet_size = 400

        max_breaks = available // (snippet_size + 6)
        if len(break_positions) > max_breaks:
            step = len(break_positions) / max_breaks
            selected = [break_positions[int(i * step)] for i in range(max_breaks)]
        else:
            selected = break_positions

        for pos in selected:
            if chars_used >= max_chars - 2000:
                break
            end = min(len(text), pos + snippet_size)
            snippet = text[pos:end]
            parts.append(snippet)
            parts.append("\n[...]\n")
            chars_used += len(snippet) + 6
    else:
        num_samples = max(15, (max_chars - 6000) // 1500)
        step = len(text) // num_samples
        for i in range(num_samples):
            pos = i * step
            snippet = text[pos:pos + 1500]
            parts.append(snippet)
            parts.append("\n[...]\n")
            chars_used += len(snippet) + 6

    parts.append(text[-2000:])

    return "".join(parts)


def _find_heading_position(text: str, exact: str, title: str) -> int:
    """Find the position of a heading in the text using multiple strategies."""
    pos = text.find(exact)
    if pos >= 0:
        return pos

    cleaned_exact = re.sub(r'\s+', ' ', exact.strip())
    cleaned_text = re.sub(r'\s+', ' ', text)
    pos = cleaned_text.find(cleaned_exact)
    if pos >= 0:
        return _map_cleaned_pos_to_original(text, cleaned_text, pos)

    pattern = re.escape(title).replace(r'\ ', r'\s+')
    m = re.search(pattern, text)
    if m:
        return m.start()

    return -1


def _map_cleaned_pos_to_original(original: str, cleaned: str, cleaned_pos: int) -> int:
    """Approximate mapping from cleaned text position back to original."""
    char_count = 0
    for i, c in enumerate(cleaned[:cleaned_pos]):
        if not c.isspace():
            char_count += 1

    count = 0
    for i, c in enumerate(original):
        if not c.isspace():
            count += 1
            if count >= char_count:
                return i
    return -1


# -- Tree building ------------------------------------------------------------

def build_tree(headings: list[Section], text: str, doc_title: str = "") -> DocumentStructure:
    """Build a section tree from detected headings and fill in end offsets."""
    root = Section(title=doc_title or "Document", level=0, start_char=0)
    root.end_char = len(text)

    if not headings:
        return DocumentStructure(title=root.title, sections=[], root=root)

    for i, h in enumerate(headings):
        if i + 1 < len(headings):
            h.end_char = headings[i + 1].start_char
        else:
            h.end_char = len(text)

    stack = [root]
    for h in headings:
        while len(stack) > 1 and stack[-1].level >= h.level:
            stack.pop()
        stack[-1].children.append(h)
        stack.append(h)

    return DocumentStructure(title=root.title, sections=headings, root=root)


# -- Section summarization ----------------------------------------------------

def summarize_sections(structure: DocumentStructure, full_text: str) -> DocumentStructure:
    """Add LLM-generated summaries to each section."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    sections = structure.sections
    if not sections:
        return structure

    section_samples = []
    for s in sections:
        content = full_text[s.start_char:s.end_char]
        word_count = len(content.split())
        sample = _sample_section_content(content)
        section_samples.append({
            "title": s.title,
            "page": s.page,
            "word_count": word_count,
            "sample": sample,
        })

    batch_size = 15
    all_summaries: dict[str, str] = {}

    for i in range(0, len(section_samples), batch_size):
        batch = section_samples[i:i + batch_size]

        prompt_parts = []
        for j, ss in enumerate(batch):
            prompt_parts.append(
                f"### Section {j+1}: \"{ss['title']}\" (page {ss['page']}, ~{ss['word_count']} words)\n"
                f"{ss['sample']}\n"
            )

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            temperature=0.0,
            system=(
                "You are a document summarizer. For each section, write a concise 1-3 sentence "
                "summary based on the provided sample (beginning and end of the section). "
                "Focus on the main topic and key points. Be specific, not generic."
            ),
            tools=[SUMMARY_TOOL],
            tool_choice={"type": "tool", "name": "summarize_sections"},
            messages=[{
                "role": "user",
                "content": "Summarize each of these document sections:\n\n" + "\n".join(prompt_parts),
            }],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "summarize_sections":
                for item in block.input.get("summaries", []):
                    all_summaries[item["title"]] = item["summary"]

    for s in sections:
        if s.title in all_summaries:
            s.summary = all_summaries[s.title]

    return structure


def _sample_section_content(content: str, head_chars: int = 600, tail_chars: int = 600) -> str:
    """Get the beginning and end of a section for summarization."""
    content = content.strip()
    if len(content) <= head_chars + tail_chars + 50:
        return content

    head = content[:head_chars]
    tail = content[-tail_chars:]
    return f"{head}\n\n[... middle of section omitted ...]\n\n{tail}"


# -- Topic-based detection ----------------------------------------------------

def detect_headings_topics(text: str, chunk_size: int = 200, chunk_overlap: int = 30) -> list[Section]:
    """Detect document structure by analyzing content topics in chunks.

    Chunks the text, sends batches to the LLM to identify topic shifts,
    then converts topic segments into Section objects.
    Works on any document regardless of heading formatting.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    topic_chunk_size = max(chunk_size, 400)
    topic_overlap = 50
    chunks_with_pos = _chunk_paragraphs(text, topic_chunk_size, topic_overlap)
    chunks = [c for c, _ in chunks_with_pos]
    chunk_char_positions = [pos for _, pos in chunks_with_pos]
    chunk_char_positions.append(len(text))  # end sentinel

    window_size = 20
    window_step = 18
    all_topics: list[dict] = []

    for win_start in range(0, len(chunks), window_step):
        win_end = min(win_start + window_size, len(chunks))
        window = chunks[win_start:win_end]

        if not window:
            break

        chunk_text_parts = []
        for i, c in enumerate(window):
            idx = win_start + i
            words = c.split()
            preview = " ".join(words[:100])
            chunk_text_parts.append(f"[Chunk {idx}] {preview}")

        context = ""
        if all_topics:
            last = all_topics[-1]
            context = f"\nPrevious topic: \"{last['title']}\" (ending at chunk {last['end_chunk']}). Continue it if the topic hasn't changed."

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            temperature=0.0,
            system=(
                "You are a document structure analyzer. Given numbered text chunks, "
                "identify the MAJOR topic segments. Group generously -- a book chapter "
                "or report section is one topic, not each paragraph.\n\n"
                "Rules:\n"
                "- Title: short descriptive name (3-8 words)\n"
                "- Summary: 1-3 sentences on what this section covers\n"
                "- Every chunk belongs to exactly one topic\n"
                "- Aim for FEWER, LARGER topics. Prefer 2-4 topics per window, not 5-10.\n"
                "- If content continues the same subject as the previous topic, reuse that exact title.\n"
                "- Only create a new topic when the subject clearly changes."
            ),
            tools=[TOPIC_TOOL],
            tool_choice={"type": "tool", "name": "report_topics"},
            messages=[{
                "role": "user",
                "content": (
                    f"Identify major topic segments in chunks "
                    f"{win_start}-{win_end - 1} (of {len(chunks)} total):"
                    f"{context}\n\n" + "\n\n".join(chunk_text_parts)
                ),
            }],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "report_topics":
                for t in block.input.get("topics", []):
                    all_topics.append(t)

        if win_end >= len(chunks):
            break

    merged = _merge_topic_windows(all_topics)

    sections = []
    for t in merged:
        start_idx = t["start_chunk"]
        end_idx = t["end_chunk"]
        start_char = chunk_char_positions[start_idx] if start_idx < len(chunk_char_positions) else 0
        end_char = chunk_char_positions[end_idx + 1] if end_idx + 1 < len(chunk_char_positions) else len(text)

        sections.append(Section(
            title=t["title"],
            level=1,
            start_char=start_char,
            end_char=end_char,
            page=_estimate_page(text, start_char),
            heading_style="topics",
            summary=t.get("summary", ""),
        ))

    return sections


def _merge_topic_windows(topics: list[dict]) -> list[dict]:
    """Merge topics from overlapping windows. Same/similar title = continuation."""
    if not topics:
        return []

    merged = [topics[0]]

    for t in topics[1:]:
        prev = merged[-1]
        if (t["title"].lower() == prev["title"].lower()
                or t["start_chunk"] <= prev["end_chunk"]):
            prev["end_chunk"] = max(prev["end_chunk"], t["end_chunk"])
            if len(t.get("summary", "")) > len(prev.get("summary", "")):
                prev["summary"] = t["summary"]
        else:
            if t["start_chunk"] > prev["end_chunk"] + 1:
                prev["end_chunk"] = t["start_chunk"] - 1
            merged.append(t)

    return merged



# -- Chunk-to-section mapping -------------------------------------------------

def map_chunks_to_sections(
    chunks: list[str],
    full_text: str,
    structure: DocumentStructure,
) -> list[dict]:
    """Map each chunk to its section.

    Returns list of {chunk_index, section_title, section_level, page}.
    """
    results = []
    for i, chunk in enumerate(chunks):
        section = structure.find_section_for_chunk(chunk, full_text)
        results.append({
            "chunk_index": i,
            "section_title": section.title if section else "(unknown)",
            "section_level": section.level if section else -1,
            "page": section.page if section else None,
        })
    return results


# -- High-level API -----------------------------------------------------------

def detect_structure(text: str, method: str = "topics", doc_title: str = "") -> DocumentStructure:
    """Detect document structure using the specified method.

    Automatically detects and strips running headers before heading detection,
    but all offsets are against the original text.

    Args:
        text: Full document text.
        method: "topics" (content-based, default), "llm" (heading detection),
                or "regex" (pattern-based, free).
        doc_title: Optional document title for the root node.

    Returns:
        DocumentStructure with sections and tree.
    """
    running_headers = detect_running_headers(text)

    if method == "topics":
        headings = detect_headings_topics(text)
        structure = build_tree(headings, text, doc_title=doc_title)
        structure.method = method
        return structure
    elif method == "llm":
        headings = detect_headings_llm(text, running_headers=running_headers)
    else:
        headings = detect_headings_regex(text)

    structure = build_tree(headings, text, doc_title=doc_title)
    structure.method = method
    return structure
