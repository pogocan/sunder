"""
sunder.curator -- LLM-assisted chunk curation.

Two-step process:
  1. curate_text()  -- sends sections to Claude, returns a curation report
  2. apply_curation() -- reads approved report, writes curated chunks

This separation ensures humans can review all decisions before
anything is written to disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .types import SunderConfig


def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers from an LLM response.

    Claude sometimes wraps JSON in a markdown code fence despite being told
    not to. Accept both fenced and unfenced responses.
    """
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s

SYSTEM_PROMPT = """You are a document curator preparing text for a RAG system.

You will receive a section of extracted text and a set of project goals.
Evaluate the section and return a JSON object in this exact format:

{
  "decision": "KEEP" | "SPLIT" | "REWRITE" | "DISCARD",
  "reason": "brief reason",
  "chunks": ["chunk text 1", "chunk text 2"]
}

Decision rules:
- KEEP: content is useful and well-formatted as-is. Return original text as single item in chunks.
- SPLIT: content covers multiple distinct topics that should be retrieved independently. Split into focused self-contained chunks. Return each as a separate item in chunks.
- REWRITE: content is useful but has formatting artifacts, noise, or unclear structure from PDF extraction. Rewrite cleanly while preserving all technical content exactly. Return rewritten text as single item in chunks.
- DISCARD: content is not relevant to the project goals. Return empty chunks array.

Important:
- Never alter technical content, syntax definitions, code examples, or parameter descriptions
- Never split a syntax definition from its parameters and examples -- they belong together as one chunk
- When in doubt between KEEP and DISCARD, always KEEP
- Preserve all code formatting exactly
- Return ONLY the JSON object, no preamble, no markdown fences"""

USER_TEMPLATE = """Project goals:
{goals}

Evaluate this section:

---
{section}
---"""


def _format_goals(goals: dict) -> str:
    """Format goals dict into a clear prompt string."""
    lines = []
    if "purpose" in goals:
        lines.append(f"Purpose: {goals['purpose']}\n")
    if "keep" in goals:
        lines.append("Keep:")
        for item in goals["keep"]:
            lines.append(f"  - {item}")
        lines.append("")
    if "discard" in goals:
        lines.append("Discard:")
        for item in goals["discard"]:
            lines.append(f"  - {item}")
        lines.append("")
    if "rewrite_instructions" in goals:
        lines.append("Rewrite instructions:")
        for item in goals["rewrite_instructions"]:
            lines.append(f"  - {item}")
    return "\n".join(lines)


# Patterns that indicate a new top-level topic boundary
# Matches lines that are standalone uppercase words/phrases
# e.g. "CHAR", "DATE", "DEFINE RECORD", "ALTER LOG", "COLLECT"
_HEADING_PATTERN = re.compile(
    r'^([A-Z][A-Z0-9 ]{1,40})$',
    re.MULTILINE
)


def _is_heading(line: str) -> bool:
    """Return True if line looks like a standalone statement/function heading."""
    line = line.strip()
    if not line:
        return False
    return bool(_HEADING_PATTERN.match(line))


def _split_on_headings(text: str) -> list[str]:
    """
    Split text on standalone uppercase heading lines.
    Returns list of sections, each starting with its heading.
    """
    lines = text.split('\n')
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _is_heading(line) and current:
            # Flush current section when we hit a new heading
            section = '\n'.join(current).strip()
            if section:
                sections.append(section)
            current = [line]
        else:
            current.append(line)

    if current:
        section = '\n'.join(current).strip()
        if section:
            sections.append(section)

    return sections


def _split_into_sections(text: str, max_chars: int = 2500) -> list[str]:
    """
    Split text into sections for curation.

    Strategy:
    1. First split on standalone uppercase headings (CHAR, DATE, DEFINE RECORD etc.)
       This keeps each function/statement definition intact.
    2. If any resulting section is still larger than max_chars * 2,
       further split on double newlines.
    3. Last resort: split on single newlines.

    Never produces a section larger than max_chars * 2.
    """
    # Step 1: split on headings
    heading_sections = _split_on_headings(text)

    # If heading detection found nothing useful, fall back to original behavior
    if len(heading_sections) <= 1:
        heading_sections = [text]

    # Step 2 & 3: further split oversized sections
    final_sections: list[str] = []

    for section in heading_sections:
        if len(section) <= max_chars * 2:
            final_sections.append(section)
            continue

        # Too large — split on double newlines
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        current: list[str] = []
        current_chars = 0

        for para in paragraphs:
            if len(para) > max_chars:
                # Single paragraph too large — split on single newlines
                if current:
                    final_sections.append('\n\n'.join(current))
                    current = []
                    current_chars = 0
                lines = [l.strip() for l in para.split('\n') if l.strip()]
                line_current: list[str] = []
                line_chars = 0
                for line in lines:
                    if line_chars + len(line) > max_chars and line_current:
                        final_sections.append('\n'.join(line_current))
                        line_current = []
                        line_chars = 0
                    line_current.append(line)
                    line_chars += len(line)
                if line_current:
                    final_sections.append('\n'.join(line_current))
            else:
                if current_chars + len(para) > max_chars and current:
                    final_sections.append('\n\n'.join(current))
                    current = []
                    current_chars = 0
                current.append(para)
                current_chars += len(para)

        if current:
            final_sections.append('\n\n'.join(current))

    return [s for s in final_sections if s.strip()]


def curate_text(
    text: str,
    doc_id: str,
    config: SunderConfig,
) -> list[dict]:
    """Send extracted text to Claude and return a curation report.

    Does NOT write anything to disk. Returns a list of decision records
    that can be reviewed before applying.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=config.anthropic_api_key)
    goals_str = _format_goals(config.curation_goals or {})
    sections = _split_into_sections(text)
    report: list[dict] = []

    print(f"  Curating {len(sections)} sections for {doc_id}...")

    for i, section in enumerate(sections):
        if not section.strip():
            continue

        print(f"  Section {i + 1}/{len(sections)}...", end="\r")

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_TEMPLATE.format(
                    goals=goals_str,
                    section=section,
                ),
            }],
        )

        try:
            result = json.loads(_strip_code_fences(response.content[0].text))
            decision = result.get("decision", "KEEP")
            chunks = result.get("chunks", [])
            reason = result.get("reason", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            decision = "KEEP"
            chunks = [section]
            reason = "WARNING: malformed Claude response, kept original"

        report.append({
            "section_id": i,
            "doc_id": doc_id,
            "decision": decision,
            "reason": reason,
            "original": section,
            "chunks": [c.strip() for c in chunks if c.strip()],
        })

    kept = sum(1 for r in report if r["decision"] != "DISCARD")
    discarded = sum(1 for r in report if r["decision"] == "DISCARD")
    print(f"\n  Done. {len(sections)} sections -> {kept} kept, {discarded} discarded")
    return report


def apply_curation(report: list[dict]) -> list[str]:
    """Extract curated chunks from an approved curation report."""
    chunks: list[str] = []
    for record in report:
        if record["decision"] == "DISCARD":
            continue
        for chunk in record.get("chunks", []):
            if chunk.strip():
                chunks.append(chunk.strip())
    return chunks


def save_report(report: list[dict], path: str | Path) -> None:
    """Save curation report to JSON for human review."""
    Path(path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(f"Curation report saved to {path}")


def load_report(path: str | Path) -> list[dict]:
    """Load a previously saved curation report."""
    return json.loads(Path(path).read_text())
