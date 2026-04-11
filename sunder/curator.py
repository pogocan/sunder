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
from pathlib import Path

from .types import SunderConfig

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


def _split_into_sections(text: str, max_chars: int = 2500) -> list[str]:
    """Split text into sections of roughly max_chars.

    Splits on double newlines first, then single newlines if a single
    paragraph is itself too large. Sized in characters (not words) so the
    prompt stays within the model's structured-JSON response budget.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    sections: list[str] = []
    current: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if current:
            sections.append('\n\n'.join(current))
            current = []
            current_chars = 0

    for para in paragraphs:
        if len(para) > max_chars:
            # Paragraph is too large on its own -- split on single newlines.
            lines = [l.strip() for l in para.split('\n') if l.strip()]
            for line in lines:
                if current_chars + len(line) > max_chars and current:
                    flush()
                current.append(line)
                current_chars += len(line)
        else:
            if current_chars + len(para) > max_chars and current:
                flush()
            current.append(para)
            current_chars += len(para)

    flush()
    return sections


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
            result = json.loads(response.content[0].text)
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
