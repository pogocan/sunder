"""
sunder.constants -- Default values, model names, regex patterns, tool schemas.
"""

import re

# -- Model defaults -----------------------------------------------------------

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OLLAMA_URL = "http://localhost:11434"

# -- Document markers ---------------------------------------------------------

PAGE_BREAK = "--- Page Break ---"

# -- Chunking defaults --------------------------------------------------------

CHUNK_SIZE = 200       # words per chunk
CHUNK_OVERLAP = 30     # word overlap between chunks

# -- Ontology seeds -----------------------------------------------------------
# General-purpose starting types. Extractors expand beyond these as needed.

ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "SYSTEM",
    "PROCESS", "CONCEPT", "TECHNOLOGY", "DOCUMENT", "EVENT",
]

RELATION_TYPES = [
    "IS_PART_OF", "DEPENDS_ON", "PRODUCES", "USES", "CONTAINS",
    "CONNECTS_TO", "PRECEDES", "FOLLOWS", "REFERENCES", "REQUIRES",
    "IMPLEMENTS", "DEFINES", "CONFIGURES", "MONITORS", "MANAGES",
]

# -- Regex patterns for heading detection -------------------------------------

# "PART I" / "PART II" / "PART IV."
RE_PART = re.compile(r'^(PART\s+[IVX]+\.?)$', re.MULTILINE)

# "Chapter 3: Introduction" / "CHAPTER III"
RE_CHAPTER = re.compile(r'^(Chapter\s+[\dIVX]+[\.:]?\s*.*)', re.MULTILINE | re.IGNORECASE)

# "3.1 Section title" / "2.6 Exercises"
RE_NUMBERED = re.compile(r'^(\d+\.\d+\.?\s+[A-Z][a-zA-Z\s\-\':,\(\)]+)', re.MULTILINE)

# ALL CAPS lines: "INTRODUCTION", "METHODOLOGY", "CONCLUSIONS"
RE_ALLCAPS = re.compile(r'^([A-Z][A-Z\s\d\.\,\-\[\]\(\)\/\'&:]+)$', re.MULTILINE)

# Strings that indicate a numbered line is NOT a section heading
# (common boilerplate found across technical and business documents)
SKIP_NUMBERED = (
    "Table of Contents", "Index", "Appendix", "Glossary",
    "References", "Bibliography", "Footnote", "Figure", "Table",
    "List of", "See also", "Note", "Notes",
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December",
    "The ",
)

# -- Tool schemas (for Anthropic tool_use) ------------------------------------

LLM_STRUCTURE_TOOL = {
    "name": "report_document_structure",
    "description": "Report the detected document structure as a list of section headings",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Section heading text, cleaned up"},
                        "level": {"type": "integer", "description": "1=top-level (Part/Chapter), 2=section (Item/numbered), 3=subsection"},
                        "heading_text_exact": {"type": "string", "description": "The exact text as it appears in the document (for matching)"},
                    },
                    "required": ["title", "level", "heading_text_exact"],
                },
            },
        },
        "required": ["sections"],
    },
}

SUMMARY_TOOL = {
    "name": "summarize_sections",
    "description": "Provide a 1-3 sentence summary for each document section",
    "input_schema": {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string", "description": "1-3 sentence summary of what this section covers"},
                    },
                    "required": ["title", "summary"],
                },
            },
        },
        "required": ["summaries"],
    },
}

TOPIC_TOOL = {
    "name": "report_topics",
    "description": "Report the topic segments found in the document chunks",
    "input_schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short descriptive title for this topic (3-8 words)"},
                        "summary": {"type": "string", "description": "1-3 sentence summary of what this section covers"},
                        "start_chunk": {"type": "integer", "description": "First chunk index where this topic starts"},
                        "end_chunk": {"type": "integer", "description": "Last chunk index belonging to this topic (inclusive)"},
                    },
                    "required": ["title", "summary", "start_chunk", "end_chunk"],
                },
            },
        },
        "required": ["topics"],
    },
}

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extraction expert. Extract factual triples (subject, relation, object) from the given text.

Rules:
- Each triple must have: subject, subject_type, relation, object, object_type
- Entity types should come from this list when possible: {entity_types}
- Relation types should come from this list when possible: {relation_types}
- You MAY use new entity/relation types if the existing ones don't fit
- Normalize entity names: use proper nouns, not pronouns or vague references
- Replace pronouns ("it", "they", "this") with the actual entity name from context
- Relations should be UPPER_SNAKE_CASE
- Be specific: prefer precise relation types over generic ones when the context is clear
- Only extract facts explicitly stated in the text, not inferences"""

EXTRACTION_TOOL = {
    "name": "extract_triples",
    "description": "Extract knowledge graph triples from text",
    "input_schema": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Entity name (proper noun, not pronoun)"},
                        "subject_type": {"type": "string", "description": "Entity type (e.g. ORGANIZATION, PERSON)"},
                        "relation": {"type": "string", "description": "Relation in UPPER_SNAKE_CASE"},
                        "object": {"type": "string", "description": "Entity name (proper noun, not pronoun)"},
                        "object_type": {"type": "string", "description": "Entity type"},
                    },
                    "required": ["subject", "subject_type", "relation", "object", "object_type"],
                },
            },
        },
        "required": ["triples"],
    },
}
