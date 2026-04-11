"""
sunder -- Document intelligence pipeline. PDF to searchable knowledge.
"""

__version__ = "0.1.0"

from .agent import Agent
from .agent_types import AgentMemory, AgentResult, AgentTool
from .curator import apply_curation, curate_text, load_report, save_report
from .ingest import ingest
from .index import Corpus
from .types import (
    AgentConfig,
    Chunk,
    Ontology,
    SearchHit,
    Sentence,
    SunderConfig,
    Triple,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentMemory",
    "AgentResult",
    "AgentTool",
    "ingest",
    "Corpus",
    "SunderConfig",
    "Chunk",
    "Sentence",
    "Triple",
    "SearchHit",
    "Ontology",
    "curate_text",
    "apply_curation",
    "save_report",
    "load_report",
]
