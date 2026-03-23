"""
sunder -- Document intelligence pipeline. PDF to searchable knowledge.
"""

__version__ = "0.1.0"

from .agent import Agent
from .agent_types import AgentMemory, AgentResult, AgentTool
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
]
