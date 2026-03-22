"""
sunder -- Document intelligence pipeline. PDF to searchable knowledge.
"""

__version__ = "0.1.0"

from .ingest import ingest
from .index import Corpus
from .types import (
    Chunk,
    Ontology,
    SearchHit,
    Sentence,
    SunderConfig,
    Triple,
)

__all__ = [
    "ingest",
    "Corpus",
    "SunderConfig",
    "Chunk",
    "Sentence",
    "Triple",
    "SearchHit",
    "Ontology",
]
