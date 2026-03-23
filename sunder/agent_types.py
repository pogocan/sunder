"""
sunder.agent_types -- Data types for the agent loop.

Separated from agent.py to keep imports light and avoid circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentResult:
    """Result of an agent.ask() call."""
    answer: str
    citations: list[str]        # chunk_ids referenced in answer
    chunks_read: list[str]      # chunk_ids actually opened via read_chunk
    follow_ups: list[str]       # 2-3 suggested follow-up questions
    steps: int
    run_id: str                 # uuid4 hex string


@dataclass
class AgentTool:
    """An injectable tool the agent can call during its loop.

    func receives a query string and returns a result string.
    The agent sees the tool's name and description, then can invoke it.
    """
    name: str
    description: str
    func: Callable[[str], str]


@dataclass
class AgentMemory:
    """Persistent memory across agent runs.

    All fields are optional -- the agent degrades gracefully if any are empty.
    """
    knowledge_sheet: dict = field(default_factory=dict)      # structured key/value facts
    answer_cache: list[dict] = field(default_factory=list)   # past {question, answer, chunk_ids}
    user_notes: str = ""                                     # freeform persistent notes
