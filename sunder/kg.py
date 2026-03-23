"""
sunder.kg -- Knowledge graph extraction and normalization.

Extractors:
  - LLMExtractor: Provider-agnostic extractor via LLMProvider (preferred)
  - AnthropicExtractor: Backward-compat wrapper (creates AnthropicProvider)
  - OllamaExtractor: Backward-compat wrapper (creates OllamaProvider)

All implement the KGExtractor protocol.
"""

from __future__ import annotations

import re
import time

from .constants import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TOOL,
)
from .llm import LLMProvider, get_provider
from .types import Chunk, ExtractionResult, Ontology, Triple


# -- Entity / relation normalization ------------------------------------------

def normalize_entity(name: str, aliases: dict[str, str] | None = None) -> str:
    """Normalize entity name to canonical form.

    Args:
        name: Raw entity name.
        aliases: Optional mapping of lowercase aliases to canonical names.
                 e.g. {"the system": "PostgreSQL", "the framework": "Django"}
    """
    stripped = name.strip().strip('"').strip("'")
    if aliases:
        lookup = stripped.lower()
        if lookup in aliases:
            return aliases[lookup]
    if stripped and stripped[0].islower():
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def normalize_relation(label: str) -> str:
    """Normalize relation labels to UPPER_SNAKE_CASE."""
    label = label.strip()
    if re.match(r'^[A-Z_]+$', label):
        return label
    label = re.sub(r'[^a-zA-Z0-9]', '_', label)
    label = re.sub(r'_+', '_', label).strip('_')
    return label.upper()


def normalize_triples(
    triples: list[Triple],
    aliases: dict[str, str] | None = None,
) -> list[Triple]:
    """Apply entity and relation normalization to all triples."""
    for t in triples:
        t.subject = normalize_entity(t.subject, aliases)
        t.object = normalize_entity(t.object, aliases)
        t.relation = normalize_relation(t.relation)
        if t.subject_type:
            t.subject_type = t.subject_type.upper().replace(" ", "_")
        if t.object_type:
            t.object_type = t.object_type.upper().replace(" ", "_")
    return triples


def deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    """Remove duplicate (subject, relation, object) triples, keeping first occurrence."""
    seen = set()
    result = []
    for t in triples:
        key = (t.subject.lower(), t.relation, t.object.lower())
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


# -- Provider-agnostic extractor ----------------------------------------------

class LLMExtractor:
    """KG extractor using any LLMProvider."""

    name = "llm"

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or get_provider()

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        ontology: Ontology,
        max_triples_per_chunk: int = 15,
    ) -> ExtractionResult:
        all_triples: list[Triple] = []
        total_input = 0
        total_output = 0
        api_calls = 0

        t0 = time.perf_counter()

        for chunk in chunks:
            system = EXTRACTION_SYSTEM_PROMPT.format(
                entity_types=", ".join(ontology.entity_types),
                relation_types=", ".join(ontology.relation_types),
            )

            result = self.provider.complete_with_tool(
                messages=[{
                    "role": "user",
                    "content": f"Extract up to {max_triples_per_chunk} knowledge graph triples from this text:\n\n{chunk.text}",
                }],
                tools=[EXTRACTION_TOOL],
                tool_choice="extract_triples",
                system=system,
                max_tokens=2048,
                temperature=0.1,
            )

            api_calls += 1
            total_input += result.input_tokens
            total_output += result.output_tokens

            raw_triples = result.input.get("triples", [])
            new_entity_types = set()
            new_relation_types = set()

            for triple_idx, t in enumerate(raw_triples):
                triple = Triple(
                    subject=t["subject"],
                    subject_type=t.get("subject_type", ""),
                    relation=t.get("relation", ""),
                    object=t["object"],
                    object_type=t.get("object_type", ""),
                    triple_id=f"{chunk.chunk_id}_triple_{triple_idx:03d}",
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                )
                all_triples.append(triple)
                new_entity_types.add(triple.subject_type)
                new_entity_types.add(triple.object_type)
                new_relation_types.add(triple.relation)

            ontology.expand(new_entity_types, new_relation_types)

        elapsed = time.perf_counter() - t0

        # Estimate cost (Haiku pricing: $0.80/MTok input, $4/MTok output)
        cost = (total_input * 0.80 + total_output * 4.0) / 1_000_000

        entity_types_found = set()
        relation_types_found = set()
        for t in all_triples:
            entity_types_found.add(t.subject_type)
            entity_types_found.add(t.object_type)
            relation_types_found.add(t.relation)

        return ExtractionResult(
            triples=all_triples,
            elapsed=elapsed,
            api_calls=api_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=cost,
            entity_types_discovered=entity_types_found,
            relation_types_discovered=relation_types_found,
        )


# -- Backward-compat wrappers ------------------------------------------------

class AnthropicExtractor(LLMExtractor):
    """KG extractor using Anthropic Claude API. Thin wrapper around LLMExtractor."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from .llm import AnthropicProvider
        provider = AnthropicProvider(model=model, api_key=api_key)
        super().__init__(provider)


class OllamaExtractor(LLMExtractor):
    """KG extractor using a local Ollama model. Thin wrapper around LLMExtractor."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        from .llm import OllamaProvider
        provider = OllamaProvider(model=model, base_url=base_url)
        super().__init__(provider)
