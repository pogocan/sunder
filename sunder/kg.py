"""
sunder.kg -- Knowledge graph extraction and normalization.

Extractors:
  - AnthropicExtractor: Claude API with tool_use for structured output
  - OllamaExtractor: Local LLM via Ollama with tool_use or JSON fallback

Both implement the KGExtractor protocol, so you can swap or add your own.
"""

from __future__ import annotations

import json
import os
import re
import time

from .constants import (
    ANTHROPIC_MODEL,
    ENTITY_TYPES,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TOOL,
    OLLAMA_URL,
    RELATION_TYPES,
)
from .types import Chunk, ExtractionResult, KGExtractor, Ontology, Triple


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


# -- Anthropic Extractor ------------------------------------------------------

class AnthropicExtractor:
    """KG extractor using the Anthropic Claude API with tool_use."""
    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, api_key: str | None = None):
        import anthropic

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

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

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                system=system,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_triples"},
                messages=[{
                    "role": "user",
                    "content": f"Extract up to {max_triples_per_chunk} knowledge graph triples from this text:\n\n{chunk.text}",
                }],
            )

            api_calls += 1
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            triple_idx = 0
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_triples":
                    raw_triples = block.input.get("triples", [])
                    new_entity_types = set()
                    new_relation_types = set()

                    for t in raw_triples:
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
                        triple_idx += 1
                        new_entity_types.add(triple.subject_type)
                        new_entity_types.add(triple.object_type)
                        new_relation_types.add(triple.relation)

                    ontology.expand(new_entity_types, new_relation_types)

        elapsed = time.perf_counter() - t0

        # Haiku pricing: $0.80/MTok input, $4/MTok output
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


# -- Ollama Extractor ---------------------------------------------------------

class OllamaExtractor:
    """KG extractor using a local Ollama model with tool_use or JSON fallback."""
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = OLLAMA_URL):
        import requests

        self.model = model
        self.base_url = base_url
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {self.base_url}: {e}")

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        ontology: Ontology,
        max_triples_per_chunk: int = 15,
    ) -> ExtractionResult:
        all_triples: list[Triple] = []
        api_calls = 0

        t0 = time.perf_counter()

        for chunk in chunks:
            prompt = self._build_prompt(chunk.text, max_triples_per_chunk)

            raw_triples = self._try_tool_use(prompt, ontology, max_triples_per_chunk)
            if raw_triples is None:
                raw_triples = self._try_json_prompt(chunk.text, ontology, max_triples_per_chunk)

            api_calls += 1

            new_entity_types = set()
            new_relation_types = set()
            for triple_idx, raw in enumerate(raw_triples or []):
                triple = Triple(
                    subject=raw["subject"],
                    subject_type=raw.get("subject_type", ""),
                    relation=raw.get("relation", ""),
                    object=raw["object"],
                    object_type=raw.get("object_type", ""),
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
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            entity_types_discovered=entity_types_found,
            relation_types_discovered=relation_types_found,
        )

    def _try_tool_use(
        self, prompt: str, ontology: Ontology, max_triples: int,
    ) -> list[dict] | None:
        """Try Ollama's tool_use support. Returns list of raw dicts or None."""
        import requests

        tools = [{
            "type": "function",
            "function": {
                "name": "extract_triples",
                "description": "Extract knowledge graph triples from text",
                "parameters": EXTRACTION_TOOL["input_schema"],
            },
        }]

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT.format(
                            entity_types=", ".join(ontology.entity_types),
                            relation_types=", ".join(ontology.relation_types),
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    "tools": tools,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

            message = data.get("message", {})
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                return None

            for tc in tool_calls:
                if tc.get("function", {}).get("name") == "extract_triples":
                    args = tc["function"].get("arguments", {})
                    return args.get("triples", [])
        except Exception:
            return None

        return None

    def _try_json_prompt(
        self, chunk_text: str, ontology: Ontology, max_triples: int,
    ) -> list[dict]:
        """Fallback: ask model to output JSON directly. Returns list of raw dicts."""
        import requests

        json_prompt = f"""Extract up to {max_triples} knowledge graph triples from the text below.

Entity types: {", ".join(ontology.entity_types)}
Relation types: {", ".join(ontology.relation_types)}

Output ONLY a JSON array of objects with keys: subject, subject_type, relation, object, object_type.
No markdown, no explanation, just the JSON array.

Text:
{chunk_text}

JSON:"""

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": json_prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")

            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw = parsed.get("triples", parsed.get("results", []))
            elif isinstance(parsed, list):
                raw = parsed
            else:
                return []

            return [
                t for t in raw
                if isinstance(t, dict) and "subject" in t and "object" in t
            ]
        except Exception:
            return []

    def _build_prompt(self, chunk_text: str, max_triples: int) -> str:
        return f"Extract up to {max_triples} knowledge graph triples from this text:\n\n{chunk_text}"
