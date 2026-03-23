"""Tests for SunderConfig persistence and AgentConfig runtime behavior."""

import json
from dataclasses import asdict
from pathlib import Path

from sunder.types import AgentConfig, SunderConfig


class TestSunderConfigPersistence:
    """SunderConfig round-trips through config.json correctly."""

    def test_vector_store_persists(self, tmp_path):
        """vector_store field survives write → read cycle."""
        cfg = SunderConfig(vector_store="faiss")
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2))

        loaded = json.loads(config_path.read_text())
        restored = SunderConfig(**loaded)
        assert restored.vector_store == "faiss"

    def test_all_fields_round_trip(self, tmp_path):
        """Every SunderConfig field survives serialization."""
        cfg = SunderConfig(
            chunking_mode="topic_sentence",
            chunk_size=150,
            chunk_overlap=20,
            min_chunk_size=30,
            atomic_line_length=120,
            embedding_model="custom/model",
            embedding_dim=768,
            embedding_batch_size=64,
            graph_max_chunks=100,
            vector_store="faiss",
            default_top_k=5,
            search_oversample=2,
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2))

        loaded = json.loads(config_path.read_text())
        restored = SunderConfig(**loaded)
        assert asdict(restored) == asdict(cfg)

    def test_default_vector_store_is_faiss(self):
        """Default value for vector_store is 'faiss'."""
        cfg = SunderConfig()
        assert cfg.vector_store == "faiss"


class TestAgentConfigDefaults:
    """AgentConfig has the expected default values."""

    def test_default_provider(self):
        cfg = AgentConfig()
        assert cfg.provider == "anthropic"

    def test_default_model_is_none(self):
        cfg = AgentConfig()
        assert cfg.model is None

    def test_default_max_steps(self):
        cfg = AgentConfig()
        assert cfg.max_steps == 10

    def test_default_max_reads(self):
        cfg = AgentConfig()
        assert cfg.max_reads == 4

    def test_default_ollama_base_url(self):
        cfg = AgentConfig()
        assert cfg.ollama_base_url == "http://localhost:11434"


class TestAgentConfigNotPersisted:
    """AgentConfig must NOT appear in config.json (it's runtime-only)."""

    def test_agent_config_not_in_sunder_config_json(self, tmp_path):
        """Serializing SunderConfig does not include AgentConfig fields."""
        cfg = SunderConfig()
        data = asdict(cfg)
        # AgentConfig fields must not leak into SunderConfig serialization
        assert "provider" not in data
        assert "max_steps" not in data
        assert "max_reads" not in data
        assert "ollama_base_url" not in data

    def test_agent_config_is_separate_type(self):
        """AgentConfig and SunderConfig are distinct classes."""
        assert AgentConfig is not SunderConfig
        ac = AgentConfig()
        sc = SunderConfig()
        # AgentConfig should not have chunking fields
        assert not hasattr(ac, "chunk_size")
        # SunderConfig should not have agent fields
        assert not hasattr(sc, "max_steps")
