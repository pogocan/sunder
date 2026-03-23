"""Tests for sunder.llm -- provider factory and routing."""

import os
import sys
from unittest import mock

import pytest

from sunder.llm import get_provider, AnthropicProvider, OllamaProvider, OpenAIProvider
from sunder.types import AgentConfig


class TestGetProviderRouting:
    """get_provider() selects the right provider class."""

    def test_anthropic_when_api_key_set(self, monkeypatch):
        """ANTHROPIC_API_KEY in env → AnthropicProvider (auto-detect)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
        # Mock the anthropic import so we don't need the real package
        fake_anthropic = mock.MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)
        assert provider.model  # should have a default model

    def test_ollama_when_config_says_ollama(self, monkeypatch):
        """AgentConfig(provider="ollama") → OllamaProvider."""
        # Mock requests so OllamaProvider doesn't actually connect
        fake_requests = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.raise_for_status = mock.MagicMock()
        fake_requests.get.return_value = fake_response
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        cfg = AgentConfig(provider="ollama", ollama_base_url="http://fake:11434")
        provider = get_provider(cfg)
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://fake:11434"

    def test_openai_missing_package_raises_clear_error(self, monkeypatch):
        """provider="openai" with no openai package → ImportError with install hint."""
        # Temporarily hide the openai module
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
        monkeypatch.delitem(sys.modules, "openai", raising=False)

        # Make import fail
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        cfg = AgentConfig(provider="openai")
        with pytest.raises(ImportError, match="pip install openai"):
            get_provider(cfg)

    def test_unknown_provider_raises_valueerror(self):
        """Unknown provider name → clear ValueError."""
        cfg = AgentConfig(provider="grok")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider(cfg)

    def test_auto_detect_prefers_anthropic_over_openai(self, monkeypatch):
        """When both ANTHROPIC_API_KEY and OPENAI_API_KEY are set, anthropic wins."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-fake")
        fake_anthropic = mock.MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_explicit_config_overrides_env(self, monkeypatch):
        """AgentConfig(provider="ollama") wins even if ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        fake_requests = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.raise_for_status = mock.MagicMock()
        fake_requests.get.return_value = fake_response
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        cfg = AgentConfig(provider="ollama")
        provider = get_provider(cfg)
        assert isinstance(provider, OllamaProvider)
