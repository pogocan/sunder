"""
sunder.llm -- LLM provider abstraction.

All LLM calls in sunder go through LLMProvider so the pipeline is
provider-agnostic. Provider packages (anthropic, openai) are imported
lazily -- only the one you use needs to be installed.

Providers:
  - AnthropicProvider: Claude API (default)
  - OpenAIProvider: OpenAI-compatible APIs
  - OllamaProvider: Local models via Ollama HTTP API

Factory:
  - get_provider(config) -> LLMProvider
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .constants import ANTHROPIC_MODEL


# -- Result types -------------------------------------------------------------

@dataclass
class ToolResult:
    """Structured output from an LLM tool-use call."""
    tool_name: str
    input: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


# -- Base class ---------------------------------------------------------------

class LLMProvider:
    """Abstract LLM provider. Subclass and implement both methods."""

    name: str = "base"

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Plain text completion. Returns the assistant's text response."""
        raise NotImplementedError

    def complete_with_tool(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ToolResult:
        """Forced tool-use completion.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in *Anthropic* format:
                   [{"name": ..., "description": ..., "input_schema": {...}}]
            tool_choice: Name of the tool to force.
            system: System prompt.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.

        Returns:
            ToolResult with the parsed tool input dict and usage stats.
        """
        raise NotImplementedError


# -- Anthropic ----------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Claude API via the anthropic Python SDK."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Set the environment variable or "
                "pass api_key= to AnthropicProvider."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or ANTHROPIC_MODEL

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def complete_with_tool(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ToolResult:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_choice},
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_choice:
                return ToolResult(
                    tool_name=block.name,
                    input=block.input,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

        # Fallback: tool not found in response (shouldn't happen with forced choice)
        return ToolResult(
            tool_name=tool_choice,
            input={},
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )


# -- OpenAI -------------------------------------------------------------------

def _anthropic_tool_to_openai(tool: dict) -> dict:
    """Convert Anthropic tool schema to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API via the openai Python SDK."""

    name = "openai"

    _DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set the environment variable or "
                "pass api_key= to OpenAIProvider."
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model or self._DEFAULT_MODEL

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
        )
        return response.choices[0].message.content or ""

    def complete_with_tool(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ToolResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        oai_tools = [_anthropic_tool_to_openai(t) for t in tools]

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
            tools=oai_tools,
            tool_choice={"type": "function", "function": {"name": tool_choice}},
        )

        usage = response.usage
        message = response.choices[0].message

        if message.tool_calls:
            tc = message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            return ToolResult(
                tool_name=tc.function.name,
                input=args,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )

        return ToolResult(
            tool_name=tool_choice,
            input={},
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


# -- Ollama -------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Local LLM via Ollama HTTP API. Uses tool_use with JSON fallback."""

    name = "ollama"

    _DEFAULT_MODEL = "qwen2.5:3b"

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
    ):
        import requests as _requests  # stdlib-adjacent, always available

        self._requests = _requests
        self.model = model or self._DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")

        try:
            r = _requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {self.base_url}: {e}")

    def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        r = self._requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": msgs,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def complete_with_tool(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> ToolResult:
        # Try native tool_use first
        result = self._try_tool_use(messages, tools, tool_choice, system, max_tokens, temperature)
        if result is not None:
            return result

        # Fallback: ask for JSON matching the tool's schema
        return self._json_fallback(messages, tools, tool_choice, system, max_tokens, temperature)

    def _try_tool_use(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> ToolResult | None:
        """Try Ollama's native tool_use support."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        # Ollama uses OpenAI-style tool format
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

        try:
            r = self._requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "tools": ollama_tools,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

            tool_calls = data.get("message", {}).get("tool_calls", [])
            if not tool_calls:
                return None

            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == tool_choice:
                    return ToolResult(
                        tool_name=tool_choice,
                        input=fn.get("arguments", {}),
                    )
        except Exception:
            return None

        return None

    def _json_fallback(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> ToolResult:
        """Fallback: ask the model to output JSON matching the tool schema."""
        # Find the target tool's schema
        target_schema = {}
        for t in tools:
            if t["name"] == tool_choice:
                target_schema = t["input_schema"]
                break

        # Build a prompt that asks for JSON
        user_msg = messages[-1]["content"] if messages else ""
        json_prompt = (
            f"{system}\n\n{user_msg}\n\n"
            f"Respond with ONLY a JSON object matching this schema:\n"
            f"{json.dumps(target_schema, indent=2)}\n"
            f"No markdown, no explanation, just the JSON object."
        )

        try:
            r = self._requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": json_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )
            r.raise_for_status()
            text = r.json().get("response", "")
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                parsed = {}
            return ToolResult(tool_name=tool_choice, input=parsed)
        except Exception:
            return ToolResult(tool_name=tool_choice, input={})


# -- Factory ------------------------------------------------------------------

# Default models per provider (used when AgentConfig.model is None)
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": ANTHROPIC_MODEL,
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:3b",
}


def get_provider(config: Any = None) -> LLMProvider:
    """Create an LLMProvider from an AgentConfig, or auto-detect from env vars.

    Resolution order:
      1. If config is provided, use config.provider
      2. If ANTHROPIC_API_KEY is set -> AnthropicProvider
      3. If OPENAI_API_KEY is set -> OpenAIProvider
      4. If OLLAMA_HOST is set -> OllamaProvider
      5. Default: AnthropicProvider (will fail with a clear error if no key)

    Args:
        config: An AgentConfig instance, or None for auto-detection.

    Returns:
        An initialized LLMProvider.
    """
    # Import here to avoid circular dependency (types imports nothing from llm)
    from .types import AgentConfig

    if config is None:
        config = AgentConfig()
        # Auto-detect: override default provider based on available env vars
        if os.environ.get("ANTHROPIC_API_KEY"):
            config.provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            config.provider = "openai"
        elif os.environ.get("OLLAMA_HOST"):
            config.provider = "ollama"
            config.ollama_base_url = os.environ["OLLAMA_HOST"]

    model = config.model  # None means "use provider default"

    if config.provider == "anthropic":
        return AnthropicProvider(model=model)
    elif config.provider == "openai":
        return OpenAIProvider(model=model)
    elif config.provider == "ollama":
        return OllamaProvider(model=model, base_url=config.ollama_base_url)
    else:
        raise ValueError(
            f"Unknown LLM provider: {config.provider!r}. "
            f"Supported: 'anthropic', 'openai', 'ollama'"
        )
