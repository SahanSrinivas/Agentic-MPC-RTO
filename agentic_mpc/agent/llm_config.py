"""The ONE place the LLM backend is configured (importable everywhere).

For the paper we point an OpenAI-compatible client at a local Ollama server running
``qwen3:4b``. Swapping to Claude Sonnet for production is a single change here: set
``base_url`` to the Anthropic OpenAI-compatible endpoint, ``api_key`` to the Anthropic
key, and ``model`` to a ``claude-sonnet-4-*`` id -- nothing else in the codebase reads
the backend config, so no other file changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    """Backend configuration for the supervisory agent's LLM."""

    base_url: str = "http://localhost:11434/v1"   # Ollama default
    api_key: str = "ollama"                        # Ollama ignores this; SDK requires a value
    model: str = "qwen3:4b"
    temperature: float = 0.2


# Single source of truth. To run on Claude Sonnet instead, change ONLY this object, e.g.:
#   LLM_CONFIG = LLMConfig(base_url="https://api.anthropic.com/v1/",
#                          api_key=os.environ["ANTHROPIC_API_KEY"],
#                          model="claude-sonnet-4-6")
LLM_CONFIG = LLMConfig()


def make_client(config: LLMConfig = LLM_CONFIG) -> OpenAI:
    """Construct the OpenAI-compatible client for the configured backend."""
    return OpenAI(base_url=config.base_url, api_key=config.api_key)
