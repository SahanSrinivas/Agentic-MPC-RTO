"""The ONE place the LLM backend is configured (importable everywhere).

By default an OpenAI-compatible client points at a local Ollama server running ``qwen3:30b``.
The backend is selected by the ``AGENTIC_MPC_BACKEND`` environment variable (``ollama`` default):
set it to ``anthropic`` to run the SAME agent (same prompt, tools, cadence) against Claude via
Anthropic's OpenAI-compatible endpoint -- the controlled model-swap experiment. Nothing else in
the codebase reads the backend config.

Credentials/selection are read from the environment; a ``.env`` file at the repo root (gitignored)
is also loaded if present, so you can put the key in one file instead of exporting it:

    AGENTIC_MPC_BACKEND=anthropic
    ANTHROPIC_API_KEY=sk-ant-...
    AGENTIC_MPC_MODEL=claude-sonnet-4-6     # optional; this is the default
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from openai import OpenAI

_ENV_FILE = pathlib.Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True)
class LLMConfig:
    """Backend configuration for the supervisory agent's LLM."""
    base_url: str = "http://localhost:11434/v1"   # Ollama default
    api_key: str = "ollama"                        # Ollama ignores this; SDK requires a value
    model: str = "qwen3:30b"                       # pod's RunPod default
    temperature: float = 0.1                       # pod's RunPod default
    seed: int | None = None                        # request seed for reproducible LLM sampling


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from a repo-root ``.env`` into os.environ (no dependency).

    Existing environment variables win (an explicit ``export`` overrides the file). Blank lines
    and ``#`` comments are skipped; surrounding quotes are stripped.
    """
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _resolve_config() -> LLMConfig:
    """Select the backend from ``AGENTIC_MPC_BACKEND`` (default ollama)."""
    _load_dotenv()
    backend = os.environ.get("AGENTIC_MPC_BACKEND", "ollama").lower()
    if backend == "anthropic":
        return LLMConfig(
            base_url="https://api.anthropic.com/v1/",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.environ.get("AGENTIC_MPC_MODEL", "claude-sonnet-4-6"),
            temperature=0.1,
            seed=42,
        )
    # default: Ollama / qwen (unchanged)
    return LLMConfig()


# Single source of truth, resolved at import time from the environment.
LLM_CONFIG = _resolve_config()


def make_client(config: LLMConfig = LLM_CONFIG) -> OpenAI:
    """Construct the OpenAI-compatible client for the configured backend."""
    return OpenAI(base_url=config.base_url, api_key=config.api_key)
