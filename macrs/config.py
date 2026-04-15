"""LLM configuration. Uses LiteLLM — set the model string to any supported provider."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"  # any litellm-compatible model string
    temperature: float = 0.7
    max_tokens: int = 300
