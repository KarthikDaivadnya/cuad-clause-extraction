"""
llm_provider.py
----------------
A thin, provider-agnostic wrapper around whichever LLM API you want to use.

Why this exists:
    The assignment says "use an LLM (API-based or open-source)". Rather than
    hard-wiring one vendor, this module defines a common `LLMProvider`
    interface with three concrete implementations (Groq, OpenAI, Anthropic).
    This makes the codebase:
      1. Easy to demo with a free/fast API (Groq) out of the box.
      2. Trivial to swap for model comparisons (see compare_models.py).
      3. Resilient to any single vendor's rate limits / outages.

Usage:
    provider = get_provider("groq", model="llama-3.3-70b-versatile")
    text = provider.complete(system="You are...", prompt="Extract...")
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Optional

from src import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


class LLMProvider(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        ...

    def complete(
        self,
        prompt: str,
        system: str = "You are a precise, helpful assistant.",
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        """Call the LLM with simple exponential-backoff retries."""
        last_err: Optional[Exception] = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                return self._call(system, prompt, max_tokens, temperature)
            except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
                last_err = exc
                wait = min(2 ** attempt, 20)
                logger.warning(
                    "LLM call failed (attempt %d/%d) on %s: %s. Retrying in %ds.",
                    attempt, config.MAX_RETRIES, self.__class__.__name__, exc, wait,
                )
                time.sleep(wait)
        raise LLMError(f"{self.__class__.__name__} failed after {config.MAX_RETRIES} attempts") from last_err


class GroqProvider(LLMProvider):
    """Fast, free-tier-friendly inference (Llama/Mixtral models)."""

    def __init__(self, model: str):
        super().__init__(model)
        from groq import Groq  # local import keeps the dependency optional
        api_key = config.API_KEYS["groq"]
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = Groq(api_key=api_key)

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        from openai import OpenAI
        api_key = config.API_KEYS["openai"]
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key)

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        super().__init__(model)
        import anthropic
        api_key = config.API_KEYS["anthropic"]
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


_PROVIDER_MAP = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str = None, model: str = None) -> LLMProvider:
    name = (name or config.DEFAULT_PROVIDER).lower()
    if name not in _PROVIDER_MAP:
        raise ValueError(f"Unknown provider '{name}'. Choose from {list(_PROVIDER_MAP)}")
    model = model or config.DEFAULT_MODELS[name]
    return _PROVIDER_MAP[name](model)
