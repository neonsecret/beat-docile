"""[ACTIVE] Provider-agnostic Anthropic LLM client with rate-limiting and prompt caching.

Status: ACTIVE — used in current best (v2_ensemble).
See KNOWLEDGE_BASE.md §3 for the architecture map.

Backend selected via BD_LLM_BACKEND env var:
  "anthropic" (default): direct Anthropic API, requires ANTHROPIC_API_KEY.
  "vertex":              Vertex AI backend, requires VERTEX_PROJECT_ID and VERTEX_LOCATION.

Ref: EVAL_SPEC §1 (caching mandatory).
"""

from __future__ import annotations

import asyncio
import os

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_client: anthropic.Anthropic | None = None
_semaphore: asyncio.Semaphore | None = None


def get_client() -> anthropic.Anthropic:
    """Return the singleton Anthropic client, initializing it on first call.

    Backend controlled by BD_LLM_BACKEND env var ("anthropic" or "vertex").
    """
    global _client
    if _client is None:
        backend = os.environ.get("BD_LLM_BACKEND", "anthropic")
        if backend == "vertex":
            from anthropic import AnthropicVertex

            from .config import VERTEX_LOCATION, VERTEX_PROJECT_ID

            _client = AnthropicVertex(project_id=VERTEX_PROJECT_ID, region=VERTEX_LOCATION)
        else:
            _client = anthropic.Anthropic()
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(8)
    return _semaphore


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
async def complete(
    model: str,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    cache_system: bool = True,
    temperature: float = 1.0,
) -> anthropic.types.Message:
    """Send a chat completion request via the configured LLM backend with optional prompt caching."""
    client = get_client()
    sem = _get_semaphore()

    system_param: list[dict] | str
    if cache_system:
        system_param = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_param = system

    async with sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_param,
                messages=messages,
                temperature=temperature,
            ),
        )
