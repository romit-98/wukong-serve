"""Async HTTP client for the Ollama backend.

Wraps Ollama's ``/api/generate`` endpoint with:

* a configurable per-request timeout
* a circuit breaker that opens after N consecutive failures
* streaming and non-streaming variants returning a uniform shape

The rest of the application should never talk to Ollama directly — go through
``OllamaClient`` so that timeouts, retries, and breaker state are all
consistent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.services.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised on any non-recoverable Ollama failure (HTTP error, bad JSON)."""


class OllamaTimeoutError(OllamaError):
    """Raised when the backend takes longer than the configured timeout."""


@dataclass
class OllamaResponse:
    """Non-streaming response from Ollama, normalised."""

    text: str
    model: str
    tokens_generated: int


class OllamaClient:
    """Thin async wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        breaker: CircuitBreaker,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._breaker = breaker
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we created it."""
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        """Return True if the Ollama base URL is reachable."""
        try:
            r = await self._client.get(f"{self._base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> OllamaResponse:
        """Send a non-streaming generation request to Ollama."""
        await self._breaker.before_call()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            r = await self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            r.raise_for_status()
            data: dict[str, Any] = r.json()
        except httpx.TimeoutException as e:
            await self._breaker.record_failure()
            raise OllamaTimeoutError(
                f"Ollama timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPError as e:
            await self._breaker.record_failure()
            raise OllamaError(f"Ollama HTTP error: {e}") from e
        except ValueError as e:  # JSON decode
            await self._breaker.record_failure()
            raise OllamaError("Ollama returned invalid JSON") from e

        await self._breaker.record_success()
        return OllamaResponse(
            text=data.get("response", ""),
            model=data.get("model", model),
            tokens_generated=int(data.get("eval_count", 0)),
        )

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield Ollama streaming chunks as dicts.

        Each yielded dict contains at least ``response`` (the new token text)
        and ``done`` (bool). The final chunk also contains ``eval_count`` so
        the caller can record token-generation metrics.
        """
        await self._breaker.before_call()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        logger.warning(
                            "ollama_stream_bad_chunk", extra={"line": line[:200]}
                        )
                        continue
                    yield chunk
                    if chunk.get("done"):
                        break
        except httpx.TimeoutException as e:
            await self._breaker.record_failure()
            raise OllamaTimeoutError(
                f"Ollama timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPError as e:
            await self._breaker.record_failure()
            raise OllamaError(f"Ollama HTTP error: {e}") from e
        else:
            await self._breaker.record_success()


__all__ = [
    "CircuitOpenError",
    "OllamaClient",
    "OllamaError",
    "OllamaResponse",
    "OllamaTimeoutError",
]
