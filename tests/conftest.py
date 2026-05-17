"""Shared pytest fixtures.

The strategy:

* Replace the Redis client with a ``fakeredis`` instance so tests don't need
  a running Redis container.
* Replace the Ollama client with a hand-rolled fake that records calls and
  returns canned responses. CI never reaches the real Ollama server.
* Build the FastAPI app fresh per test via the application factory so state
  doesn't bleed across tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Configure env BEFORE importing the app so settings pick it up. We use direct
# assignment (not setdefault) because CI sets some of these to different values
# and the test suite has its own contract about what they should be.
os.environ["ADMIN_SECRET"] = "test-admin-secret"
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ["OLLAMA_DEFAULT_MODEL"] = "test-model"
os.environ["RATE_LIMIT_FREE_RPM"] = "3"
os.environ["RATE_LIMIT_PRO_RPM"] = "10"

from app.config import get_settings  # noqa: E402
from app.deps import (  # noqa: E402
    get_breaker_dep,
    get_ollama_dep,
    get_redis_dep,
    get_session_store_dep,
)
from app.main import create_app  # noqa: E402
from app.services.circuit_breaker import CircuitBreaker  # noqa: E402
from app.services.ollama_client import OllamaResponse  # noqa: E402
from app.services.session import SessionStore  # noqa: E402


@dataclass
class FakeOllama:
    """Minimal stand-in for ``OllamaClient`` used in tests."""

    response_text: str = "hello world"
    tokens: int = 7
    raise_error: Exception | None = None
    healthy: bool = True
    stream_chunks: list[str] = field(
        default_factory=lambda: ["he", "llo", " world"]
    )
    calls: list[dict] = field(default_factory=list)

    async def generate(
        self, model: str, prompt: str, max_tokens: int
    ) -> OllamaResponse:
        self.calls.append(
            {"model": model, "prompt": prompt, "max_tokens": max_tokens}
        )
        if self.raise_error is not None:
            raise self.raise_error
        return OllamaResponse(
            text=self.response_text, model=model, tokens_generated=self.tokens
        )

    async def generate_stream(
        self, model: str, prompt: str, max_tokens: int
    ) -> AsyncIterator[dict]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            }
        )
        if self.raise_error is not None:
            raise self.raise_error
        for chunk in self.stream_chunks:
            yield {"response": chunk, "done": False}
        yield {"response": "", "done": True, "eval_count": self.tokens}

    async def health(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:  # pragma: no cover — nothing to close
        return None


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """A clean fakeredis instance for the duration of one test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.flushall()
        await client.aclose()


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest_asyncio.fixture
async def app(fake_redis, fake_ollama):  # type: ignore[no-untyped-def]
    """Build a fresh FastAPI app with all external deps overridden."""
    settings = get_settings()
    app = create_app()
    breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_threshold,
        reset_timeout_seconds=settings.circuit_breaker_timeout_seconds,
    )
    session_store = SessionStore(
        redis=fake_redis,
        ttl_seconds=settings.session_ttl_seconds,
        max_turns=settings.session_max_turns,
    )

    app.dependency_overrides[get_redis_dep] = lambda: fake_redis
    app.dependency_overrides[get_ollama_dep] = lambda: fake_ollama
    app.dependency_overrides[get_breaker_dep] = lambda: breaker
    app.dependency_overrides[get_session_store_dep] = lambda: session_store
    return app


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:  # type: ignore[no-untyped-def]
    """Async HTTP client bound directly to the ASGI app (no socket)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def free_key(client) -> str:  # type: ignore[no-untyped-def]
    """Mint a free-tier API key via the admin endpoint and return it."""
    r = await client.post(
        "/admin/keys",
        json={"owner": "alice", "tier": "free"},
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    assert r.status_code == 201, r.text
    return r.json()["key"]


@pytest_asyncio.fixture
async def pro_key(client) -> str:  # type: ignore[no-untyped-def]
    r = await client.post(
        "/admin/keys",
        json={"owner": "bob", "tier": "pro"},
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    assert r.status_code == 201, r.text
    return r.json()["key"]
