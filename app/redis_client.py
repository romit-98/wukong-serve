"""Shared async Redis client.

A single connection pool is created per process and reused everywhere that
needs Redis (auth lookups, rate-limit counters, session history, the
active-model setting). Keeping it centralised means tests can swap in
``fakeredis`` by overriding ``get_redis`` via FastAPI's dependency system.
"""

from __future__ import annotations

import redis.asyncio as redis

from app.config import get_settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Return the process-wide async Redis client, creating it lazily."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _client


async def close_redis() -> None:
    """Close the Redis connection pool on shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
