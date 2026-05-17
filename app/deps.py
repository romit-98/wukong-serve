"""FastAPI dependency providers.

Long-lived objects (Redis client, Ollama client, circuit breaker) are attached
to ``app.state`` in the lifespan handler. These helpers fetch them out of the
request so individual routes never instantiate their own.
"""

from __future__ import annotations

from fastapi import Request
from redis.asyncio import Redis

from app.services.circuit_breaker import CircuitBreaker
from app.services.ollama_client import OllamaClient
from app.services.session import SessionStore


def get_redis_dep(request: Request) -> Redis:
    """Return the process-wide Redis client from app state."""
    return request.app.state.redis


def get_ollama_dep(request: Request) -> OllamaClient:
    """Return the shared Ollama client."""
    return request.app.state.ollama


def get_breaker_dep(request: Request) -> CircuitBreaker:
    """Return the shared circuit breaker."""
    return request.app.state.breaker


def get_session_store_dep(request: Request) -> SessionStore:
    """Return the shared session store."""
    return request.app.state.session_store
