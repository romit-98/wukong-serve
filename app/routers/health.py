"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from redis.asyncio import Redis

from app.deps import get_ollama_dep, get_redis_dep
from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["health"])


class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    components: list[ComponentHealth]


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    redis: Redis = Depends(get_redis_dep),
    ollama: OllamaClient = Depends(get_ollama_dep),
) -> HealthResponse:
    """Report Redis and Ollama reachability.

    Returns HTTP 200 if both dependencies are healthy, HTTP 503 otherwise.
    Kubernetes-style: liveness should call ``/health/live`` and readiness
    should call this endpoint.
    """
    components: list[ComponentHealth] = []

    try:
        pong = await redis.ping()
        components.append(ComponentHealth(name="redis", healthy=bool(pong)))
    except Exception as e:  # noqa: BLE001 — surface any redis error verbatim
        components.append(
            ComponentHealth(name="redis", healthy=False, detail=str(e))
        )

    ollama_ok = await ollama.health()
    components.append(ComponentHealth(name="ollama", healthy=ollama_ok))

    overall_ok = all(c.healthy for c in components)
    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if overall_ok else "degraded", components=components
    )


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Process-only liveness check — does not touch dependencies."""
    return {"status": "alive"}
