"""FastAPI application entry point.

Builds the app, attaches long-lived resources (Redis, Ollama client, circuit
breaker, session store) to ``app.state`` via the lifespan handler, mounts
routers, configures structured logging, and wires the Prometheus instrumentor.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.logging_setup import configure_logging
from app.redis_client import close_redis, get_redis
from app.routers import admin, health, inference
from app.services.circuit_breaker import CircuitBreaker
from app.services.ollama_client import OllamaClient
from app.services.session import SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise long-lived resources and tear them down on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("app.startup")

    app.state.redis = get_redis()

    app.state.breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_threshold,
        reset_timeout_seconds=settings.circuit_breaker_timeout_seconds,
    )
    app.state.ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        breaker=app.state.breaker,
    )
    app.state.session_store = SessionStore(
        redis=app.state.redis,
        ttl_seconds=settings.session_ttl_seconds,
        max_turns=settings.session_max_turns,
    )

    logger.info(
        "startup_complete",
        extra={
            "ollama_base_url": settings.ollama_base_url,
            "default_model": settings.ollama_default_model,
        },
    )
    try:
        yield
    finally:
        # Graceful shutdown — FastAPI / uvicorn already waits for in-flight
        # requests to finish before invoking the teardown half of lifespan,
        # so by the time we get here it's safe to close clients.
        logger.info("shutdown_starting")
        await app.state.ollama.aclose()
        await close_redis()
        logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Application factory. Kept separate so tests can build fresh instances."""
    app = FastAPI(
        title="Wukong-serve",
        description="Production-realistic LLM inference serving system.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id and emit one structured log per request."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        logger = logging.getLogger("app.access")
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": int(elapsed * 1000),
                },
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "internal_server_error"},
                headers={"X-Request-ID": request_id},
            )
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": int(elapsed * 1000),
            },
        )
        return response

    app.include_router(health.router)
    app.include_router(inference.router)
    app.include_router(admin.router)

    # Prometheus /metrics endpoint and per-request instrumentation.
    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health", "/health/live"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
