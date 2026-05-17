"""Inference endpoints (non-streaming + SSE streaming)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.deps import get_ollama_dep, get_redis_dep, get_session_store_dep
from app.middleware.auth import ApiKeyPrincipal
from app.middleware.rate_limit import enforce_rate_limit
from app.observability.metrics import (
    active_inference_requests,
    inference_latency_seconds,
    inference_requests_total,
    inference_tokens_generated_total,
)
from app.services.ollama_client import (
    CircuitOpenError,
    OllamaClient,
    OllamaError,
    OllamaTimeoutError,
)
from app.services.session import SessionStore, Turn

router = APIRouter(prefix="/v1", tags=["inference"])
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = Field(
        default=None,
        description="Override the active model. If omitted, uses the server's "
        "currently selected model.",
    )
    max_tokens: int = Field(default=512, ge=1, le=4096)
    session_id: str | None = None


class InferenceResponse(BaseModel):
    response: str
    model: str
    latency_ms: int
    tokens_generated: int
    session_id: str


async def _resolve_active_model(redis: Redis, override: str | None) -> str:
    """Return the model to use, considering caller override and admin setting."""
    if override:
        return override
    settings = get_settings()
    stored = await redis.get(settings.key_active_model)
    return stored or settings.ollama_default_model


@router.post(
    "/inference",
    response_model=InferenceResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Backend unavailable or circuit open"},
        504: {"description": "Backend timeout"},
    },
)
async def inference(
    payload: InferenceRequest,
    principal: ApiKeyPrincipal = Depends(enforce_rate_limit),
    redis: Redis = Depends(get_redis_dep),
    ollama: OllamaClient = Depends(get_ollama_dep),
    sessions: SessionStore = Depends(get_session_store_dep),
) -> InferenceResponse:
    """Run a single non-streaming inference request."""
    model = await _resolve_active_model(redis, payload.model)
    session_id = payload.session_id or str(uuid.uuid4())

    history = (
        await sessions.get_history(session_id) if payload.session_id else []
    )
    full_prompt = SessionStore.build_prompt(history, payload.prompt)

    started = time.perf_counter()
    active_inference_requests.inc()
    try:
        result = await ollama.generate(
            model=model, prompt=full_prompt, max_tokens=payload.max_tokens
        )
    except CircuitOpenError as e:
        inference_requests_total.labels(
            model=model, status="circuit_open", tier=principal.tier
        ).inc()
        logger.warning("inference_circuit_open", extra={"model": model})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model backend temporarily unavailable (circuit open)",
        ) from e
    except OllamaTimeoutError as e:
        inference_requests_total.labels(
            model=model, status="timeout", tier=principal.tier
        ).inc()
        logger.warning("inference_timeout", extra={"model": model})
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except OllamaError as e:
        inference_requests_total.labels(
            model=model, status="error", tier=principal.tier
        ).inc()
        logger.exception("inference_backend_error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    finally:
        active_inference_requests.dec()

    elapsed = time.perf_counter() - started
    inference_latency_seconds.labels(model=model).observe(elapsed)
    inference_requests_total.labels(
        model=model, status="success", tier=principal.tier
    ).inc()
    inference_tokens_generated_total.labels(model=model).inc(
        result.tokens_generated
    )

    if payload.session_id:
        await sessions.append_turn(
            session_id, Turn(user=payload.prompt, assistant=result.text)
        )

    logger.info(
        "inference_completed",
        extra={
            "model": model,
            "tier": principal.tier,
            "owner": principal.owner,
            "latency_ms": int(elapsed * 1000),
            "tokens": result.tokens_generated,
            "session_id": session_id,
        },
    )

    return InferenceResponse(
        response=result.text,
        model=result.model,
        latency_ms=int(elapsed * 1000),
        tokens_generated=result.tokens_generated,
        session_id=session_id,
    )


@router.post("/inference/stream")
async def inference_stream(
    payload: InferenceRequest,
    request: Request,
    principal: ApiKeyPrincipal = Depends(enforce_rate_limit),
    redis: Redis = Depends(get_redis_dep),
    ollama: OllamaClient = Depends(get_ollama_dep),
    sessions: SessionStore = Depends(get_session_store_dep),
) -> EventSourceResponse:
    """Stream tokens back to the client over Server-Sent Events.

    Each ``data:`` event is a JSON object: ``{"token": "...", "done": false}``.
    The final event carries ``done: true`` plus the full ``session_id`` and
    aggregate ``tokens_generated``.
    """
    model = await _resolve_active_model(redis, payload.model)
    session_id = payload.session_id or str(uuid.uuid4())
    history = (
        await sessions.get_history(session_id) if payload.session_id else []
    )
    full_prompt = SessionStore.build_prompt(history, payload.prompt)

    async def event_gen() -> AsyncIterator[dict[str, str]]:
        started = time.perf_counter()
        active_inference_requests.inc()
        collected: list[str] = []
        tokens = 0
        try:
            async for chunk in ollama.generate_stream(
                model=model, prompt=full_prompt, max_tokens=payload.max_tokens
            ):
                # Bail early if the client went away.
                if await request.is_disconnected():
                    logger.info(
                        "inference_stream_client_gone",
                        extra={"session_id": session_id},
                    )
                    return
                token_text = chunk.get("response", "")
                if token_text:
                    collected.append(token_text)
                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {"token": token_text, "done": False}
                        ),
                    }
                if chunk.get("done"):
                    tokens = int(chunk.get("eval_count", 0))
                    break
        except CircuitOpenError:
            inference_requests_total.labels(
                model=model, status="circuit_open", tier=principal.tier
            ).inc()
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "circuit_open", "message": "backend unavailable"}
                ),
            }
            return
        except OllamaTimeoutError:
            inference_requests_total.labels(
                model=model, status="timeout", tier=principal.tier
            ).inc()
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": "timeout", "message": "backend timed out"}
                ),
            }
            return
        except OllamaError as e:
            inference_requests_total.labels(
                model=model, status="error", tier=principal.tier
            ).inc()
            yield {
                "event": "error",
                "data": json.dumps({"error": "backend_error", "message": str(e)}),
            }
            return
        finally:
            active_inference_requests.dec()

        elapsed = time.perf_counter() - started
        inference_latency_seconds.labels(model=model).observe(elapsed)
        inference_requests_total.labels(
            model=model, status="success", tier=principal.tier
        ).inc()
        inference_tokens_generated_total.labels(model=model).inc(tokens)

        full_text = "".join(collected)
        if payload.session_id:
            await sessions.append_turn(
                session_id, Turn(user=payload.prompt, assistant=full_text)
            )

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "done": True,
                    "session_id": session_id,
                    "tokens_generated": tokens,
                    "latency_ms": int(elapsed * 1000),
                }
            ),
        }

    return EventSourceResponse(event_gen())
