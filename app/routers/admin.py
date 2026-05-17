"""Admin endpoints: API key management and live model switching."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.config import get_settings
from app.deps import get_redis_dep
from app.middleware.auth import (
    create_api_key,
    require_admin,
    revoke_api_key,
)

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)
logger = logging.getLogger(__name__)


class CreateKeyRequest(BaseModel):
    owner: str = Field(..., min_length=1, max_length=128)
    tier: Literal["free", "pro"] = "free"


class CreateKeyResponse(BaseModel):
    key: str
    owner: str
    tier: str
    created_at: str


class ModelSwitchRequest(BaseModel):
    model: str = Field(..., min_length=1)


class ModelSwitchResponse(BaseModel):
    active_model: str


@router.post(
    "/keys",
    response_model=CreateKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(
    payload: CreateKeyRequest,
    redis: Redis = Depends(get_redis_dep),
) -> CreateKeyResponse:
    """Mint a new API key. The raw key is returned exactly once."""
    principal = await create_api_key(redis, owner=payload.owner, tier=payload.tier)
    logger.info(
        "api_key_created",
        extra={"owner": principal.owner, "tier": principal.tier},
    )
    return CreateKeyResponse(
        key=principal.key,
        owner=principal.owner,
        tier=principal.tier,
        created_at=principal.created_at,
    )


@router.delete("/keys/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_key(key: str, redis: Redis = Depends(get_redis_dep)) -> Response:
    """Revoke an API key."""
    deleted = await revoke_api_key(redis, key)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Key not found"
        )
    logger.info("api_key_revoked", extra={"key_prefix": key[:8]})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/model", response_model=ModelSwitchResponse)
async def switch_model(
    payload: ModelSwitchRequest,
    redis: Redis = Depends(get_redis_dep),
) -> ModelSwitchResponse:
    """Update the server-wide active model. Takes effect on the next request."""
    settings = get_settings()
    await redis.set(settings.key_active_model, payload.model)
    logger.info("active_model_changed", extra={"model": payload.model})
    return ModelSwitchResponse(active_model=payload.model)


@router.get("/model", response_model=ModelSwitchResponse)
async def get_active_model(
    redis: Redis = Depends(get_redis_dep),
) -> ModelSwitchResponse:
    """Report the currently active model."""
    settings = get_settings()
    stored = await redis.get(settings.key_active_model)
    return ModelSwitchResponse(active_model=stored or settings.ollama_default_model)
