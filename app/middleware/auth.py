"""API key authentication.

API keys are opaque tokens stored in Redis under ``wukong:apikey:<key>``.
Each entry is a hash with fields ``owner``, ``tier``, and ``created_at``.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Depends, Header, HTTPException, status
from redis.asyncio import Redis

from app.config import get_settings
from app.deps import get_redis_dep


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Represents an authenticated caller resolved from an API key."""

    key: str
    owner: str
    tier: str  # "free" | "pro"
    created_at: str


def _key_redis_path(key: str) -> str:
    return f"{get_settings().key_prefix_api_key}{key}"


async def create_api_key(redis: Redis, owner: str, tier: str) -> ApiKeyPrincipal:
    """Generate a new API key, persist it, and return the principal."""
    if tier not in {"free", "pro"}:
        raise ValueError("tier must be 'free' or 'pro'")
    key = f"wk_{secrets.token_urlsafe(24)}"
    principal = ApiKeyPrincipal(
        key=key,
        owner=owner,
        tier=tier,
        created_at=datetime.now(UTC).isoformat(),
    )
    await redis.set(
        _key_redis_path(key),
        json.dumps(
            {
                "owner": principal.owner,
                "tier": principal.tier,
                "created_at": principal.created_at,
            }
        ),
    )
    return principal


async def revoke_api_key(redis: Redis, key: str) -> bool:
    """Delete an API key. Returns True if a key was actually removed."""
    deleted = await redis.delete(_key_redis_path(key))
    return bool(deleted)


async def lookup_api_key(redis: Redis, key: str) -> ApiKeyPrincipal | None:
    """Resolve a raw API key to a principal, or None if unknown."""
    raw = await redis.get(_key_redis_path(key))
    if raw is None:
        return None
    data = json.loads(raw)
    return ApiKeyPrincipal(
        key=key,
        owner=data["owner"],
        tier=data["tier"],
        created_at=data["created_at"],
    )


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1]


async def require_api_key(
    authorization: str | None = Header(default=None),
    redis: Redis = Depends(get_redis_dep),
) -> ApiKeyPrincipal:
    """FastAPI dependency: resolve and validate the Bearer API key."""
    raw = _extract_bearer(authorization)
    principal = await lookup_api_key(redis, raw)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


async def require_admin(
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret"),
) -> None:
    """FastAPI dependency: validate the admin shared secret."""
    expected = get_settings().admin_secret
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin secret",
        )
