"""Redis-backed token-bucket rate limiter.

Each API key gets its own bucket keyed under ``wukong:ratelimit:<key>``.
We model the bucket with two Redis fields — ``tokens`` (current count) and
``ts`` (last refill, monotonic-ish unix time) — and refill lazily on each
check. The whole operation runs inside a small Lua script so the read,
update and write happen atomically; otherwise two concurrent requests could
both see "1 token left" and both succeed.

State lives entirely in Redis so the limit is correctly enforced across
multiple application replicas and survives restarts.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from app.config import get_settings
from app.deps import get_redis_dep
from app.middleware.auth import ApiKeyPrincipal, require_api_key
from app.observability.metrics import rate_limit_rejections_total

# Lua: refill bucket up to capacity, try to consume 1 token, return
# (allowed, remaining, retry_after_ms).
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
    tokens = capacity
    ts = now_ms
end

local delta_ms = math.max(0, now_ms - ts)
local refill = (delta_ms / 1000.0) * refill_per_sec
tokens = math.min(capacity, tokens + refill)

local allowed = 0
local retry_after_ms = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
else
    retry_after_ms = math.ceil(((1 - tokens) / refill_per_sec) * 1000)
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, ttl)
return {allowed, tostring(tokens), retry_after_ms}
"""


@dataclass(frozen=True)
class _LimitConfig:
    capacity: int
    refill_per_sec: float


def _limits_for(tier: str) -> _LimitConfig:
    s = get_settings()
    rpm = s.rate_limit_pro_rpm if tier == "pro" else s.rate_limit_free_rpm
    return _LimitConfig(capacity=rpm, refill_per_sec=rpm / 60.0)


async def _consume(redis: Redis, principal: ApiKeyPrincipal) -> tuple[bool, int]:
    """Try to consume one token. Returns (allowed, retry_after_seconds)."""
    cfg = _limits_for(principal.tier)
    key = f"{get_settings().key_prefix_rate_limit}{principal.key}"
    # Bucket TTL slightly larger than the time it takes to fully refill.
    ttl_ms = int(((cfg.capacity / cfg.refill_per_sec) * 1000) * 2) + 1000

    import time as _t

    now_ms = int(_t.time() * 1000)
    allowed, _tokens, retry_ms = await redis.eval(
        _TOKEN_BUCKET_LUA,
        1,
        key,
        cfg.capacity,
        cfg.refill_per_sec,
        now_ms,
        ttl_ms,
    )
    return bool(int(allowed)), max(1, int(retry_ms) // 1000) if not int(allowed) else 0


async def enforce_rate_limit(
    request: Request,
    principal: ApiKeyPrincipal = Depends(require_api_key),
    redis: Redis = Depends(get_redis_dep),
) -> ApiKeyPrincipal:
    """FastAPI dependency: enforce the per-key rate limit.

    Returns the authenticated principal on success so the route can reuse it
    without needing a second ``Depends(require_api_key)``.
    """
    allowed, retry_after = await _consume(redis, principal)
    if not allowed:
        rate_limit_rejections_total.labels(tier=principal.tier).inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for tier '{principal.tier}'",
            headers={"Retry-After": str(retry_after)},
        )
    # Stash on request.state so downstream code can log/observe it.
    request.state.principal = principal
    return principal
