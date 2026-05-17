"""In-process circuit breaker.

The breaker is kept in-process (not in Redis) on purpose: each replica needs
its own view of whether *its* connection to the backend is healthy. A shared
Redis-backed breaker would either flap globally or fail to detect localised
network issues.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.observability.metrics import circuit_breaker_state


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the breaker is open."""


@dataclass
class CircuitBreaker:
    """Classic three-state circuit breaker (closed → open → half-open)."""

    failure_threshold: int
    reset_timeout_seconds: float
    _failures: int = 0
    _opened_at: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_open(self) -> bool:
        """Whether the breaker currently rejects calls."""
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout_seconds:
            return False  # cooldown elapsed — allow a probe
        return True

    async def before_call(self) -> None:
        """Raise ``CircuitOpenError`` if calls are currently disallowed."""
        async with self._lock:
            if self.is_open:
                raise CircuitOpenError(
                    "Ollama circuit breaker is open; refusing call."
                )

    async def record_success(self) -> None:
        """Reset failure counter and close the breaker."""
        async with self._lock:
            self._failures = 0
            self._opened_at = None
            circuit_breaker_state.set(0)

    async def record_failure(self) -> None:
        """Increment failure counter; open the breaker if threshold reached."""
        async with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                circuit_breaker_state.set(1)
