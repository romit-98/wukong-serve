"""Redis-backed conversation session store.

Sessions are simple ring buffers of (user, assistant) turn pairs, capped at
``SESSION_MAX_TURNS`` and expiring after ``SESSION_TTL_SECONDS`` of idleness.
We keep the schema deliberately tiny — every turn is a JSON line pushed to
a Redis list — so a future replica can resume mid-conversation without any
in-memory state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import get_settings


@dataclass
class Turn:
    """One conversational turn: a user message and the assistant reply."""

    user: str
    assistant: str

    def to_json(self) -> str:
        return json.dumps({"user": self.user, "assistant": self.assistant})

    @classmethod
    def from_json(cls, raw: str) -> Turn:
        d = json.loads(raw)
        return cls(user=d["user"], assistant=d["assistant"])


class SessionStore:
    """High-level API over Redis lists for session history."""

    def __init__(self, redis: Redis, ttl_seconds: int, max_turns: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._max_turns = max_turns

    def _key(self, session_id: str) -> str:
        return f"{get_settings().key_prefix_session}{session_id}"

    async def get_history(self, session_id: str) -> list[Turn]:
        """Return all stored turns for ``session_id`` in chronological order."""
        raw = await self._redis.lrange(self._key(session_id), 0, -1)
        return [Turn.from_json(r) for r in raw]

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        """Append a turn, trim to max length, and refresh the TTL."""
        key = self._key(session_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, turn.to_json())
        pipe.ltrim(key, -self._max_turns, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

    @staticmethod
    def build_prompt(history: list[Turn], new_user_message: str) -> str:
        """Render history + new user message into a single prompt string.

        Format is intentionally model-agnostic ("User:" / "Assistant:") so it
        works with any chat-tuned model behind Ollama without needing per-model
        templates.
        """
        if not history:
            return new_user_message
        parts: list[str] = []
        for turn in history:
            parts.append(f"User: {turn.user}")
            parts.append(f"Assistant: {turn.assistant}")
        parts.append(f"User: {new_user_message}")
        parts.append("Assistant:")
        return "\n".join(parts)
