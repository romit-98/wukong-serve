"""End-to-end behaviour of the inference endpoints."""

from __future__ import annotations

import json

import pytest

from app.services.ollama_client import (
    CircuitOpenError,
    OllamaError,
    OllamaTimeoutError,
)

pytestmark = pytest.mark.asyncio


async def test_inference_returns_structured_response(client, free_key, fake_ollama):
    fake_ollama.response_text = "the answer is 42"
    fake_ollama.tokens = 5
    r = await client.post(
        "/v1/inference",
        json={"prompt": "what is the answer?", "max_tokens": 64},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "the answer is 42"
    assert body["tokens_generated"] == 5
    assert body["model"] == "test-model"
    assert body["latency_ms"] >= 0
    assert body["session_id"]  # auto-generated when not provided


async def test_inference_uses_session_history(client, free_key, fake_ollama):
    sid = "sess-abc"
    await client.post(
        "/v1/inference",
        json={"prompt": "my name is Romit", "session_id": sid},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    fake_ollama.response_text = "you said Romit"
    r = await client.post(
        "/v1/inference",
        json={"prompt": "what is my name?", "session_id": sid},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 200
    # The second call must have received the first turn in its prompt.
    last_prompt = fake_ollama.calls[-1]["prompt"]
    assert "my name is Romit" in last_prompt
    assert "what is my name?" in last_prompt


async def test_inference_timeout_returns_504(client, free_key, fake_ollama):
    fake_ollama.raise_error = OllamaTimeoutError("slow")
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 504


async def test_inference_circuit_open_returns_503(client, free_key, fake_ollama):
    fake_ollama.raise_error = CircuitOpenError("open")
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 503


async def test_inference_backend_error_returns_502(client, free_key, fake_ollama):
    fake_ollama.raise_error = OllamaError("boom")
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 502


async def test_admin_model_switch_changes_active_model(client, free_key, fake_ollama):
    r = await client.post(
        "/admin/model",
        json={"model": "llama3.2:7b"},
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    assert r.status_code == 200
    assert r.json()["active_model"] == "llama3.2:7b"

    await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert fake_ollama.calls[-1]["model"] == "llama3.2:7b"


async def test_streaming_yields_tokens_and_done_event(client, free_key, fake_ollama):
    fake_ollama.stream_chunks = ["foo ", "bar ", "baz"]
    fake_ollama.tokens = 3
    async with client.stream(
        "POST",
        "/v1/inference/stream",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    ) as r:
        assert r.status_code == 200
        events: list[tuple[str, dict]] = []
        current_event = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event:
                payload = json.loads(line.split(":", 1)[1].strip())
                events.append((current_event, payload))
                current_event = None

    token_events = [p for ev, p in events if ev == "token"]
    done_events = [p for ev, p in events if ev == "done"]
    assert [p["token"] for p in token_events] == ["foo ", "bar ", "baz"]
    assert len(done_events) == 1
    assert done_events[0]["done"] is True
    assert done_events[0]["tokens_generated"] == 3


async def test_health_reports_ok_when_deps_healthy(client, fake_ollama):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    names = {c["name"] for c in body["components"]}
    assert {"redis", "ollama"} <= names


async def test_health_reports_degraded_when_ollama_down(client, fake_ollama):
    fake_ollama.healthy = False
    r = await client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"


async def test_metrics_endpoint_exposes_prometheus_format(client, free_key):
    # Hit the inference endpoint once to register a metric.
    await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "inference_requests_total" in body
    assert "inference_latency_seconds" in body
