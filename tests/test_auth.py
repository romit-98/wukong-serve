"""Tests for API key auth and the admin key endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_admin_requires_secret(client):
    r = await client.post(
        "/admin/keys", json={"owner": "x", "tier": "free"}
    )
    assert r.status_code == 403


async def test_admin_rejects_wrong_secret(client):
    r = await client.post(
        "/admin/keys",
        json={"owner": "x", "tier": "free"},
        headers={"X-Admin-Secret": "nope"},
    )
    assert r.status_code == 403


async def test_create_and_use_key(client, free_key):
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi", "max_tokens": 16},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "hello world"
    assert body["tokens_generated"] == 7


async def test_inference_requires_bearer(client):
    r = await client.post("/v1/inference", json={"prompt": "hi"})
    assert r.status_code == 401


async def test_inference_rejects_bad_format(client):
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": "Token abc"},
    )
    assert r.status_code == 401


async def test_inference_rejects_unknown_key(client):
    r = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": "Bearer wk_does_not_exist"},
    )
    assert r.status_code == 401


async def test_revoke_key(client, free_key):
    r = await client.delete(
        f"/admin/keys/{free_key}",
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    assert r.status_code == 204

    r2 = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert r2.status_code == 401


async def test_revoke_missing_key_is_404(client):
    r = await client.delete(
        "/admin/keys/wk_nonexistent",
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    assert r.status_code == 404
