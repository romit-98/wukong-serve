"""Rate limiter behaviour under burst load."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_free_tier_limit_enforced(client, free_key):
    """Free tier capacity is RATE_LIMIT_FREE_RPM (3 in tests).

    First 3 calls succeed, the 4th must be 429 with a Retry-After header.
    """
    headers = {"Authorization": f"Bearer {free_key}"}
    for _ in range(3):
        r = await client.post(
            "/v1/inference", json={"prompt": "hi"}, headers=headers
        )
        assert r.status_code == 200

    r = await client.post(
        "/v1/inference", json={"prompt": "hi"}, headers=headers
    )
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1


async def test_pro_tier_has_higher_limit(client, pro_key):
    """Pro tier (10 rpm in tests) must allow at least 4 quick requests."""
    headers = {"Authorization": f"Bearer {pro_key}"}
    successes = 0
    for _ in range(4):
        r = await client.post(
            "/v1/inference", json={"prompt": "hi"}, headers=headers
        )
        if r.status_code == 200:
            successes += 1
    assert successes == 4


async def test_rate_limit_buckets_are_per_key(client, free_key):
    """Hitting one key's limit must not affect another key."""
    r2 = await client.post(
        "/admin/keys",
        json={"owner": "carol", "tier": "free"},
        headers={"X-Admin-Secret": "test-admin-secret"},
    )
    second_key = r2.json()["key"]

    # Exhaust the first key.
    for _ in range(3):
        await client.post(
            "/v1/inference",
            json={"prompt": "hi"},
            headers={"Authorization": f"Bearer {free_key}"},
        )
    blocked = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {free_key}"},
    )
    assert blocked.status_code == 429

    # Second key should be untouched.
    ok = await client.post(
        "/v1/inference",
        json={"prompt": "hi"},
        headers={"Authorization": f"Bearer {second_key}"},
    )
    assert ok.status_code == 200
