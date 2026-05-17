"""Locust load test scenarios for Wukong-serve.

Run with:

    locust -f load_tests/locustfile.py --host http://localhost:8000

Three user classes are defined:

* ``FreeUser`` — paces requests below the free-tier RPM and expects occasional
  429s when bursts coincide.
* ``ProUser`` — paces above the free limit but below the pro limit, expects
  consistent 200s.
* ``MixedUser`` — used with ``--users 100 --spawn-rate 1`` and a custom
  ramp shape to grow load from 10 → 100 users over 2 minutes and find where
  latency starts to degrade.

Each user mints its own API key against the admin endpoint at start, using
the ADMIN_SECRET env var (default ``changeme``). This means a clean test
run only needs the stack to be up — no manual key setup.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, LoadTestShape, between, events, task

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "changeme")

PROMPTS = [
    "Explain quantum entanglement in one paragraph.",
    "Write a haiku about distributed systems.",
    "What are three causes of high tail latency in microservices?",
    "Summarise the CAP theorem for a junior engineer.",
    "Give me a one-line definition of a circuit breaker.",
]


def _mint_key(client, tier: str, owner: str) -> str:
    r = client.post(
        "/admin/keys",
        json={"owner": owner, "tier": tier},
        headers={"X-Admin-Secret": ADMIN_SECRET},
        name="/admin/keys",
    )
    r.raise_for_status()
    return r.json()["key"]


class _BaseUser(HttpUser):
    """Shared behaviour: mint a key on start and use it for inference calls."""

    abstract = True
    tier = "free"

    def on_start(self) -> None:
        owner = f"locust-{self.tier}-{random.randint(1000, 9999)}"
        self.api_key = _mint_key(self.client, self.tier, owner)

    def _inference(self) -> None:
        with self.client.post(
            "/v1/inference",
            json={
                "prompt": random.choice(PROMPTS),
                "max_tokens": 128,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            name="/v1/inference",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                # Expected for the free-tier scenario; mark as success so the
                # report shows the rejection without inflating failure counts.
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}: {resp.text[:200]}")


class FreeUser(_BaseUser):
    """Slow free-tier caller. Expects to hit 429s occasionally."""

    tier = "free"
    wait_time = between(2, 5)

    @task
    def inference(self) -> None:
        self._inference()


class ProUser(_BaseUser):
    """Faster pro-tier caller. Should consistently get 200s."""

    tier = "pro"
    wait_time = between(0.5, 1.5)

    @task
    def inference(self) -> None:
        self._inference()


class MixedUser(_BaseUser):
    """Used with the ramp shape below."""

    tier = "pro"
    wait_time = between(0.2, 1.0)

    @task
    def inference(self) -> None:
        self._inference()


class RampTo100(LoadTestShape):
    """Ramp from 10 to 100 users over 120 seconds, then hold for 60s.

    Invoke with::

        locust -f load_tests/locustfile.py --host http://localhost:8000 \\
            --class-picker MixedUser --headless
    """

    stages = [
        {"duration": 30, "users": 10, "spawn_rate": 2},
        {"duration": 60, "users": 40, "spawn_rate": 4},
        {"duration": 120, "users": 100, "spawn_rate": 5},
        {"duration": 180, "users": 100, "spawn_rate": 5},
    ]

    def tick(self):  # type: ignore[no-untyped-def]
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None


@events.quitting.add_listener
def _print_summary(environment, **_kwargs):  # type: ignore[no-untyped-def]
    """Print a compact end-of-run summary to stdout."""
    stats = environment.stats.total
    print("\n=== Wukong-serve load test summary ===")
    print(f"Total requests : {stats.num_requests}")
    print(f"Failures       : {stats.num_failures}")
    print(f"Median (ms)    : {stats.median_response_time}")
    print(f"p95 (ms)       : {stats.get_response_time_percentile(0.95)}")
    print(f"p99 (ms)       : {stats.get_response_time_percentile(0.99)}")
    print(f"RPS            : {stats.total_rps:.2f}")
