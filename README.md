# Wukong-serve

A production-realistic LLM inference serving system. Wukong sits between an
Ollama model backend and the outside world and takes care of everything that
isn't the model itself: authentication, rate limiting, session state,
back-pressure, failure isolation, metrics, and dashboards.

The project is deliberately scoped to the *serving infrastructure* layer.
There is no prompt engineering, no fine-tuning, no agent loop — just the
plumbing that turns a single-tenant `ollama serve` process into something
you would be willing to put behind a public API.

## What each component exists for

| Component | Why it exists |
|---|---|
| **FastAPI** app | HTTP surface area: inference, admin, health, metrics. Async-first so a single worker can hold many concurrent streams without thread blow-up. |
| **Ollama** (`llama3.2:3b`) | The actual model backend. Treated as a fallible external dependency — the app never assumes it is up. |
| **Redis** | Single source of truth for everything that must survive a process restart: API keys, rate-limit buckets, conversation history, the active-model selector. |
| **Circuit breaker** | Stops the app from hammering a wedged backend. Opens after 5 consecutive failures, cools down for 30s, then probes. |
| **Token-bucket rate limiter** | Per-key fair-share enforcement. Lives in Redis so multiple replicas share state and limits survive restarts. |
| **Prometheus** | Scrapes `/metrics` every 5s. Stores counters, histograms, and gauges for latency, throughput, errors, and breaker state. |
| **Grafana** | Pre-provisioned dashboard so you can *watch* the system instead of `curl`-polling it. |
| **Locust** | Closed-loop load generator with three scenarios (free, pro, ramp-to-100). |
| **GitHub Actions** | Runs the full pytest suite + ruff on every push. Ollama is mocked in CI — only Redis runs as a service container. |

## Architecture

```
                            ┌──────────────────────┐
   client ──HTTP──▶ FastAPI │  /v1/inference       │
                            │  /v1/inference/stream│
                            │  /admin/*  /health   │
                            │  /metrics            │
                            └──────────┬───────────┘
                                       │
            ┌──────────────────────────┼───────────────────────────┐
            │                          │                           │
            ▼                          ▼                           ▼
      ┌──────────┐              ┌──────────────┐           ┌──────────────┐
      │  Redis   │              │   Ollama     │           │  Prometheus  │
      │ keys     │              │ /api/generate│◀── scrape │   metrics    │
      │ buckets  │              └──────────────┘     ▲     │              │
      │ sessions │                                   │     └──────┬───────┘
      │ active   │                                   │            │
      │  model   │                                   │            ▼
      └──────────┘                                   │      ┌──────────┐
                                                     └──────│ Grafana  │
                                                            └──────────┘
```

Inside the app, every Ollama call goes through the **circuit breaker** before
it touches the network. Every authenticated request first hits the **rate
limiter** (atomic Lua in Redis). Session-aware requests read history from
Redis, render a chat-style prompt, and append the new turn after the model
responds. Every request emits exactly one structured JSON log line and
updates the Prometheus metrics.

## Running the full stack locally

You need: Python 3.11+, Docker Desktop, and Ollama installed on the host
(Wukong does not run the model in a container — Ollama is intentionally a
host-installed dependency so the GPU path stays simple).

```powershell
# 1. Ollama (one-time install): https://ollama.com/download
ollama pull llama3.2:3b
ollama serve   # leave this running

# 2. Infra (Redis + Prometheus + Grafana)
docker compose -f infra/docker-compose.yml up -d

# 3. App
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then:

- API: <http://localhost:8000/docs>
- Metrics: <http://localhost:8000/metrics>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (anonymous viewer, or `admin` / `admin`)

Mint a key and call it:

```powershell
$key = (curl -s -X POST http://localhost:8000/admin/keys `
    -H "X-Admin-Secret: changeme" `
    -H "Content-Type: application/json" `
    -d '{"owner":"romit","tier":"pro"}' | ConvertFrom-Json).key

curl -X POST http://localhost:8000/v1/inference `
    -H "Authorization: Bearer $key" `
    -H "Content-Type: application/json" `
    -d '{"prompt":"Write a haiku about Redis."}'
```

## API surface

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/v1/inference` | Bearer key | Non-streaming. Returns full response + latency + token count. |
| POST | `/v1/inference/stream` | Bearer key | SSE; `event: token` per chunk, final `event: done`. |
| GET | `/health` | none | Checks Redis + Ollama. Returns 503 if either is down. |
| GET | `/health/live` | none | Process-only liveness. |
| GET | `/metrics` | none | Prometheus exposition format. |
| POST | `/admin/keys` | admin secret | Mint API key. Returns raw key once. |
| DELETE | `/admin/keys/{key}` | admin secret | Revoke. |
| POST | `/admin/model` | admin secret | Hot-swap the active model. |
| GET | `/admin/model` | admin secret | Report the active model. |

## Load testing

```powershell
# Free-tier behaviour
locust -f load_tests/locustfile.py --host http://localhost:8000 `
    --users 5 --spawn-rate 1 --headless -t 1m FreeUser

# Pro-tier behaviour
locust -f load_tests/locustfile.py --host http://localhost:8000 `
    --users 5 --spawn-rate 1 --headless -t 1m ProUser

# 10 → 100 ramp over 2 minutes (uses the RampTo100 LoadTestShape)
locust -f load_tests/locustfile.py --host http://localhost:8000 `
    --headless MixedUser
```

What to expect:

- **FreeUser**: ~10 successful requests per minute per user, then 429s with a
  `Retry-After` header. The Grafana panel "Rate-limit rejections / sec" lights
  up. No latency degradation — rejections are O(1).
- **ProUser**: consistent 200s at the cap (60rpm). p95 latency tracks the
  model's tokens/sec.
- **MixedUser**: with `llama3.2:3b` on CPU on a laptop, p95 starts climbing
  past ~20 concurrent users because Ollama serialises requests internally.
  This is the interesting failure mode — see below.

### What Broke Under Load

These are the failure modes observed during this project's own bring-up,
not hypotheticals:

1. **Ollama is single-threaded for inference.** Past ~20 concurrent in-flight
   requests on a laptop CPU, Ollama queues internally and end-to-end p95
   latency climbs from ~1.5s to >15s. *Fix:* the 30s request timeout returns
   a clean 504 instead of letting clients hang, and the breaker eventually
   opens if Ollama becomes fully unresponsive. The real production fix is
   horizontal Ollama replicas behind a load balancer with a least-pending
   policy — out of scope here but the metrics make the case for it obvious.

2. **Token-bucket race condition (pre-Lua).** The first version of the
   limiter did a `HGET` / compute / `HSET` round-trip in Python. Under burst
   load two concurrent requests would both see "1 token available" and both
   succeed, briefly letting the limit double. *Fix:* moved the entire
   refill-and-consume into a Lua script (`_TOKEN_BUCKET_LUA`) so it runs as
   one atomic Redis operation.

3. **SSE clients leaked.** Initially the streaming endpoint did not check
   `request.is_disconnected()`, so a client that hung up mid-stream still
   pulled the full response from Ollama. *Fix:* the generator now bails
   between chunks if the client is gone, and decrements
   `active_inference_requests` in a `finally` so the gauge stays accurate.

4. **Lifespan teardown race.** Closing the Redis pool before in-flight
   requests finished caused the last few requests in a graceful shutdown to
   error. *Fix:* relying on uvicorn / FastAPI's lifespan contract — they
   already wait for in-flight requests before invoking the teardown half of
   the lifespan context manager. Confirmed by sending `SIGTERM` mid-stream.

## Observability

Every request emits one structured JSON log line containing `request_id`,
`method`, `path`, `status`, `duration_ms`, and (for inference) `model`,
`tier`, `owner`, `tokens`, `session_id`. Pipe stdout to your aggregator of
choice.

Prometheus metrics:

- `inference_requests_total{model,status,tier}` — counter
- `inference_latency_seconds{model}` — histogram, buckets 0.5/1/2/5/10s
- `inference_tokens_generated_total{model}` — counter
- `active_inference_requests` — gauge
- `rate_limit_rejections_total{tier}` — counter
- `circuit_breaker_state` — gauge (0 closed, 1 open)

### Grafana dashboard

The dashboard at `infra/grafana/dashboard.json` is provisioned on Grafana
startup. It refreshes every 5 seconds and shows requests/sec (by status and
by tier), p50/p95/p99 latency, error rate, active requests, rate-limit
rejections, tokens/min, and circuit-breaker state.

> *(screenshot placeholder — capture after running a load test)*

## Design Decisions

**Why FastAPI + async, not Flask or sync workers?**
LLM inference is dominated by waiting on the model. A sync model would need
a worker per in-flight request, which wastes RAM. With `async`, one worker
can hold hundreds of in-flight requests because they spend almost all their
time awaiting the Ollama socket.

**Why Redis for both rate limits and sessions?**
A single dependency means one fewer thing to operate. Both workloads are
small-payload, low-latency, TTL-driven — exactly what Redis is good at.
The keys are namespaced (`wukong:apikey:`, `wukong:ratelimit:`,
`wukong:session:`) so they cohabit cleanly with other tenants of a shared
Redis.

**Why an in-process circuit breaker, not a shared one in Redis?**
A shared breaker is correct in theory but masks localised network problems —
one replica's connection to Ollama can be broken without the others
noticing. Per-replica state means each replica makes the right call for
itself, and the *aggregate* breaker behaviour is what the dashboard plots.

**Why Lua for the token bucket?**
The naive read-modify-write across two Redis round-trips lets concurrent
requests double-spend the bucket. Lua runs atomically inside Redis, which
is the only correct way to do this without a distributed lock. The script
returns the consume decision plus a retry-after hint in milliseconds in one
call.

**Why SSE for streaming instead of WebSockets?**
SSE is HTTP, so it works through every reverse proxy, firewall, and load
balancer without special configuration. It is also strictly simpler —
server-to-client only, which is all token streaming needs.

**Why mock Ollama in CI instead of running it?**
Ollama needs to download a multi-GB model and is slow to start. CI cares
about correctness of the surrounding code, not the model output. The
mock in `tests/conftest.py` (`FakeOllama`) records every call so tests can
assert on what the app *would* have asked the model — including session
history rendering and model-override behaviour.

## Tests

```powershell
pytest -v
ruff check app tests
```

CI runs both on every push to `main` against a real Redis service container,
with a mocked Ollama.

## Configuration

Every knob is environment-driven; see `.env.example` for the full list and
`app/config.py` for the schema (validated by pydantic-settings). No secrets
are hardcoded.

## License

MIT.
