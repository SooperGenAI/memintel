---
id: self-hosting
title: Self-Hosting
sidebar_label: Self-Hosting
---

# Self-Hosting Memintel

Memintel is open source and designed to be self-hosted. This page covers everything you need to run Memintel on your own infrastructure — from provisioning dependencies through to verifying a working deployment.

:::note
The hosted cloud version is the fastest way to get started. Self-hosting is for teams that need data residency, private infrastructure, or custom deployment configurations.
:::

---

## Architecture

Memintel has a minimal production footprint. There is no message queue, no scheduler, and no object storage dependency.

```
Your Application
      ↓
Memintel Backend  (Python / FastAPI)
      ↓           ↓
PostgreSQL 15+   Redis 7+
      ↑
LLM Provider (Anthropic / AWS Bedrock / Google Vertex AI)
```

**Three dependencies. One process. No scheduler.**

The application layer owns scheduling — your application calls `POST /execute/full` on whatever cadence it needs. Memintel does not include a built-in scheduler.

---

## Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| Python | 3.11+ | Required to run the backend |
| PostgreSQL | 15+ | Primary database. Cloud-hosted recommended |
| Redis | 7+ | Cache layer. Cloud-hosted recommended |
| Node.js | 18+ | Required for TypeScript SDK only — not needed for the backend |
| Git | Any recent | To clone the repository |
| LLM Provider API Key | — | Anthropic, AWS Bedrock, or Google Vertex AI |

---

## Step 1 — Provision Dependencies

### PostgreSQL

Set up a PostgreSQL 15+ instance. Recommended cloud providers:

| Provider | Notes | Free Tier |
|---|---|---|
| [Supabase](https://supabase.com) | Easiest setup. Managed PostgreSQL with dashboard | Yes — 500MB |
| [Neon](https://neon.tech) | Serverless PostgreSQL. Good for variable workloads | Yes — 3GB |
| [AWS RDS](https://aws.amazon.com/rds/postgresql/) | Best for production at scale | No |
| [Railway](https://railway.app) | Simple setup. Good for development and staging | Yes — limited |

Once provisioned, your connection string will look like:

```
postgresql://username:password@host:5432/memintel
```

:::warning
Never put credentials in any config file committed to Git. Always use environment variables.
:::

### Redis

Set up a Redis 7+ instance. Recommended providers:

| Provider | Notes | Free Tier |
|---|---|---|
| [Upstash](https://upstash.com) | Serverless Redis. Easiest setup. Pay per request | Yes — 10,000 req/day |
| [Redis Cloud](https://redis.com/cloud) | Official Redis hosting | Yes — 30MB |
| [AWS ElastiCache](https://aws.amazon.com/elasticache/) | Best for production at scale | No |
| [Railway](https://railway.app) | Simple setup alongside the backend | Yes — limited |

### LLM Provider

Memintel supports three LLM providers:

| Provider | Environment Variable | Notes |
|---|---|---|
| Anthropic (direct) | `ANTHROPIC_API_KEY` | Simplest setup. Recommended for getting started |
| AWS Bedrock | AWS credentials + region | Best for AWS-native deployments |
| Google Vertex AI | GCP credentials + project | Best for GCP-native deployments |

Get your Anthropic API key from [console.anthropic.com](https://console.anthropic.com). Store it as an environment variable — never in a file.

### Backend Hosting

The FastAPI backend runs on any platform that supports Python:

| Platform | Best For | Notes |
|---|---|---|
| [Railway](https://railway.app) | Quickest deployment | Supports env vars, auto-deploy from GitHub |
| [Render](https://render.com) | Simple managed hosting | Free tier available. Auto-deploy from GitHub |
| [AWS EC2 / ECS](https://aws.amazon.com/ecs/) | Full control at scale | More configuration required |
| [Google Cloud Run](https://cloud.google.com/run) | Serverless containers | Good for variable workloads |

---

## Step 2 — Configure Environment Variables

:::warning
The server will refuse to start if any required variable is missing. This is by design.
:::

### Required Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | **Yes** | — | Full PostgreSQL connection string including credentials |
| `REDIS_URL` | **Yes** | — | Full Redis connection string including credentials |
| `MEMINTEL_CONFIG_PATH` | **Yes** | — | Absolute path to `memintel_config.yaml` on the server |

### Optional Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No* | — | Required when `USE_LLM_FIXTURES=false` and using Anthropic |
| `MEMINTEL_ELEVATED_KEY` | No | — | Admin key for registry global namespace promotion. Use a strong random string |
| `USE_LLM_FIXTURES` | No | `true` | Set to `false` in production to use a real LLM provider |
| `MAX_RETRIES` | No | `3` | LLM refinement loop retries before failing |
| `DB_POOL_MIN` | No | `5` | Minimum asyncpg connection pool size |
| `DB_POOL_MAX` | No | `20` | Maximum asyncpg connection pool size |
| `DB_COMMAND_TIMEOUT` | No | `30` | Per-statement database timeout in seconds |

:::tip
`ANTHROPIC_API_KEY` is only required when `USE_LLM_FIXTURES=false` and using Anthropic as the provider. During initial setup and smoke testing, leave `USE_LLM_FIXTURES=true` — this validates the full pipeline without consuming LLM credits.
:::

:::note
If your `memintel_config.yaml` references connector environment variables (for example `ANALYTICS_DB_HOST`, `ANALYTICS_DB_USER`, `ANALYTICS_DB_PASSWORD`, `ACCOUNTS_DB_HOST`, `ACCOUNTS_DB_USER`, `ACCOUNTS_DB_PASSWORD`), those must also be set before starting the server. The server will exit with a `ConfigError` naming the missing variable if any are absent.
:::

### Setting Variables

On Linux / Mac:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/memintel"
export REDIS_URL="redis://host:6379/0"
export MEMINTEL_CONFIG_PATH="/etc/memintel/memintel_config.yaml"
export USE_LLM_FIXTURES="false"
export ANTHROPIC_API_KEY="sk-ant-..."
```

On Railway, Render, or other cloud platforms: set these in the platform's environment variables dashboard. Never put them in a `.env` file committed to Git.

---

## Step 3 — Install and Run

### Clone the Repository

```bash
git clone https://github.com/SooperGenAI/memintel.git
cd memintel/backend/memintel-backend
```

### Install Python Dependencies

Memintel requires Python 3.11. If you have a newer Python version installed, create a virtual environment explicitly using 3.11:

```bash
python3.11 -m venv venv
```

Activate it:

```bash
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Verify key packages are installed:

```bash
python -c "import fastapi, pydantic, asyncpg, aioredis; print('OK')"
```

### Run Unit Tests

```bash
python -m pytest tests/unit/ -q --tb=short
```

All unit tests should pass before proceeding.

---

## Step 4 — Run Database Migrations

With `DATABASE_URL` set, run the Alembic migrations:

```bash
alembic upgrade head
```

Expected output:

```
INFO [alembic.runtime.migration] Running upgrade -> 0001, initial_schema
INFO [alembic.runtime.migration] Running upgrade 0001 -> 0002, add_application_context
INFO [alembic.runtime.migration] Running upgrade 0002 -> 0003, add_guardrails_versions
```

:::warning
If you see any error during migration, do not proceed. Fix the migration error before starting the server.
:::

:::warning
If alembic reports a driver error such as `"scheme is expected to be either postgresql or postgres"`, temporarily set `DATABASE_URL` with the asyncpg prefix for the migration command only:

```bash
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/memintel alembic upgrade head
```

On Windows PowerShell:

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://user:password@host:5432/memintel"
alembic upgrade head
```

Then revert `DATABASE_URL` back to `postgresql://` before starting the server.
:::

Verify the schema — you should see all seven tables:

```bash
psql $DATABASE_URL -c "\dt"
```

| Table | Purpose |
|---|---|
| `tasks` | Task definitions with version pinning. Carries three provenance fields: `context_version`, `guardrails_version`, `context_warning` |
| `definitions` | Concept, primitive, and condition definitions |
| `feedback` | User feedback for calibration |
| `calibration_tokens` | Single-use calibration tokens |
| `execution_graphs` | Compiled execution graphs (IR) |
| `jobs` | Async job tracking |
| `application_context` | Application context versions |
| `guardrails_versions` | Guardrails versions — created by Migration 0003. Stores all API-posted guardrails with full audit trail |

---

## Step 5 — Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For production with multiple workers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Startup

The server logs should show:

```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
```

:::warning
If startup fails with `ConfigError`, check that all required environment variables are set and `MEMINTEL_CONFIG_PATH` points to a valid file.
:::

### Health Check

```bash
curl http://localhost:8000/health
```

If the health endpoint returns `Not Found`, verify the server is running by opening the Swagger API docs in your browser:

```
http://localhost:8000/docs
```

A working deployment will show the full Memintel API reference. If the page loads, the server is running correctly.

:::warning
If `database` or `cache` shows `"disconnected"`, verify `DATABASE_URL` and `REDIS_URL` are correct and the services are reachable from the server.
:::

---

## Step 6 — Smoke Test

Run these tests with `USE_LLM_FIXTURES=true` before switching to a real LLM. This validates the full pipeline against your real database and Redis without consuming LLM credits.

### Define Application Context

```bash
curl -X POST http://localhost:8000/context \
  -H "Content-Type: application/json" \
  -d '{
    "domain": {
      "description": "Smoke test deployment",
      "entities": [{"name": "user", "description": "test user"}],
      "decisions": ["test_decision"]
    }
  }'
```

Expected: HTTP `201` with `context_id` and `"version": "v1"`

### Create a Task

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "Alert when churn score exceeds 0.8",
    "entity_scope": "user",
    "delivery": {"type": "webhook", "endpoint": "https://example.com/hook"}
  }'
```

Expected: HTTP `200` with `task_id`, `condition_id`, `"context_version": "v1"`, `"context_warning": null`

### Execute Full Pipeline

```bash
curl -X POST http://localhost:8000/execute/full \
  -H "Content-Type: application/json" \
  -d '{
    "concept_id": "<concept_id from task>",
    "concept_version": "<concept_version from task>",
    "condition_id": "<condition_id from task>",
    "condition_version": "<condition_version from task>",
    "entity": "user_test_001",
    "timestamp": "2024-01-15T09:00:00Z"
  }'
```

Expected: HTTP `200`, `result.deterministic: true`

### Verify Determinism

Run the same execute call three times with identical parameters. All three responses must return the same `result.value` and `decision.value`:

```bash
for i in 1 2 3; do
  curl -s -X POST http://localhost:8000/execute/full \
    -H 'Content-Type: application/json' \
    -d '{...same payload...}' | jq '.result.value'
done
```

If the values diverge, stop and investigate before going to production.

---

## Step 7 — Switch to Real LLM

Once all smoke tests pass against the real database:

```bash
export USE_LLM_FIXTURES=false
export ANTHROPIC_API_KEY="sk-ant-..."
```

Restart the server after changing these variables, then repeat the smoke test with real LLM. This will consume LLM credits — run once to verify, not repeatedly.

Test one task creation for each strategy type to confirm the LLM is resolving correctly within guardrails:

| Strategy | Test Intent |
|---|---|
| `threshold` | `"Alert when churn score exceeds 0.8"` |
| `z_score` | `"Alert when login frequency drops significantly"` |
| `equals` | `"Alert when user segment is high_risk"` |
| `composite` | `"Alert when churn is high AND engagement is low"` |

---

## Step 8 — Post-Deployment Setup

### Define Application Context

The first action after deployment is verified — define your application context. This significantly improves task creation accuracy from the first real user request.

```bash
curl -X POST https://your-domain/context \
  -H "Content-Type: application/json" \
  -d @your-context-file.json
```

See the [Application Context](/docs/admin-guide/admin-application-context) page in the Admin Guide for domain-specific examples and full schema documentation.

### Define Guardrails (Recommended)

Post your guardrails policy via API. This takes effect immediately without a restart:

```bash
curl -X POST https://your-domain/guardrails \
  -H "Content-Type: application/json" \
  -H "X-Elevated-Key: your-elevated-key" \
  -d @guardrails.json
```

:::tip
Using `POST /guardrails` is preferred over editing `memintel_guardrails.yaml` — changes take effect immediately with no restart required, and every change is versioned. See the [Admin Guide](/docs/admin-guide/admin-guardrails-api) for full guardrails schema and domain-specific examples.
:::

### Monitoring

Memintel emits structured logs with `trace_id`, `entity`, `concept_id`, and `condition_id` on every execution. Configure your log aggregation tool to index on these fields:

- Index `trace_id` for request tracing
- Alert on `execution_error` log entries
- Dashboard: execution duration p50 / p95 / p99
- Dashboard: condition evaluation true/false ratio per `condition_id`

### First Week

| Day | Action |
|---|---|
| Day 1 | Define application context (`POST /context`). Post guardrails (`POST /guardrails`). Create first task. |
| Day 2–3 | Run executions. Collect initial feedback. Verify feedback is being recorded. |
| Day 4–5 | Run first calibration cycle. Review `statistically_optimal` vs `context_adjusted` values. |
| Day 7 | Review execution logs. Check for unexpected errors. Verify determinism is holding. |

---

## Estimated Deployment Time

| Phase | Estimated Time |
|---|---|
| Infrastructure setup | 30–60 minutes |
| Environment configuration | 15 minutes |
| Database migration | 5 minutes |
| Service startup and verification | 15 minutes |
| Smoke testing (fixtures mode) | 20 minutes |
| Real LLM integration test | 30 minutes |
| **Total** | **~2 hours** |

---

## Known Issues

### aioredis compatibility with Python 3.11

aioredis has a known incompatibility with Python 3.11 that causes this error on startup:

```
TypeError: duplicate base class TimeoutError
```

Fix it by patching the installed aioredis exceptions file after `pip install` completes.

**On Mac/Linux:**

```bash
sed -i 's/class TimeoutError(asyncio.TimeoutError, builtins.TimeoutError, RedisError):/class TimeoutError(asyncio.TimeoutError, RedisError):/' venv/lib/python3.11/site-packages/aioredis/exceptions.py
```

**On Windows PowerShell:**

```powershell
(Get-Content venv\Lib\site-packages\aioredis\exceptions.py) -replace 'class TimeoutError\(asyncio\.TimeoutError, builtins\.TimeoutError, RedisError\):', 'class TimeoutError(asyncio.TimeoutError, RedisError):' | Set-Content venv\Lib\site-packages\aioredis\exceptions.py
```

This patch is applied to your local venv only and does not affect the source code. It will need to be reapplied if you recreate the venv.

---

## Further Reading

- [Application Context](/docs/intro/application-context) — configure domain context for more accurate task compilation
- [Quickstart](/docs/intro/quickstart) — your first task in 5 minutes
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
- [GitHub Repository](https://github.com/SooperGenAI/memintel) — source code, issues, and discussions
 
 <!-- deploy: 2026-03-28 18:42:22 -->
