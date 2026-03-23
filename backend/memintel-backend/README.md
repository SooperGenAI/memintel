# memintel-backend

Python execution engine for the Memintel platform.

## Responsibilities

- Concept execution (ψ layer) — runs execution graphs against entity data
- Condition evaluation (φ layer) — applies condition strategies to concept results
- Action triggering (α layer) — fires best-effort delivery actions
- Compiler — transforms semantic definitions into deterministic execution graphs (IR)
- Registry — writes, versions, promotes, and deprecates definitions
- Async job queue — accepts long-running executions and surfaces results via polling
- LLM refinement loop — generates and validates concept/condition definitions at task creation time
- Interaction API backend — tasks, conditions, calibration, feedback

## Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Models | Pydantic v2 |
| Async | asyncio + httpx |
| Database | PostgreSQL 15+ via asyncpg |
| Cache | Redis 7+ via aioredis |
| Migrations | Alembic |
| Testing | pytest + pytest-asyncio |

## Structure

```
app/
  api/routes/       — FastAPI route handlers (thin — delegate to services)
  compiler/         — type_checker, dag_builder, ir_generator, validator
  runtime/          — executor, condition_evaluator, action_trigger, data_resolver, cache
  strategies/       — base + six strategy implementations
  registry/         — definition and feature registries
  models/           — Pydantic models (task, condition, concept, result, errors, calibration)
  persistence/      — DB pool, Redis client, store classes
  services/         — task_authoring, calibration, feedback, explanation
  llm/              — LLM client, prompts, fixtures
  config/           — guardrails_store, primitive_registry
  main.py           — FastAPI app, lifespan wiring, router registration

tests/
  unit/             — strategy, compiler, and store unit tests
  integration/      — full-stack tests against a real DB/Redis
  fixtures/         — shared pytest fixtures

alembic/            — Alembic migrations
  versions/
    0001_initial_schema.py
```

## Execution model

Memintel does **not** include a scheduler. Tasks define *what* to evaluate —
the application layer decides *when*. There is no `/scheduler` module.
The application calls `evaluateFull()` on its own schedule.

## Getting started

> **Local development:** set `USE_LLM_FIXTURES=true` to skip real LLM calls and use
> pre-authored fixtures instead. This is the default — no LLM provider or API key is
> needed during development.

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/memintel"
export REDIS_URL="redis://localhost:6379/0"

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | asyncpg DSN for PostgreSQL 15+. Startup fails if missing. |
| `REDIS_URL` | Yes | — | Redis 7+ connection URL. Startup fails if missing. |
| `MEMINTEL_CONFIG_PATH` | Yes | — | Absolute path to `memintel.config.md` (connectors, LLM config, guardrails). |
| `MEMINTEL_ELEVATED_KEY` | No | — | API key required for registry write endpoints. Missing → 403 on those routes. |
| `USE_LLM_FIXTURES` | No | `true` | Set to `false` to call a real LLM provider. `true` uses pre-authored fixtures (no provider needed). |
| `MAX_RETRIES` | No | `3` | Maximum LLM refinement loop attempts per task authoring request. |
| `DB_POOL_MIN` | No | `5` | Minimum asyncpg connection pool size. |
| `DB_POOL_MAX` | No | `20` | Maximum asyncpg connection pool size. |
| `DB_COMMAND_TIMEOUT` | No | `30` | Per-statement database timeout in seconds. |

## Running tests

```bash
pytest tests/unit
pytest tests/integration   # requires running Postgres + Redis
```

## TypeScript SDK

A typed TypeScript client for the Memintel API, located in the `sdk/` directory alongside this backend.

**Install:**

```bash
cd sdk && npm install
```

**Use:**

```typescript
import { MemintelClient } from '@memintel/sdk';

const client = new MemintelClient({
  baseUrl: 'http://localhost:8000/v1',
  apiKey: 'your-api-key',
});
```
