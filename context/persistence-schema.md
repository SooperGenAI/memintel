# Memintel — Persistence Schema
**Version:** 1.0  
**Status:** Authoritative  
**Audience:** Python backend implementation (Session 3 of Claude Code process)

> **Prerequisite:** Read `core-spec.md` and `py-instructions.md` before this file.
> This document defines the storage layer that all services depend on. The schemas
> here are the ground truth — Pydantic models in `/app/models/` must align with these
> table definitions, not the other way around.

---

## 1. Technology Stack

```
Primary store:   PostgreSQL 15+  (asyncpg driver)
Cache:           Redis 7+        (aioredis driver)
Migrations:      Alembic
Connection pool: asyncpg.Pool    (min=5, max=20, recommended)
```

**Why Postgres:** Transactions, unique constraints, `SELECT FOR UPDATE`, and JSONB
all map directly to Memintel's requirements. Concurrency rules (atomic token
invalidation, serialised task updates) require real transactional guarantees.

**Why Redis:** Cache TTL management, `SET NX` for atomic token redemption, and
request-scoped cache isolation are all native Redis operations.

---

## 2. Foreign Key Strategy

Memintel does **not** use foreign key constraints between tables. This is an intentional design decision, not an oversight.

```
FOREIGN KEY STRATEGY:

Memintel does NOT use database-level foreign key constraints.

Reasons:
  1. Definitions are versioned and immutable — a task referencing
     concept_id='org.churn_risk' version='1.2' is a valid reference
     regardless of whether that definition is in the same DB shard.

  2. Runtime execution must not be blocked by referential integrity
     checks. A missing FK target in the DB does not mean the reference
     is invalid — it may simply be in a different namespace or registry.

  3. Cross-namespace and cross-version references are valid by design.
     Promotion (personal → team → org → global) means the same logical
     definition may exist in multiple namespaces simultaneously.

  4. Write throughput: FK constraints add overhead on every insert and
     delete, which matters for the feedback_records and jobs tables that
     receive high write volume.

Reference integrity is enforced at the APPLICATION layer instead:
  - Compiler layer:     validates all operator and strategy references
                        before producing an ir_hash
  - Registry layer:     checks definition existence before registering tasks
  - Interaction layer:  validates condition_version exists before rebinding

DO NOT add FK constraints to this schema. Future engineers should add
a comment here explaining the strategy rather than adding FKs.
```

---

## 3. Connection Setup

```python
# /app/persistence/db.py

import asyncpg
import aioredis
from fastapi import FastAPI

async def create_db_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn,
        min_size=5,
        max_size=20,
        command_timeout=30,
        statement_cache_size=100,
    )

async def create_redis_client(url: str) -> aioredis.Redis:
    return await aioredis.from_url(url, encoding='utf-8', decode_responses=True)

# FastAPI lifespan wiring:
async def lifespan(app: FastAPI):
    app.state.db = await create_db_pool(settings.DATABASE_URL)
    app.state.redis = await create_redis_client(settings.REDIS_URL)
    yield
    await app.state.db.close()
    await app.state.redis.close()
```

---

## 4. Table Schemas

### 3.1 `tasks`

Stores all tasks. Immutable fields (concept, condition, action identity) are set
at creation and must never be updated. Mutable fields (condition_version, delivery,
entity_scope, status) may be updated via PATCH /tasks/{id}.

```sql
CREATE TABLE tasks (
    task_id           TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
    intent            TEXT        NOT NULL,

    -- Version-pinned references (immutable after creation)
    concept_id        TEXT        NOT NULL,
    concept_version   TEXT        NOT NULL,
    condition_id      TEXT        NOT NULL,
    condition_version TEXT        NOT NULL,  -- mutable: rebind via PATCH
    action_id         TEXT        NOT NULL,
    action_version    TEXT        NOT NULL,

    entity_scope      TEXT        NOT NULL,
    delivery          JSONB       NOT NULL,  -- DeliveryConfig as JSON
    status            TEXT        NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active', 'paused', 'deleted', 'preview')),

    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_triggered_at TIMESTAMPTZ          DEFAULT NULL,

    -- Optimistic locking for concurrent update safety
    version           INTEGER     NOT NULL DEFAULT 1
);

-- Indexes
CREATE INDEX idx_tasks_status         ON tasks (status);
CREATE INDEX idx_tasks_condition      ON tasks (condition_id, condition_version);
CREATE INDEX idx_tasks_concept        ON tasks (concept_id, concept_version);
CREATE INDEX idx_tasks_created_at     ON tasks (created_at DESC);
```

**Concurrency rule:** Task updates MUST use optimistic locking. Read `version`,
increment it in the UPDATE, add `WHERE version = $old_version`. Retry on conflict.
Do not use `SELECT FOR UPDATE` for tasks — optimistic locking is preferred to
avoid long lock chains.

**Immutability enforcement:** The store layer (not the DB) enforces that
`concept_id`, `concept_version`, `condition_id`, `action_id`, `action_version`
cannot be changed via PATCH. The DB schema does not re-enforce this — it is an
application-level invariant validated in `TaskStore.update()`.

**Soft delete semantics:**

```
SOFT DELETE BEHAVIOUR:

status='deleted' marks a task as logically deleted but retains the row for audit.

Query rules enforced in TaskStore:
  TaskStore.list()  → MUST exclude deleted tasks by default
                      (filter WHERE status != 'deleted' unless status='deleted' is
                      explicitly requested in the filter params)
  TaskStore.get()   → MAY return deleted tasks (single lookup by id, for audit)
  TaskStore.update()→ MUST reject updates to deleted tasks → HTTP 409
                      Check status='deleted' before any update and raise immediately

There is no hard delete path in the API. Deleted rows are permanent in the DB.
```

**`preview` tasks are never persisted.** A task with `status='preview'` (dry_run
result) must never be written to this table. The store must reject any attempt to
persist a preview task.

---

### 3.2 `definitions`

Stores concept, condition, and action definitions. **Immutable once registered.**
The unique constraint on `(definition_id, version)` is the DB-level enforcement
of immutability — a second registration attempt with the same id+version returns
HTTP 409.

```sql
CREATE TABLE definitions (
    id              BIGSERIAL   PRIMARY KEY,
    definition_id   TEXT        NOT NULL,   -- fully qualified: 'org.churn_risk'
    version         TEXT        NOT NULL,
    definition_type TEXT        NOT NULL
                        CHECK (definition_type IN ('concept', 'condition', 'action', 'primitive')),
    namespace       TEXT        NOT NULL
                        CHECK (namespace IN ('personal', 'team', 'org', 'global')),
    body            JSONB       NOT NULL,   -- full definition as JSON
    meaning_hash    TEXT                DEFAULT NULL,  -- semantic hash (concept only)
    ir_hash         TEXT                DEFAULT NULL,  -- execution graph hash (concept only)
    deprecated      BOOLEAN     NOT NULL DEFAULT FALSE,
    deprecated_at   TIMESTAMPTZ          DEFAULT NULL,
    replacement_version TEXT             DEFAULT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- IMMUTABILITY CONSTRAINT: same id+version can never be registered twice
    CONSTRAINT uq_definition_version UNIQUE (definition_id, version)
);

-- Indexes
CREATE INDEX idx_definitions_id         ON definitions (definition_id);
CREATE INDEX idx_definitions_type       ON definitions (definition_type);
CREATE INDEX idx_definitions_namespace  ON definitions (namespace);
CREATE INDEX idx_definitions_hash       ON definitions (meaning_hash)
                                         WHERE meaning_hash IS NOT NULL;
CREATE INDEX idx_definitions_created    ON definitions (created_at DESC);

-- Optional GIN index for JSONB body search.
-- Enable this when registry search by definition body fields is needed
-- (e.g. searching conditions by strategy type, or concepts by output type).
-- GIN indexes have higher write overhead — only add when query patterns require it.
-- CREATE INDEX idx_definitions_body_gin ON definitions USING GIN (body);
```

**Version ordering:** versions are stored as text. Listing newest-first uses
`created_at DESC`, not lexicographic version sorting.

**Deprecation:** deprecated definitions are retained for audit. All existing
references continue to work. `deprecated=TRUE` is advisory — the runtime does
not automatically reject deprecated definitions.

---

### 3.3 `feedback_records`

Stores user feedback on specific decisions. The unique constraint enforces
deduplication — one feedback record per decision per user (identified by
condition+entity+timestamp).

```sql
CREATE TABLE feedback_records (
    feedback_id       TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
    condition_id      TEXT        NOT NULL,
    condition_version TEXT        NOT NULL,
    entity            TEXT        NOT NULL,
    decision_timestamp TIMESTAMPTZ NOT NULL,  -- timestamp of the original decision
    feedback          TEXT        NOT NULL
                          CHECK (feedback IN ('false_positive', 'false_negative', 'correct')),
    note              TEXT                 DEFAULT NULL,  -- free text, may contain PII — never log
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- DEDUPLICATION CONSTRAINT: one feedback record per decision
    -- (condition_id, condition_version, entity, decision_timestamp) is the uniqueness key
    CONSTRAINT uq_feedback_decision
        UNIQUE (condition_id, condition_version, entity, decision_timestamp)
);

-- Indexes
CREATE INDEX idx_feedback_condition  ON feedback_records (condition_id, condition_version);
CREATE INDEX idx_feedback_entity     ON feedback_records (entity);
CREATE INDEX idx_feedback_recorded   ON feedback_records (recorded_at DESC);

-- Covering index for the two most common access patterns:
--   1. FeedbackStore.find() — deduplication check by uniqueness key
--   2. CalibrationService — fast condition+version lookup for calibration
-- The UNIQUE constraint above also serves as an index, but this explicit index
-- ensures the query planner uses it efficiently for the find() lookup pattern.
CREATE INDEX idx_feedback_lookup
    ON feedback_records (condition_id, condition_version, entity, decision_timestamp);
```

**Deduplication behaviour:** If a record already exists for the uniqueness key,
`FeedbackStore.create()` must raise a conflict error. The route handler maps this
to HTTP 409 with `error.type='conflict'`. Do not silently ignore duplicates — the
caller must be informed.

---

### 3.4 `calibration_tokens`

Single-use, 24-hour expiry tokens linking a calibration recommendation to its
parameters. Atomic invalidation is critical — concurrent redemption must guarantee
exactly one succeeds.

```sql
CREATE TABLE calibration_tokens (
    id               BIGSERIAL   PRIMARY KEY,
    token_string     TEXT        NOT NULL UNIQUE,  -- opaque random string
    condition_id     TEXT        NOT NULL,
    condition_version TEXT       NOT NULL,
    recommended_params JSONB     NOT NULL,  -- the parameter dict to apply
    expires_at       TIMESTAMPTZ NOT NULL,
    used_at          TIMESTAMPTZ          DEFAULT NULL,  -- set atomically on redemption
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tokens_string   ON calibration_tokens (token_string);
CREATE INDEX idx_tokens_expires  ON calibration_tokens (expires_at);
CREATE INDEX idx_tokens_condition ON calibration_tokens (condition_id, condition_version);
```

**Atomic invalidation — REQUIRED implementation pattern:**

```python
# CalibrationTokenStore.resolve_and_invalidate() — MUST be atomic
# Option A: Postgres UPDATE...RETURNING (preferred)
async def resolve_and_invalidate(self, token_string: str) -> CalibrationToken | None:
    async with self.pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("""
                UPDATE calibration_tokens
                SET used_at = NOW()
                WHERE token_string = $1
                  AND used_at IS NULL
                  AND expires_at > NOW()
                RETURNING *
            """, token_string)
            if row is None:
                return None  # expired, already used, or not found
            return CalibrationToken(**dict(row))

# Option B: Redis SET NX (if using Redis for tokens instead of Postgres)
# SET token:{token_string}:used "1" NX EX 86400
# Returns 1 on success (first use), 0 if already set (duplicate use)
```

**Never return parameters from an expired or already-used token.** The atomicity
of `resolve_and_invalidate()` is what makes the single-use guarantee hold under
concurrent requests.

---

### 3.5 `execution_graphs`

Compiled execution graphs. Stored by `graph_id` (a UUID generated at compile time).
The `ir_hash` is the deterministic hash of the graph content — used for audit
verification at execution time.

```sql
CREATE TABLE execution_graphs (
    graph_id    TEXT        PRIMARY KEY,  -- UUID generated at compile time
    concept_id  TEXT        NOT NULL,
    version     TEXT        NOT NULL,
    ir_hash     TEXT        NOT NULL,     -- SHA-256 of canonical graph JSON
    graph_body  JSONB       NOT NULL,     -- the full ExecutionGraph as JSON
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_graph_concept_version UNIQUE (concept_id, version)
);

-- Indexes
CREATE INDEX idx_graphs_concept   ON execution_graphs (concept_id, version);
CREATE INDEX idx_graphs_ir_hash   ON execution_graphs (ir_hash);
```

**Note:** `(concept_id, version)` uniqueness means recompiling the same version
replaces the existing graph. This is intentional — recompilation of an unchanged
definition always produces the same `ir_hash`, so replacement is safe.

**Graph replacement invariant:**

```
GRAPH REPLACEMENT INVARIANT:

Recompilation of an unchanged definition MUST produce the same ir_hash.

Before overwriting an existing graph in GraphStore.store():
  1. Compute the new ir_hash
  2. If an existing graph exists for (concept_id, version):
     a. If new_ir_hash == existing_ir_hash → safe to overwrite (no-op in practice)
     b. If new_ir_hash != existing_ir_hash → CRITICAL COMPILER BUG
        → DO NOT overwrite the existing graph
        → Raise a CompilerInvariantError with both hashes
        → Log at ERROR level with full context

A different ir_hash for the same (concept_id, version) means either:
  - the definition body was mutated after registration (should be impossible)
  - the compiler is non-deterministic (critical bug)
Either case requires investigation before proceeding.
```

---

### 3.6 `jobs`

Async job queue for long-running executions. Status transitions are strictly
ordered — no backward transitions.

```sql
CREATE TABLE jobs (
    job_id          TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::text,
    job_type        TEXT        NOT NULL DEFAULT 'execute',
    status          TEXT        NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    request_body    JSONB       NOT NULL,  -- the original ExecuteRequest
    result_body     JSONB                DEFAULT NULL,  -- populated on completion
    error_body      JSONB                DEFAULT NULL,  -- populated on failure
    poll_interval_s INTEGER     NOT NULL DEFAULT 2,
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ          DEFAULT NULL,
    completed_at    TIMESTAMPTZ          DEFAULT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_jobs_status   ON jobs (status);
CREATE INDEX idx_jobs_enqueued ON jobs (enqueued_at DESC);
```

**Valid status transitions (enforced in `JobStore.update_status()`):**

```
queued    → running   ✅
queued    → cancelled ✅
running   → completed ✅
running   → failed    ✅
running   → cancelled ✅
completed → *         ❌ (terminal — no further transitions)
failed    → *         ❌ (terminal — no further transitions)
cancelled → *         ❌ (terminal — no further transitions)
```

Any attempt to transition from a terminal state must raise an error (HTTP 409).
Implement a `VALID_TRANSITIONS` dict in `JobStore` and check before every update.

---

## 5. Result Cache (Redis)

The result cache stores computed concept outputs keyed by execution context.
It lives in Redis, not Postgres — it is a performance layer, not a store of record.

### 4.1 Cache Key Structure

```python
# EXACT KEY FORMAT — do not deviate from this
# Key: "result:{concept_id}:{version}:{entity}:{timestamp_or_SNAPSHOT}"

def make_cache_key(concept_id: str, version: str, entity: str,
                   timestamp: str | None) -> str:
    ts = timestamp if timestamp is not None else "SNAPSHOT"
    return f"result:{concept_id}:{version}:{entity}:{ts}"

# Examples:
# "result:org.churn_risk:1.2:user_abc:2024-03-15T09:00:00Z"   ← deterministic
# "result:org.churn_risk:1.2:user_abc:SNAPSHOT"               ← snapshot mode

# CRITICAL: None timestamp and any specific timestamp are DIFFERENT KEYS.
# They must never resolve to the same cache entry.
# SNAPSHOT keys must NEVER be stored in Redis (see TTL rules below).

# CACHE KEY SAFETY — collision prevention:
# Key components (entity, timestamp) may contain colons or special characters.
# Raw interpolation can cause key collisions:
#   entity='user:123' → "result:org.churn_risk:1.2:user:123:2024-01-15T09:00:00Z"
#   This is ambiguous — is 'user' the entity and '123' part of the timestamp?
#
# REQUIRED: URL-encode (percent-encode) entity and timestamp before building the key.
# concept_id and version are already dot-and-number safe from the namespace rules.

from urllib.parse import quote

def make_cache_key(concept_id: str, version: str, entity: str,
                   timestamp: str | None) -> str:
    safe_entity = quote(entity, safe='')     # encodes colons, slashes, etc.
    safe_ts = quote(timestamp, safe='') if timestamp is not None else "SNAPSHOT"
    return f"result:{concept_id}:{version}:{safe_entity}:{safe_ts}"

# Examples (safe):
# entity='user:123', ts='2024-01-15T09:00:00Z'
# → "result:org.churn_risk:1.2:user%3A123:2024-01-15T09%3A00%3A00Z"
#
# entity='user_abc', ts=None
# → "result:org.churn_risk:1.2:user_abc:SNAPSHOT"  (no encoding needed, still safe)
```

### 4.2 TTL Rules

```python
class ResultCache:
    async def set(self, key: str, result: Result) -> None:
        if "SNAPSHOT" in key:
            # NEVER persist snapshot results — they are current-state only
            # Any call that would write a SNAPSHOT key must be a no-op
            return

        # Deterministic results: cache indefinitely (same input = same output always)
        # Use a very long TTL rather than no expiry to allow Redis memory management
        TTL_DETERMINISTIC = 60 * 60 * 24 * 365  # 1 year

        serialised = result.model_dump_json()
        await self.redis.setex(key, TTL_DETERMINISTIC, serialised)

    async def get(self, key: str) -> Result | None:
        if "SNAPSHOT" in key:
            # Snapshot results are never in cache
            return None
        raw = await self.redis.get(key)
        if raw is None:
            return None
        return Result.model_validate_json(raw)
```

### 4.3 Graph Cache (In-Memory)

The graph cache is a process-level in-memory dict, not Redis. It caches compiled
`graph_id` values at startup to avoid re-compiling hot concepts on every request.

```python
# /app/config/graph_cache.py

GRAPH_CACHE: dict[str, str] = {}  # "concept_id:version" → graph_id

async def warm_graph_cache(active_concepts: list[tuple[str, str]],
                            compiler: Compiler) -> None:
    """Call during FastAPI startup event."""
    for concept_id, version in active_concepts:
        graph = await compiler.compile(concept_id, version)
        GRAPH_CACHE[f"{concept_id}:{version}"] = graph.graph_id
```

---

## 6. Store Interfaces

Each store is a class that takes a db pool (or redis client) in its constructor.
All methods are `async`. Stores are injected into services via FastAPI dependencies.

### 5.1 `TaskStore`

```python
class TaskStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def create(self, task: Task) -> Task:
        # Must reject preview tasks (status='preview') — raise ValueError
        # Sets task_id via DB default (gen_random_uuid)
        # Returns the created task with task_id populated

    async def get(self, task_id: str) -> Task | None: ...

    async def list(
        self,
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,   # opaque cursor (last task_id seen)
    ) -> TaskList: ...

    async def update(self, task_id: str, updates: dict) -> Task:
        # Optimistic locking: increment version, retry on conflict
        # Raise HTTP 409 if task status is 'deleted'
        # Raise HTTP 400 if immutable fields are in updates dict:
        #   concept_id, concept_version, condition_id, action_id, action_version
        # Returns updated task

    async def find_by_condition_version(
        self, condition_id: str, version: str
    ) -> list[Task]:
        # Used by apply-calibration to find tasks_pending_rebind
        # Returns tasks where condition_id=condition_id AND condition_version=version
        # AND status IN ('active', 'paused')  — exclude deleted
```

### 5.2 `DefinitionStore`

```python
class DefinitionStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict,
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        # Raises ConflictError (→ HTTP 409) if (definition_id, version) already exists
        # IMPORTANT: caller must validate definition through compiler BEFORE calling register()
        # This store does NOT validate — it only persists

    async def get(
        self, definition_id: str, version: str
    ) -> dict | None: ...
    # Returns raw body dict or None (→ HTTP 404 in route)

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        # Returns all versions of a definition, newest-first (by created_at DESC)

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult: ...

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse: ...

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        # Promoting to 'global' requires elevated_key=True → raise HTTP 403 if not
```

### 5.3 `FeedbackStore`

```python
class FeedbackStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def create(self, record: FeedbackRecord) -> FeedbackRecord:
        # Raises ConflictError (→ HTTP 409) on uniqueness key violation:
        #   (condition_id, condition_version, entity, decision_timestamp)
        # The DB unique constraint is the final guard, but the store
        # should also check before insert for a cleaner error message

    async def get_by_condition(
        self,
        condition_id: str,
        version: str,
    ) -> list[FeedbackRecord]:
        # Returns all feedback for a condition version
        # Used by CalibrationService.derive_direction()
        # Ordered by recorded_at ASC (oldest first — preserves chronological signal)

    async def find(
        self,
        condition_id: str,
        condition_version: str,
        entity: str,
        timestamp: str,
    ) -> FeedbackRecord | None:
        # Looks up a specific feedback record by its uniqueness key
        # Used to check if feedback already exists before submitting
```

### 5.4 `CalibrationTokenStore`

```python
class CalibrationTokenStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def create(self, token: CalibrationToken) -> str:
        # Generates a cryptographically random token_string
        # Stores in calibration_tokens table with expires_at = NOW() + 24h
        # Returns the token_string (opaque to the caller)

    async def resolve_and_invalidate(self, token_string: str) -> CalibrationToken | None:
        # MUST be atomic — use UPDATE...RETURNING in a single statement
        # Sets used_at = NOW() WHERE token_string = $1 AND used_at IS NULL AND expires_at > NOW()
        # Returns the CalibrationToken if successful, None if expired/used/not found
        # A None return → HTTP 400 'invalid or expired calibration token'
        # Second concurrent call for the same token MUST return None (at-most-once)
```

### 5.5 `GraphStore`

```python
class GraphStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def store(self, graph: ExecutionGraph) -> str:
        # Upserts execution_graphs: if (concept_id, version) exists, replace it
        # Recompilation of unchanged definition always produces same ir_hash → safe
        # Returns graph_id

    async def get(self, graph_id: str) -> ExecutionGraph | None: ...

    async def get_by_concept(
        self, concept_id: str, version: str
    ) -> ExecutionGraph | None: ...
```

### 5.6 `JobStore`

```python
VALID_TRANSITIONS = {
    'queued':    {'running', 'cancelled'},
    'running':   {'completed', 'failed', 'cancelled'},
    'completed': set(),   # terminal
    'failed':    set(),   # terminal
    'cancelled': set(),   # terminal
}

class JobStore:
    def __init__(self, pool: asyncpg.Pool): ...

    async def enqueue(self, request_body: dict) -> Job:
        # Creates job with status='queued'
        # Returns Job with job_id, status, poll_interval_s

    async def get(self, job_id: str) -> Job | None: ...

    async def update_status(
        self,
        job_id: str,
        new_status: str,
        result: dict | None = None,
        error: dict | None = None,
    ) -> Job:
        # Validates transition via VALID_TRANSITIONS
        # Raises HTTP 409 if transition is invalid (e.g. completed → running)
        # Sets started_at on first transition to 'running'
        # Sets completed_at on transition to completed/failed/cancelled

    async def cancel(self, job_id: str) -> Job:
        # Convenience wrapper: update_status(job_id, 'cancelled')
        # Raises HTTP 409 if already in terminal state
```

---

## 7. Alembic Migration Setup

```
/memintel-backend
  alembic.ini
  /alembic
    env.py
    /versions
      0001_initial_schema.py   ← creates all 6 tables in one migration
```

```python
# alembic/versions/0001_initial_schema.py — creates all tables

def upgrade():
    op.execute("""
        CREATE TABLE tasks ( ... );
        CREATE TABLE definitions ( ... );
        CREATE TABLE feedback_records ( ... );
        CREATE TABLE calibration_tokens ( ... );
        CREATE TABLE execution_graphs ( ... );
        CREATE TABLE jobs ( ... );
        -- all indexes
    """)

def downgrade():
    op.execute("""
        DROP TABLE IF EXISTS jobs;
        DROP TABLE IF EXISTS execution_graphs;
        DROP TABLE IF EXISTS calibration_tokens;
        DROP TABLE IF EXISTS feedback_records;
        DROP TABLE IF EXISTS definitions;
        DROP TABLE IF EXISTS tasks;
    """)
```

**Migration rule:** every schema change gets a new numbered migration file.
Never edit an existing migration — always add a new one.

---

## 8. Concurrency Safety Summary

These rules must be enforced in the store implementations, not assumed:

| Operation | Mechanism | Failure mode |
|---|---|---|
| Calibration token redemption | `UPDATE ... WHERE used_at IS NULL` (atomic) | Returns `None` — second concurrent caller gets nothing |
| Task update | Optimistic locking (`WHERE version = $old`) | Retry on conflict |
| Definition registration | DB unique constraint `(definition_id, version)` | `asyncpg.UniqueViolationError` → HTTP 409 |
| Feedback submission | DB unique constraint `(condition_id, condition_version, entity, decision_timestamp)` | `asyncpg.UniqueViolationError` → HTTP 409 |
| Cache write | Redis `SETEX` (idempotent by design) | Same key+value written twice → no-op |
| Job status update | Transition table check before every update | Invalid transition → HTTP 409 |

---

## 9. Environment Variables

```bash
DATABASE_URL=postgresql://user:${DB_PASSWORD}@localhost:5432/memintel
REDIS_URL=redis://localhost:6379/0

# Connection pool tuning (optional overrides)
DB_POOL_MIN=5
DB_POOL_MAX=20
DB_COMMAND_TIMEOUT=30
```

**Never hardcode credentials.** All connection strings must reference environment
variables. The `ConfigLoader` (Session 2.5) validates these at startup — the system
refuses to start if any referenced variable is unset.
