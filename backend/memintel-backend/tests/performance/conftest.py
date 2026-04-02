"""
tests/performance/conftest.py
──────────────────────────────────────────────────────────────────────────────
Performance test infrastructure — raw asyncpg pool for EXPLAIN plan queries
and index existence checks.  No FastAPI app needed; we query the DB directly.

Database: postgresql://postgres:admin@localhost:5433/memintel_test
Tests are skipped gracefully if the database is unavailable.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Stub aioredis before any app import.
if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import asyncpg
import pytest
from datetime import timedelta

# ── Constants ──────────────────────────────────────────────────────────────────

TEST_DATABASE_URL: str = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:admin@localhost:5433/memintel_test",
)

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _admin_url() -> str:
    base, _ = TEST_DATABASE_URL.rsplit("/", 1)
    return f"{base}/postgres"


def _run_alembic_migrations() -> None:
    env = {**os.environ, "DATABASE_URL": TEST_DATABASE_URL}
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed (exit {result.returncode}):\n{result.stderr}"
        )


async def _create_test_database() -> None:
    conn = await asyncpg.connect(_admin_url())
    try:
        await conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'memintel_test' AND pid <> pg_backend_pid()
            """
        )
        await conn.execute("DROP DATABASE IF EXISTS memintel_test")
        await conn.execute("CREATE DATABASE memintel_test")
    finally:
        await conn.close()


_ALL_TABLES: tuple[str, ...] = (
    "decisions",
    "concept_results",
    "feedback_records",
    "calibration_tokens",
    "execution_graphs",
    "jobs",
    "application_context",
    "guardrails_versions",
    "tasks",
    "definitions",
)


async def _truncate_all(pool: asyncpg.Pool) -> None:
    table_list = ", ".join(_ALL_TABLES)
    async with pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {table_list} RESTART IDENTITY CASCADE")


# ── Session-level DB setup ─────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _database_setup():
    """
    Create and migrate the memintel_test database once per pytest session.

    Skips all performance tests gracefully if PostgreSQL is unreachable or
    migrations fail.
    """
    setup_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(setup_loop)
    try:
        setup_loop.run_until_complete(_create_test_database())
    except Exception as exc:
        setup_loop.close()
        asyncio.set_event_loop(None)
        pytest.skip(
            f"Cannot create test database — is PostgreSQL running on "
            f"{_admin_url()}? Error: {exc}"
        )
        return
    finally:
        if not setup_loop.is_closed():
            setup_loop.close()

    asyncio.set_event_loop(None)

    try:
        _run_alembic_migrations()
    except Exception as exc:
        pytest.skip(f"Alembic migrations failed: {exc}")
        return

    yield  # all performance tests run here


# ── Session-scoped event loop and pool ────────────────────────────────────────

@pytest.fixture(scope="session")
def _loop(_database_setup):
    """Session-scoped event loop shared by all performance fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(scope="session")
def db_pool(_loop):
    """Session-scoped asyncpg pool — shared across all performance tests."""
    async def _create():
        return await asyncpg.create_pool(
            TEST_DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )

    pool = _loop.run_until_complete(_create())
    yield pool
    _loop.run_until_complete(pool.close())


def run(loop: asyncio.AbstractEventLoop, coro):
    """Convenience helper: run a coroutine on the session loop."""
    return loop.run_until_complete(coro)


# ── Seed fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def seed_explain_data(_loop, db_pool):
    """
    Populate the test DB with enough rows to make EXPLAIN plan tests meaningful.

    Row counts:
      decisions        5 000
      concept_results  5 000
      feedback_records 2 000
      definitions     10 000
    """
    now = datetime.now(timezone.utc)

    async def _seed():
        await _truncate_all(db_pool)

        async with db_pool.acquire() as conn:
            # ── decisions ─────────────────────────────────────────────────────
            decisions_rows = [
                (
                    str(uuid.uuid4()),                              # decision_id
                    f"concept-{i % 50}",                           # concept_id
                    f"v{(i % 5) + 1}",                             # concept_version
                    f"condition-{i % 20}",                         # condition_id
                    f"v{(i % 3) + 1}",                             # condition_version
                    f"entity-{i % 200}",                           # entity_id
                    now - timedelta(seconds=i),                    # evaluated_at (unique per row)
                    i % 2 == 0,                                    # fired
                    str(float(i % 100) / 100.0),                   # concept_value (TEXT since migration 0006)
                    False,                                         # dry_run
                )
                for i in range(5000)
            ]
            await conn.executemany(
                """
                INSERT INTO decisions (
                    decision_id, concept_id, concept_version,
                    condition_id, condition_version, entity_id,
                    evaluated_at, fired, concept_value, dry_run
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                decisions_rows,
            )

            # ── concept_results ───────────────────────────────────────────────
            concept_results_rows = [
                (
                    f"concept-{i % 50}",      # concept_id
                    f"v{(i % 5) + 1}",        # version
                    f"entity-{i % 200}",      # entity
                    float(i % 100) / 100.0,   # value
                    "numeric",                # output_type
                    now,                      # evaluated_at
                )
                for i in range(5000)
            ]
            await conn.executemany(
                """
                INSERT INTO concept_results (
                    concept_id, version, entity, value, output_type, evaluated_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                concept_results_rows,
            )

            # ── feedback_records ──────────────────────────────────────────────
            # UNIQUE (condition_id, condition_version, entity, decision_timestamp)
            feedback_values = ["false_positive", "false_negative", "correct"]
            base_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
            feedback_rows = [
                (
                    f"condition-{i % 20}",               # condition_id
                    f"v{(i % 3) + 1}",                   # condition_version
                    f"entity-{i % 200}",                 # entity
                    base_ts + timedelta(minutes=i),      # decision_timestamp — unique per i
                    feedback_values[i % 3],              # feedback
                    now,                                 # recorded_at
                )
                for i in range(2000)
            ]
            await conn.executemany(
                """
                INSERT INTO feedback_records (
                    condition_id, condition_version, entity,
                    decision_timestamp, feedback, recorded_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                feedback_rows,
            )

            # ── definitions ───────────────────────────────────────────────────
            # UNIQUE (definition_id, version) — use i to ensure uniqueness
            def_types = ["concept", "condition", "action", "primitive", "feature"]
            namespaces = ["personal", "team", "org", "global"]
            definitions_rows = [
                (
                    f"def-{i}",                          # definition_id
                    "1.0",                               # version
                    def_types[i % len(def_types)],       # definition_type
                    namespaces[i % len(namespaces)],     # namespace
                    '{"type": "number"}',                # body (JSONB)
                    now,                                 # created_at
                )
                for i in range(10000)
            ]
            await conn.executemany(
                """
                INSERT INTO definitions (
                    definition_id, version, definition_type, namespace, body, created_at
                ) VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                definitions_rows,
            )

    run(_loop, _seed())
    yield
