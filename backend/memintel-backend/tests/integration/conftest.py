"""
tests/integration/conftest.py
──────────────────────────────────────────────────────────────────────────────
Pytest fixtures for real-database integration tests.

Event-loop strategy
───────────────────
asyncpg connections and pools are bound to the event loop they are created in.
Using a single persistent module-level loop causes "Future attached to a
different loop" errors on Windows/Python 3.11 because asyncpg internally
calls asyncio.get_event_loop() at points where no loop is running, and gets
whatever loop pytest-asyncio or the OS has set as the thread default.

Fix: each test function gets a FRESH event loop (and a fresh pool created in
that loop). asyncio.set_event_loop() makes it the thread-current loop for the
duration of the test. Pool + loop are torn down after each test.

Cost: ~20 ms per test for pool creation. Acceptable for integration tests.

Session-level work (create DB, run migrations) is done ONCE using its own
short-lived event loop — this is safe because no pool is kept alive across
that loop's lifetime.

Environment
───────────
TEST_DATABASE_URL overrides the default:
  postgresql://postgres:admin@localhost:5433/memintel_test
"""
from __future__ import annotations

import asyncio
import os
import subprocess

import asyncpg
import pytest

# ── Connection URL ─────────────────────────────────────────────────────────────

TEST_DATABASE_URL: str = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:admin@localhost:5433/memintel_test",
)

# ── Tables created by all migrations ──────────────────────────────────────────

_ALL_TABLES: tuple[str, ...] = (
    "decisions",
    "concept_results",
    "feedback_records",
    "calibration_tokens",
    "execution_graphs",
    "jobs",
    "application_context",  # migration 0002 — singular
    "guardrails_versions",  # migration 0003
    "tasks",
    "definitions",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _admin_url() -> str:
    """URL to the postgres admin database (used to CREATE/DROP memintel_test)."""
    base, _ = TEST_DATABASE_URL.rsplit("/", 1)
    return f"{base}/postgres"


async def _create_test_database() -> None:
    """Drop-and-recreate memintel_test, terminating stale connections first."""
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


def _run_alembic_migrations() -> None:
    """
    Run `alembic upgrade head` against memintel_test.

    Uses a subprocess so alembic can set up its own async engine independently
    of our test event loops.  DATABASE_URL is injected via environment.
    """
    env = {**os.environ, "DATABASE_URL": TEST_DATABASE_URL}
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )


async def _truncate_all(pool: asyncpg.Pool) -> None:
    """TRUNCATE every application table, resetting serial sequences."""
    table_list = ", ".join(_ALL_TABLES)
    async with pool.acquire() as conn:
        await conn.execute(
            f"TRUNCATE {table_list} RESTART IDENTITY CASCADE"
        )


# ── Session-level database setup (runs ONCE per test session) ─────────────────

@pytest.fixture(scope="session", autouse=True)
def _database_setup():
    """
    Create the memintel_test database and run all Alembic migrations.

    Runs exactly once per pytest session.  All integration tests that depend
    on `db_pool` (directly or via `clean_tables` / `run`) are automatically
    skipped if this step fails.

    autouse=True ensures this runs before any test in the session, even tests
    that don't explicitly request this fixture.
    """
    # Use a dedicated short-lived loop for the one-time DB creation.
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
        pytest.skip(f"Alembic migrations failed — cannot run integration tests: {exc}")
        return

    yield   # all tests run here

    # No teardown: the test database persists after the session for inspection.


# ── Per-test pool + event loop ─────────────────────────────────────────────────

@pytest.fixture
def db_pool(_database_setup):
    """
    Per-test asyncpg connection pool.

    Creates a FRESH event loop + pool for each test, then tears them down.
    This eliminates "Future attached to a different loop" errors on Windows
    by ensuring pool and all of its connections are always in the same loop.

    Also truncates all tables before yielding, providing a clean slate for
    every test without relying on a separate clean_tables fixture call.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        pool: asyncpg.Pool = loop.run_until_complete(
            asyncpg.create_pool(
                TEST_DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        )
    except Exception as exc:
        loop.close()
        asyncio.set_event_loop(None)
        pytest.skip(f"Cannot connect to test database pool: {exc}")
        return

    # Truncate all tables so each test starts with an empty database.
    loop.run_until_complete(_truncate_all(pool))

    yield pool

    loop.run_until_complete(pool.close())
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def run(db_pool):
    """
    Helper that runs a coroutine in the current test's event loop.

    Usage::

        def test_something(db_pool, run):
            store = SomeStore(db_pool)
            result = run(store.get("id"))
            assert result is not None

    Must be requested AFTER db_pool (which sets the event loop).
    Requesting both in the test signature is sufficient — pytest resolves
    db_pool first because run depends on it.
    """
    loop = asyncio.get_event_loop()  # the loop set by db_pool fixture

    def _runner(coro):
        return loop.run_until_complete(coro)

    return _runner


@pytest.fixture
def clean_tables(db_pool, run):
    """
    No-op compatibility fixture — tables are already truncated by db_pool.

    Kept so tests can declare `clean_tables` in their signature for clarity
    without needing to change when/how truncation happens.
    """
    yield
