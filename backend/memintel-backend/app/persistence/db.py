"""
app/persistence/db.py
──────────────────────────────────────────────────────────────────────────────
Database and Redis connection factory functions and FastAPI lifespan wiring.

This module is the single place where asyncpg pools and aioredis clients are
created. It is imported by main.py to wire the lifespan context manager, and
by FastAPI dependency functions to inject stores into route handlers.

Environment variables
─────────────────────
  DATABASE_URL        — asyncpg DSN (required)
                        e.g. postgresql://user:password@localhost:5432/memintel
  REDIS_URL           — aioredis URL (required)
                        e.g. redis://localhost:6379/0
  DB_POOL_MIN         — minimum pool connections (default: 5)
  DB_POOL_MAX         — maximum pool connections (default: 20)
  DB_COMMAND_TIMEOUT  — per-statement timeout in seconds (default: 30)

Credentials must never be hardcoded. DATABASE_URL must reference an
environment variable for the password (e.g. ${DB_PASSWORD}). The system
refuses to start if DATABASE_URL or REDIS_URL are unset.

Pool lifetime
─────────────
The pool and Redis client are created once at startup and stored on
app.state. They are closed gracefully in the lifespan finally block.
All store constructors receive the pool via FastAPI dependency injection —
they never create their own connections.

FastAPI dependency functions
────────────────────────────
get_db()     → yields the asyncpg pool from app.state.db
get_redis()  → yields the aioredis client from app.state.redis

Usage in routes:

    from fastapi import Depends
    from app.persistence.db import get_db, get_redis
    from app.stores import TaskStore

    @router.get("/tasks/{task_id}")
    async def get_task(
        task_id: str,
        pool: asyncpg.Pool = Depends(get_db),
    ):
        store = TaskStore(pool)
        ...
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
import aioredis
from fastapi import FastAPI, Request

log = logging.getLogger(__name__)


# ── Pool / client factory functions ───────────────────────────────────────────

async def create_db_pool(dsn: str) -> asyncpg.Pool:
    """
    Create and return an asyncpg connection pool.

    Pool parameters follow the persistence-schema.md recommendations:
      min_size=5, max_size=20, command_timeout=30, statement_cache_size=100.
    Override via DB_POOL_MIN / DB_POOL_MAX / DB_COMMAND_TIMEOUT env vars.
    """
    min_size = int(os.getenv("DB_POOL_MIN", "5"))
    max_size = int(os.getenv("DB_POOL_MAX", "20"))
    command_timeout = float(os.getenv("DB_COMMAND_TIMEOUT", "30"))

    pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        statement_cache_size=100,
    )
    log.info(
        "db_pool_created",
        extra={"min_size": min_size, "max_size": max_size},
    )
    return pool


async def create_redis_client(url: str) -> aioredis.Redis:
    """
    Create and return an aioredis client.

    decode_responses=True ensures all values are returned as str, matching
    the cache layer's expectation of string-serialised JSON.
    """
    client = await aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
    )
    log.info("redis_client_created")
    return client


# ── FastAPI lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Creates the DB pool and Redis client at startup; closes them gracefully
    at shutdown. Wire into FastAPI via:

        app = FastAPI(lifespan=lifespan)

    Raises RuntimeError at startup if DATABASE_URL or REDIS_URL are unset —
    the system must not start with missing connection configuration.
    """
    database_url = os.getenv("DATABASE_URL")
    redis_url = os.getenv("REDIS_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it to a valid asyncpg DSN before starting the server."
        )
    if not redis_url:
        raise RuntimeError(
            "REDIS_URL environment variable is not set. "
            "Set it to a valid Redis URL before starting the server."
        )

    log.info("startup: creating database pool and Redis client")
    app.state.db = await create_db_pool(database_url)
    app.state.redis = await create_redis_client(redis_url)

    try:
        yield
    finally:
        log.info("shutdown: closing database pool and Redis client")
        await app.state.db.close()
        await app.state.redis.close()


# ── FastAPI dependency functions ───────────────────────────────────────────────

async def get_db(request: Request) -> asyncpg.Pool:
    """
    FastAPI dependency — returns the asyncpg pool from app.state.

    Inject into route handlers via Depends(get_db). The pool is shared
    across all requests; connections are checked out per-query by asyncpg.
    """
    return request.app.state.db


async def get_redis(request: Request) -> aioredis.Redis:
    """
    FastAPI dependency — returns the aioredis client from app.state.

    Inject into route handlers or ResultCache via Depends(get_redis).
    """
    return request.app.state.redis
