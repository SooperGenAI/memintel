"""
app/persistence/stores.py
──────────────────────────────────────────────────────────────────────────────
FastAPI dependency functions for every store in the persistence layer.

Each function is a thin wrapper that accepts the asyncpg pool (via
Depends(get_db)) and returns the corresponding store instance. Route handlers
and services never construct stores directly — they receive them through these
dependencies.

Why a dedicated wiring module
──────────────────────────────
Separating dependency wiring from store implementations means:
  - Store classes stay testable in isolation (pass a pool directly).
  - Route handlers stay clean — one Depends(get_task_store) per handler.
  - Swapping a store implementation never touches route files.
  - All injection points are discoverable in one place.

Usage in routes:

    from fastapi import APIRouter, Depends
    from app.persistence.stores import get_task_store
    from app.stores import TaskStore

    router = APIRouter()

    @router.get("/tasks/{task_id}")
    async def get_task(
        task_id: str,
        store: TaskStore = Depends(get_task_store),
    ):
        task = await store.get(task_id)
        ...

Available dependencies
──────────────────────
  get_task_store()             → TaskStore
  get_definition_store()       → DefinitionStore
  get_feedback_store()         → FeedbackStore
  get_calibration_token_store() → CalibrationTokenStore
  get_graph_store()            → GraphStore
  get_job_store()              → JobStore
"""
from __future__ import annotations

import asyncpg
from fastapi import Depends

from app.persistence.db import get_db
from app.stores import (
    CalibrationTokenStore,
    ContextStore,
    DefinitionStore,
    FeedbackStore,
    GraphStore,
    JobStore,
    TaskStore,
)


async def get_task_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> TaskStore:
    """FastAPI dependency — returns a TaskStore backed by the shared pool."""
    return TaskStore(pool)


async def get_definition_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> DefinitionStore:
    """FastAPI dependency — returns a DefinitionStore backed by the shared pool."""
    return DefinitionStore(pool)


async def get_feedback_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> FeedbackStore:
    """FastAPI dependency — returns a FeedbackStore backed by the shared pool."""
    return FeedbackStore(pool)


async def get_calibration_token_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> CalibrationTokenStore:
    """FastAPI dependency — returns a CalibrationTokenStore backed by the shared pool."""
    return CalibrationTokenStore(pool)


async def get_graph_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> GraphStore:
    """FastAPI dependency — returns a GraphStore backed by the shared pool."""
    return GraphStore(pool)


async def get_job_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> JobStore:
    """FastAPI dependency — returns a JobStore backed by the shared pool."""
    return JobStore(pool)


async def get_context_store(
    pool: asyncpg.Pool = Depends(get_db),
) -> ContextStore:
    """FastAPI dependency — returns a ContextStore backed by the shared pool."""
    return ContextStore(pool)
