"""
tests/unit/test_fix3_guardrails_version.py
──────────────────────────────────────────────────────────────────────────────
FIX 3 tests: guardrails_version is written to and read from the DB.

Uses a MockPool that records all INSERT VALUES and SELECT columns to verify
that guardrails_version is included in both paths.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.models.task import DeliveryConfig, Task, TaskStatus
from app.stores.task import TaskStore, _row_to_task


def run(coro):
    return asyncio.run(coro)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(guardrails_version: str | None = "v2") -> Task:
    return Task(
        intent="Alert when churn is high",
        concept_id="org.churn_score",
        concept_version="1.0",
        condition_id="high_churn",
        condition_version="1.0",
        action_id="org.notify",
        action_version="1.0",
        entity_scope="all_users",
        delivery=DeliveryConfig(type="webhook", endpoint="https://example.com/hook"),
        status=TaskStatus.ACTIVE,
        context_version="ctx-1",
        guardrails_version=guardrails_version,
    )


def _task_row(guardrails_version: str | None = "v2") -> dict:
    """Minimal asyncpg-like record dict for a task row."""
    import datetime

    class _Record(dict):
        """Quacks like an asyncpg.Record."""
        def __getitem__(self, key):
            return super().__getitem__(key)

    return _Record({
        "task_id": "task-001",
        "intent": "Alert when churn is high",
        "concept_id": "org.churn_score",
        "concept_version": "1.0",
        "condition_id": "high_churn",
        "condition_version": "1.0",
        "action_id": "org.notify",
        "action_version": "1.0",
        "entity_scope": "all_users",
        "delivery": json.dumps({
            "type": "webhook",
            "endpoint": "https://example.com/hook",
        }),
        "status": "active",
        "created_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        "updated_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        "last_triggered_at": None,
        "version": 1,
        "context_version": "ctx-1",
        "guardrails_version": guardrails_version,
    })


class _RecordingPool:
    """Fake asyncpg pool that records all fetchrow/fetch calls."""

    def __init__(self, return_row=None):
        self._return_row = return_row
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        return self._return_row

    async def fetch(self, query: str, *args: Any) -> list:
        self.fetch_calls.append((query, args))
        return [self._return_row] if self._return_row else []

    async def fetchval(self, query: str, *args: Any) -> Any:
        return 0

    async def execute(self, query: str, *args: Any) -> str:
        return "UPDATE 1"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_task_store_create_includes_guardrails_version_in_insert():
    """
    TaskStore.create() INSERT must include guardrails_version as a column
    and pass the task's guardrails_version as the corresponding parameter.

    FIX 3: the column was added in migration 0003 but omitted from the INSERT
    before this fix — guardrails_version was silently discarded.
    """
    task = _make_task(guardrails_version="v2")
    pool = _RecordingPool(return_row=_task_row(guardrails_version="v2"))
    store = TaskStore(pool=pool)

    result = run(store.create(task))

    assert len(pool.fetchrow_calls) == 1, "create() must call fetchrow once"
    query, args = pool.fetchrow_calls[0]

    assert "guardrails_version" in query, (
        "INSERT must include guardrails_version column"
    )
    assert "v2" in args, (
        f"guardrails_version='v2' must be passed as an SQL parameter. Args: {args}"
    )
    # Returned task must carry the value.
    assert result.guardrails_version == "v2", (
        f"Returned task.guardrails_version must be 'v2', got {result.guardrails_version!r}"
    )


def test_task_store_create_includes_guardrails_version_null():
    """
    TaskStore.create() handles guardrails_version=None correctly
    (no active API guardrails version → stores NULL).
    """
    task = _make_task(guardrails_version=None)
    pool = _RecordingPool(return_row=_task_row(guardrails_version=None))
    store = TaskStore(pool=pool)

    result = run(store.create(task))

    query, args = pool.fetchrow_calls[0]
    assert "guardrails_version" in query
    # None must appear in the args list (will be bound as SQL NULL).
    assert None in args, f"guardrails_version=None must be in args. Args: {args}"
    assert result.guardrails_version is None


def test_task_store_get_selects_guardrails_version():
    """
    TaskStore.get() SELECT must include guardrails_version and _row_to_task()
    must read it from the row.
    """
    pool = _RecordingPool(return_row=_task_row(guardrails_version="v3"))
    store = TaskStore(pool=pool)

    result = run(store.get("task-001"))

    assert len(pool.fetchrow_calls) == 1
    query, _ = pool.fetchrow_calls[0]
    assert "guardrails_version" in query, (
        "SELECT in get() must include guardrails_version"
    )
    assert result is not None
    assert result.guardrails_version == "v3", (
        f"task.guardrails_version must be 'v3', got {result.guardrails_version!r}"
    )


def test_row_to_task_reads_guardrails_version():
    """
    _row_to_task() must read row['guardrails_version'] and map it to
    task.guardrails_version. This was missing before FIX 3.
    """
    row = _task_row(guardrails_version="v5")
    task = _row_to_task(row)

    assert task.guardrails_version == "v5", (
        f"_row_to_task() must populate guardrails_version from row. Got: {task.guardrails_version!r}"
    )
