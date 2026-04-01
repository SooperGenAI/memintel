"""
app/stores/task.py
──────────────────────────────────────────────────────────────────────────────
TaskStore — asyncpg-backed persistence for the `tasks` table.

Concurrency model
─────────────────
Updates use optimistic locking: the current `version` value is read with the
row, incremented by 1 in the UPDATE's SET clause, and matched in the WHERE
clause (`WHERE task_id = $1 AND version = $old_version`). If another writer
updated the row between the read and the write, the row count is 0 and the
store retries up to MAX_UPDATE_RETRIES times before raising ConflictError.

This avoids long lock chains and is safe for the low-contention update pattern
(PATCH /tasks is called infrequently compared to evaluations).

Soft deletes
────────────
list() excludes status='deleted' by default unless the caller explicitly
filters by status='deleted'. get() always returns deleted tasks for audit.
update() rejects any attempt to mutate a deleted task.

preview tasks
─────────────
create() raises ValueError immediately if the task has status='preview'.
These tasks are never written to the DB — they are transient dry-run results.

Column ↔ field mapping
──────────────────────
DB column          Python field
─────────────────  ───────────────────────────────
task_id            task.task_id        (None on input; DB default on insert)
intent             task.intent
concept_id         task.concept_id
concept_version    task.concept_version
condition_id       task.condition_id
condition_version  task.condition_version
action_id          task.action_id
action_version     task.action_version
entity_scope       task.entity_scope
delivery           task.delivery       (JSONB — serialised DeliveryConfig)
status             task.status
created_at         task.created_at
updated_at         task.updated_at     (excluded from API; internal)
last_triggered_at  task.last_triggered_at
version            task.version        (excluded from API; optimistic lock)
context_version    task.context_version    (nullable; set at creation time)
guardrails_version task.guardrails_version (nullable; set at creation time)
"""
from __future__ import annotations

import json
import structlog
from typing import Any

import asyncpg

from app.models.task import IMMUTABLE_TASK_FIELDS, DeliveryConfig, Task, TaskList, TaskStatus
from app.models.errors import ConflictError, MemintelError, ErrorType

log = structlog.get_logger(__name__)

#: Maximum number of optimistic-lock retry attempts for update().
MAX_UPDATE_RETRIES: int = 5


class TaskStore:
    """Async store for the `tasks` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── create ────────────────────────────────────────────────────────────────

    async def create(self, task: Task) -> Task:
        """
        Persist a new task and return it with task_id populated.

        Raises ValueError if task.status is 'preview' — preview tasks are
        transient dry-run results and must never reach the DB.
        """
        if task.status == TaskStatus.PREVIEW:
            raise ValueError(
                "preview tasks must not be persisted. "
                "Check dry_run=True path in TaskAuthoringService."
            )

        row = await self._pool.fetchrow(
            """
            INSERT INTO tasks (
                intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                context_version, guardrails_version
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING
                task_id, intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                created_at, updated_at, last_triggered_at, version,
                context_version, guardrails_version
            """,
            task.intent,
            task.concept_id,
            task.concept_version,
            task.condition_id,
            task.condition_version,
            task.action_id,
            task.action_version,
            task.entity_scope,
            task.delivery.model_dump_json(),
            task.status.value,
            task.context_version,
            task.guardrails_version,
        )
        return _row_to_task(row)

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, task_id: str) -> Task | None:
        """
        Return the task by task_id, or None if not found.

        Deleted tasks are returned — callers that need to reject deleted tasks
        must check task.status themselves (e.g. update()).
        """
        row = await self._pool.fetchrow(
            """
            SELECT
                task_id, intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                created_at, updated_at, last_triggered_at, version,
                context_version, guardrails_version
            FROM tasks
            WHERE task_id = $1
            """,
            task_id,
        )
        return _row_to_task(row) if row else None

    # ── list ──────────────────────────────────────────────────────────────────

    async def list(
        self,
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> TaskList:
        """
        Return a cursor-paginated list of tasks.

        Deleted tasks are excluded unless status='deleted' is explicitly passed.
        cursor is the task_id of the last item seen on the previous page.
        limit is capped at 100 to prevent runaway queries.
        """
        limit = min(limit, 100)

        # Build the WHERE predicate
        conditions: list[str] = []
        args: list[Any] = []
        arg_idx = 1

        if status is not None:
            conditions.append(f"status = ${arg_idx}")
            args.append(status)
            arg_idx += 1
        else:
            # Exclude deleted by default
            conditions.append(f"status != ${arg_idx}")
            args.append(TaskStatus.DELETED.value)
            arg_idx += 1

        if cursor is not None:
            # Keyset pagination: fetch rows created before the cursor task
            conditions.append(
                f"created_at < (SELECT created_at FROM tasks WHERE task_id = ${arg_idx})"
            )
            args.append(cursor)
            arg_idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Fetch limit + 1 to determine has_more
        fetch_limit = limit + 1
        args.append(fetch_limit)

        rows = await self._pool.fetch(
            f"""
            SELECT
                task_id, intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                created_at, updated_at, last_triggered_at, version,
                context_version, guardrails_version
            FROM tasks
            {where}
            ORDER BY created_at DESC
            LIMIT ${arg_idx}
            """,
            *args,
        )

        has_more = len(rows) > limit
        page = rows[:limit]
        items = [_row_to_task(r) for r in page]
        next_cursor = items[-1].task_id if has_more and items else None

        # Total count query (same filter minus cursor and limit)
        count_conditions = [c for c in conditions if "created_at <" not in c]
        count_where = "WHERE " + " AND ".join(count_conditions) if count_conditions else ""
        count_args = args[: len(count_conditions)]
        total_count = await self._pool.fetchval(
            f"SELECT COUNT(*) FROM tasks {count_where}", *count_args
        )

        return TaskList(
            items=items,
            has_more=has_more,
            next_cursor=next_cursor,
            total_count=total_count or 0,
        )

    # ── update ────────────────────────────────────────────────────────────────

    async def update(self, task_id: str, updates: dict[str, Any]) -> Task:
        """
        Apply a partial update to a task using optimistic locking.

        Raises:
          MemintelError(PARAMETER_ERROR) — if any immutable field is in updates.
          ConflictError                  — if the task is deleted, or if the
                                           optimistic lock fails after MAX_UPDATE_RETRIES.
        """
        # Guard: reject immutable field changes before touching the DB
        forbidden = IMMUTABLE_TASK_FIELDS & updates.keys()
        if forbidden:
            raise MemintelError(
                ErrorType.PARAMETER_ERROR,
                f"Cannot update immutable field(s): {sorted(forbidden)}. "
                "Create a new task to change concept, condition identity, or action.",
                location=", ".join(sorted(forbidden)),
            )

        for attempt in range(MAX_UPDATE_RETRIES):
            task = await self.get(task_id)
            if task is None:
                raise MemintelError(
                    ErrorType.NOT_FOUND,
                    f"Task '{task_id}' not found.",
                    location="task_id",
                )
            if task.status == TaskStatus.DELETED:
                raise ConflictError(
                    f"Task '{task_id}' has been deleted and cannot be updated.",
                    location="task_id",
                )

            old_version = task.version

            # Build the SET clause dynamically from the updates dict
            set_clauses: list[str] = ["updated_at = NOW()", "version = version + 1"]
            set_args: list[Any] = []
            arg_idx = 1

            for key, value in updates.items():
                if key == "delivery" and isinstance(value, DeliveryConfig):
                    set_clauses.append(f"delivery = ${arg_idx}")
                    set_args.append(value.model_dump_json())
                elif key == "delivery" and isinstance(value, dict):
                    set_clauses.append(f"delivery = ${arg_idx}")
                    set_args.append(json.dumps(value))
                elif key == "status":
                    set_clauses.append(f"status = ${arg_idx}")
                    # Accept both str and enum
                    set_args.append(value.value if hasattr(value, "value") else value)
                else:
                    set_clauses.append(f"{key} = ${arg_idx}")
                    set_args.append(value)
                arg_idx += 1

            # Append task_id and old_version for the WHERE clause
            set_args.append(task_id)
            set_args.append(old_version)

            result = await self._pool.execute(
                f"""
                UPDATE tasks
                SET {', '.join(set_clauses)}
                WHERE task_id = ${arg_idx} AND version = ${arg_idx + 1}
                """,
                *set_args,
            )

            # asyncpg returns 'UPDATE N' — check that exactly one row was updated
            rows_updated = int(result.split()[-1])
            if rows_updated == 1:
                updated = await self.get(task_id)
                return updated  # type: ignore[return-value]

            # Another writer updated the row; retry
            log.debug(
                "task_update_conflict",
                extra={"task_id": task_id, "attempt": attempt + 1},
            )

        raise ConflictError(
            f"Task '{task_id}' could not be updated after {MAX_UPDATE_RETRIES} attempts "
            "due to concurrent modifications.",
            location="task_id",
        )

    # ── find_by_condition_version ─────────────────────────────────────────────

    async def find_by_condition_version(
        self, condition_id: str, version: str
    ) -> list[Task]:
        """
        Return active and paused tasks bound to a specific condition version.

        Used by ApplyCalibrationService to populate tasks_pending_rebind in the
        ApplyCalibrationResult. Deleted tasks are excluded.
        """
        rows = await self._pool.fetch(
            """
            SELECT
                task_id, intent,
                concept_id, concept_version,
                condition_id, condition_version,
                action_id, action_version,
                entity_scope, delivery, status,
                created_at, updated_at, last_triggered_at, version,
                context_version, guardrails_version
            FROM tasks
            WHERE condition_id = $1
              AND condition_version = $2
              AND status IN ('active', 'paused')
            ORDER BY created_at DESC
            """,
            condition_id,
            version,
        )
        return [_row_to_task(r) for r in rows]


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_task(row: asyncpg.Record) -> Task:
    """
    Convert an asyncpg Record from the tasks table into a Task model.

    The `delivery` column is stored as JSONB; asyncpg returns it as a dict
    (or string, depending on codec registration). Both cases are handled.
    """
    delivery_raw = row["delivery"]
    if isinstance(delivery_raw, str):
        delivery_raw = json.loads(delivery_raw)
    delivery = DeliveryConfig.model_validate(delivery_raw)

    return Task(
        task_id=row["task_id"],
        intent=row["intent"],
        concept_id=row["concept_id"],
        concept_version=row["concept_version"],
        condition_id=row["condition_id"],
        condition_version=row["condition_version"],
        action_id=row["action_id"],
        action_version=row["action_version"],
        entity_scope=row["entity_scope"],
        delivery=delivery,
        status=TaskStatus(row["status"]),
        created_at=row["created_at"],
        last_triggered_at=row["last_triggered_at"],
        updated_at=row["updated_at"],
        version=row["version"],
        context_version=row["context_version"],
        guardrails_version=row["guardrails_version"],
    )
