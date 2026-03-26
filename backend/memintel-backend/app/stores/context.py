"""
app/stores/context.py
────────────────────────────────────────────────────────────────────────────────
ContextStore — asyncpg-backed persistence for the `application_context` table.

Invariants
──────────
  - Only one row has is_active = TRUE at any time.
  - create() deactivates the previous active context atomically in a transaction.
  - Contexts are never deleted — only superseded.
  - version is auto-assigned: "v1" for the first row, "v{n+1}" for each subsequent.
  - get_active() returns None when no context exists (never raises).

Column ↔ field mapping
──────────────────────
DB column              Python field
─────────────────────  ─────────────────────────────────────
context_id             context.context_id   (TEXT PRIMARY KEY)
version                context.version      (VARCHAR 10)
domain_json            context.domain       (DomainContext, JSONB)
behavioural_json       context.behavioural  (BehaviouralContext, JSONB)
semantic_hints_json    context.semantic_hints (list[SemanticHint], JSONB)
calibration_bias_json  context.calibration_bias (CalibrationBias|None, JSONB)
created_at             context.created_at
is_active              context.is_active
"""
from __future__ import annotations

import json

import asyncpg

from app.models.context import ApplicationContext, ContextImpactResult


def _row_to_context(row: asyncpg.Record) -> ApplicationContext:
    """Convert a DB row from application_context into an ApplicationContext."""
    return ApplicationContext.model_validate({
        "context_id":      row["context_id"],
        "version":         row["version"],
        "domain":          json.loads(row["domain_json"]),
        "behavioural":     json.loads(row["behavioural_json"]),
        "semantic_hints":  json.loads(row["semantic_hints_json"]),
        "calibration_bias": (
            json.loads(row["calibration_bias_json"])
            if row["calibration_bias_json"] is not None
            else None
        ),
        "created_at": row["created_at"],
        "is_active":  row["is_active"],
    })


class ContextStore:
    """
    Persistence layer for ApplicationContext.

    Public API
    ----------
    create(context)             → ApplicationContext
        Persist a new context, deactivating the previous one atomically.
        Assigns version ("v1", "v2", …) before inserting.

    get_active()                → ApplicationContext | None
        Return the current active context, or None.

    get_version(version)        → ApplicationContext | None
        Return a specific version by version string ("v1", "v2", …), or None.

    list_versions()             → list[ApplicationContext]
        Return all context versions, newest first.

    get_impact()                → ContextImpactResult
        Return task distribution across context versions.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, context: ApplicationContext) -> ApplicationContext:
        """
        Persist ``context`` as the new active version.

        Steps (all in one transaction):
          1. Count existing rows to determine the next version number.
          2. Set version on context ("v{n+1}").
          3. Set is_active = FALSE on the current active row (if any).
          4. INSERT the new row with is_active = TRUE.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                count: int = await conn.fetchval(
                    "SELECT COUNT(*) FROM application_context"
                )
                context.version = f"v{count + 1}"

                # Deactivate previous active context
                await conn.execute(
                    "UPDATE application_context SET is_active = FALSE WHERE is_active = TRUE"
                )

                await conn.execute(
                    """
                    INSERT INTO application_context (
                        context_id, version, domain_json, behavioural_json,
                        semantic_hints_json, calibration_bias_json, created_at, is_active
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    context.context_id,
                    context.version,
                    context.domain.model_dump_json(),
                    context.behavioural.model_dump_json(),
                    json.dumps([s.model_dump() for s in context.semantic_hints]),
                    context.calibration_bias.model_dump_json() if context.calibration_bias else None,
                    context.created_at,
                    True,
                )

        return context

    async def get_active(self) -> ApplicationContext | None:
        """Return the active context, or None if none exists."""
        row = await self._pool.fetchrow(
            "SELECT * FROM application_context WHERE is_active = TRUE LIMIT 1"
        )
        if row is None:
            return None
        return _row_to_context(row)

    async def get_version(self, version: str) -> ApplicationContext | None:
        """Return the context with the given version string, or None."""
        row = await self._pool.fetchrow(
            "SELECT * FROM application_context WHERE version = $1",
            version,
        )
        if row is None:
            return None
        return _row_to_context(row)

    async def list_versions(self) -> list[ApplicationContext]:
        """Return all context versions ordered by created_at DESC."""
        rows = await self._pool.fetch(
            "SELECT * FROM application_context ORDER BY created_at DESC"
        )
        return [_row_to_context(r) for r in rows]

    async def get_impact(self) -> ContextImpactResult:
        """
        Return task distribution across context versions.

        Tasks whose context_version matches the active context are counted as
        'on current version'. All others (including NULL context_version) are
        counted as 'on older versions'.
        """
        active = await self.get_active()
        active_version = active.version if active else None

        total: int = await self._pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE status != 'deleted'"
        )

        if active_version is None:
            # No context exists — all tasks are on "older" (no) version
            older_ids_rows = await self._pool.fetch(
                "SELECT task_id FROM tasks WHERE status != 'deleted'"
            )
            older_ids = [r["task_id"] for r in older_ids_rows]
            return ContextImpactResult(
                total_tasks=total,
                tasks_on_current_version=0,
                tasks_on_older_versions=total,
                older_version_task_ids=older_ids,
            )

        current_count: int = await self._pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE status != 'deleted' AND context_version = $1",
            active_version,
        )

        older_ids_rows = await self._pool.fetch(
            "SELECT task_id FROM tasks WHERE status != 'deleted' AND (context_version IS NULL OR context_version != $1)",
            active_version,
        )
        older_ids = [r["task_id"] for r in older_ids_rows]

        return ContextImpactResult(
            total_tasks=total,
            tasks_on_current_version=current_count,
            tasks_on_older_versions=total - current_count,
            older_version_task_ids=older_ids,
        )
