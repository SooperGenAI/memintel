"""
app/stores/guardrails.py
────────────────────────────────────────────────────────────────────────────────
GuardrailsStore — asyncpg-backed persistence for the `guardrails_versions` table.

Invariants
──────────
  - Only one row has is_active = TRUE at any time.
  - create() deactivates the previous active version atomically in a transaction.
  - Versions are never deleted — only superseded.
  - version is auto-assigned: "v1" for the first row, "v{n+1}" for each subsequent.
  - get_active() returns None when no version exists (never raises).

Column ↔ field mapping
──────────────────────
DB column             Python field
────────────────────  ─────────────────────────────────────
guardrails_id         version.guardrails_id   (UUID PRIMARY KEY)
version               version.version         (VARCHAR 10)
guardrails_json       version.guardrails      (GuardrailsDefinition, JSONB)
change_note           version.change_note     (TEXT nullable)
created_at            version.created_at
is_active             version.is_active
source                version.source          (VARCHAR 10)
"""
from __future__ import annotations

import json
import uuid

import asyncpg

from app.models.guardrails_api import GuardrailsDefinition, GuardrailsImpactResult, GuardrailsVersion


def _row_to_version(row: asyncpg.Record) -> GuardrailsVersion:
    """Convert a DB row from guardrails_versions into a GuardrailsVersion."""
    return GuardrailsVersion.model_validate({
        "guardrails_id": str(row["guardrails_id"]),
        "version":       row["version"],
        "guardrails":    json.loads(row["guardrails_json"]),
        "change_note":   row["change_note"],
        "created_at":    row["created_at"],
        "is_active":     row["is_active"],
        "source":        row["source"],
    })


class GuardrailsStore:
    """
    Persistence layer for GuardrailsVersion.

    Public API
    ----------
    create(version)             → GuardrailsVersion
        Persist a new version, deactivating the previous one atomically.
        Assigns version ("v1", "v2", …) before inserting.

    get_active()                → GuardrailsVersion | None
        Return the current active version, or None. Never raises.

    get_version(version)        → GuardrailsVersion | None
        Return a specific version by version string, or None.

    list_versions()             → list[GuardrailsVersion]
        Return all versions, newest first.

    get_impact()                → GuardrailsImpactResult
        Return task distribution across guardrails versions.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, guardrails: GuardrailsVersion) -> GuardrailsVersion:
        """
        Persist ``guardrails`` as the new active version.

        Steps (all in one transaction):
          1. Count existing rows to determine the next version number.
          2. Set version on guardrails ("v{n+1}").
          3. Set is_active = FALSE on the current active row (if any).
          4. INSERT the new row with is_active = TRUE.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                count: int = await conn.fetchval(
                    "SELECT COUNT(*) FROM guardrails_versions"
                )
                guardrails.version = f"v{count + 1}"

                # Deactivate previous active version
                await conn.execute(
                    "UPDATE guardrails_versions SET is_active = FALSE WHERE is_active = TRUE"
                )

                await conn.execute(
                    """
                    INSERT INTO guardrails_versions (
                        guardrails_id, version, guardrails_json, change_note,
                        created_at, is_active, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    uuid.UUID(guardrails.guardrails_id),
                    guardrails.version,
                    guardrails.guardrails.model_dump_json(),
                    guardrails.change_note,
                    guardrails.created_at,
                    True,
                    guardrails.source,
                )

        return guardrails

    async def get_active(self) -> GuardrailsVersion | None:
        """Return the active guardrails version, or None if none exists."""
        row = await self._pool.fetchrow(
            "SELECT * FROM guardrails_versions WHERE is_active = TRUE LIMIT 1"
        )
        if row is None:
            return None
        return _row_to_version(row)

    async def get_version(self, version: str) -> GuardrailsVersion | None:
        """Return the guardrails version with the given version string, or None."""
        row = await self._pool.fetchrow(
            "SELECT * FROM guardrails_versions WHERE version = $1",
            version,
        )
        if row is None:
            return None
        return _row_to_version(row)

    async def list_versions(self) -> list[GuardrailsVersion]:
        """Return all guardrails versions ordered by created_at DESC."""
        rows = await self._pool.fetch(
            "SELECT * FROM guardrails_versions ORDER BY created_at DESC"
        )
        return [_row_to_version(r) for r in rows]

    async def get_impact(self) -> GuardrailsImpactResult:
        """
        Return task distribution across guardrails versions.

        Tasks whose guardrails_version matches the active version are counted as
        'on current version'. All others (including NULL guardrails_version) are
        counted as 'on older versions'.
        """
        active = await self.get_active()
        active_version = active.version if active else None

        total: int = await self._pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE status != 'deleted'"
        )

        if active_version is None:
            # No guardrails API version exists — all tasks are on "older" (no) version
            older_ids_rows = await self._pool.fetch(
                "SELECT task_id FROM tasks WHERE status != 'deleted'"
            )
            older_ids = [r["task_id"] for r in older_ids_rows]
            return GuardrailsImpactResult(
                total_tasks=total,
                tasks_on_current_version=0,
                tasks_on_older_guardrails_version=total,
                older_version_task_ids=older_ids,
            )

        current_count: int = await self._pool.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE status != 'deleted' AND guardrails_version = $1",
            active_version,
        )

        older_ids_rows = await self._pool.fetch(
            "SELECT task_id FROM tasks WHERE status != 'deleted'"
            " AND (guardrails_version IS NULL OR guardrails_version != $1)",
            active_version,
        )
        older_ids = [r["task_id"] for r in older_ids_rows]

        return GuardrailsImpactResult(
            total_tasks=total,
            tasks_on_current_version=current_count,
            tasks_on_older_guardrails_version=total - current_count,
            older_version_task_ids=older_ids,
        )
