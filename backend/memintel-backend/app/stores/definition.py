"""
app/stores/definition.py
──────────────────────────────────────────────────────────────────────────────
DefinitionStore — asyncpg-backed persistence for the `definitions` table.

Immutability contract
─────────────────────
Once registered, a (definition_id, version) pair is permanent. The DB unique
constraint `uq_definition_version` is the final guard. register() also catches
asyncpg.UniqueViolationError and converts it to ConflictError → HTTP 409.

The store does NOT validate definitions. Callers (compiler, registry service)
must run the definition through the compiler before calling register(). This
store only persists pre-validated bodies.

Deprecation
───────────
deprecated() marks a version advisory-deprecated. Deprecated definitions
remain fully resolvable — the runtime never auto-rejects them. The
replacement_version field is informational only.

Promotion
─────────
promote() copies a definition from one namespace to another by inserting a
new row with the target namespace. Promoting to 'global' requires
elevated_key=True; the store raises HTTP 403 otherwise.

Column ↔ field mapping
──────────────────────
DB column           Python field
──────────────────  ──────────────────────────────────────────
id                  (internal BIGSERIAL — never exposed)
definition_id       definition_id
version             version
definition_type     definition_type
namespace           namespace
body                body (JSONB — raw dict)
meaning_hash        meaning_hash  (concepts only)
ir_hash             ir_hash       (concepts only)
deprecated          deprecated
deprecated_at       deprecated_at
replacement_version replacement_version
created_at          created_at
updated_at          updated_at
"""
from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from app.models.action import ActionDefinition
from app.models.concept import DefinitionResponse, SearchResult, VersionSummary
from app.models.errors import ConflictError, ErrorType, MemintelError
from app.models.task import Namespace

log = logging.getLogger(__name__)

#: Valid definition_type values — matches the DB CHECK constraint.
#: Migration 0005 added 'feature' to the constraint; keep in sync.
VALID_DEFINITION_TYPES: frozenset[str] = frozenset({
    "concept", "condition", "action", "primitive", "feature",
})

#: Namespaces that require elevated privileges to promote into.
ELEVATED_NAMESPACES: frozenset[str] = frozenset({"global"})


class DefinitionStore:
    """Async store for the `definitions` table."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── register ──────────────────────────────────────────────────────────────

    async def register(
        self,
        definition_id: str,
        version: str,
        definition_type: str,
        namespace: str,
        body: dict[str, Any],
        meaning_hash: str | None = None,
        ir_hash: str | None = None,
    ) -> DefinitionResponse:
        """
        Persist a validated definition and return a DefinitionResponse.

        Raises ConflictError if (definition_id, version) is already registered.
        The caller is responsible for compiler validation before calling this.
        """
        try:
            row = await self._pool.fetchrow(
                """
                INSERT INTO definitions (
                    definition_id, version, definition_type, namespace,
                    body, meaning_hash, ir_hash
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING
                    definition_id, version, definition_type, namespace,
                    meaning_hash, ir_hash,
                    deprecated, deprecated_at, replacement_version,
                    created_at, updated_at
                """,
                definition_id,
                version,
                definition_type,
                namespace,
                json.dumps(body),
                meaning_hash,
                ir_hash,
            )
        except asyncpg.UniqueViolationError:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' is already registered. "
                "Definitions are immutable — create a new version instead.",
                location=f"{definition_id}:{version}",
            )

        return _row_to_definition_response(row)

    # ── get ───────────────────────────────────────────────────────────────────

    async def get(self, definition_id: str, version: str) -> dict[str, Any] | None:
        """
        Return the raw body dict for a definition, or None if not found.

        Route handlers convert None → HTTP 404. Deprecated definitions are
        returned — callers that need to reject deprecated versions must check
        the definition's deprecated flag themselves.
        """
        row = await self._pool.fetchrow(
            """
            SELECT body
            FROM definitions
            WHERE definition_id = $1 AND version = $2
            """,
            definition_id,
            version,
        )
        if row is None:
            return None
        raw = row["body"]
        return json.loads(raw) if isinstance(raw, str) else raw

    # ── get_metadata ──────────────────────────────────────────────────────────

    async def get_metadata(
        self, definition_id: str, version: str
    ) -> DefinitionResponse | None:
        """
        Return the full DefinitionResponse (without body) for a definition.

        Used when the caller needs deprecation/hash metadata but not the body.
        """
        row = await self._pool.fetchrow(
            """
            SELECT
                definition_id, version, definition_type, namespace,
                meaning_hash, ir_hash,
                deprecated, deprecated_at, replacement_version,
                created_at, updated_at
            FROM definitions
            WHERE definition_id = $1 AND version = $2
            """,
            definition_id,
            version,
        )
        return _row_to_definition_response(row) if row else None

    # ── versions ──────────────────────────────────────────────────────────────

    async def versions(self, definition_id: str) -> list[VersionSummary]:
        """
        Return all versions of a definition, newest-first (by created_at DESC).

        Version strings are NOT lexicographically sorted — ordering is always
        by insertion time.
        """
        rows = await self._pool.fetch(
            """
            SELECT version, created_at, deprecated, meaning_hash, ir_hash
            FROM definitions
            WHERE definition_id = $1
            ORDER BY created_at DESC
            """,
            definition_id,
        )
        return [
            VersionSummary(
                version=r["version"],
                created_at=r["created_at"],
                deprecated=r["deprecated"],
                meaning_hash=r["meaning_hash"],
                ir_hash=r["ir_hash"],
            )
            for r in rows
        ]

    # ── list ──────────────────────────────────────────────────────────────────

    async def list(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        """
        Return a cursor-paginated list of definitions.

        Filters by definition_type and/or namespace when provided.
        cursor is the definition_id of the last item seen on the previous page.
        limit is capped at 100.
        """
        limit = min(limit, 100)

        conditions: list[str] = []
        args: list[Any] = []
        arg_idx = 1

        if definition_type is not None:
            conditions.append(f"definition_type = ${arg_idx}")
            args.append(definition_type)
            arg_idx += 1

        if namespace is not None:
            conditions.append(f"namespace = ${arg_idx}")
            args.append(namespace)
            arg_idx += 1

        if cursor is not None:
            conditions.append(
                f"created_at < ("
                f"SELECT created_at FROM definitions WHERE definition_id = ${arg_idx} "
                f"ORDER BY created_at DESC LIMIT 1)"
            )
            args.append(cursor)
            arg_idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        fetch_limit = limit + 1
        args.append(fetch_limit)

        rows = await self._pool.fetch(
            f"""
            SELECT
                definition_id, version, definition_type, namespace,
                meaning_hash, ir_hash,
                deprecated, deprecated_at, replacement_version,
                created_at, updated_at
            FROM definitions
            {where}
            ORDER BY created_at DESC
            LIMIT ${arg_idx}
            """,
            *args,
        )

        has_more = len(rows) > limit
        page = rows[:limit]
        items = [_row_to_definition_response(r) for r in page]
        next_cursor = items[-1].definition_id if has_more and items else None

        # Total count (same filter, no cursor/limit)
        count_conditions = [c for c in conditions if "created_at <" not in c]
        count_where = "WHERE " + " AND ".join(count_conditions) if count_conditions else ""
        count_args = args[: len(count_conditions)]
        total_count = await self._pool.fetchval(
            f"SELECT COUNT(*) FROM definitions {count_where}", *count_args
        )

        return SearchResult(
            items=items,
            has_more=has_more,
            next_cursor=next_cursor,
            total_count=total_count or 0,
        )

    # ── list_actions ──────────────────────────────────────────────────────────

    async def list_actions(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ActionDefinition]:
        """
        Return up to ``limit`` non-deprecated ActionDefinitions for ``namespace``.

        Uses a single SELECT that fetches the body column to avoid N+1 queries.
        Rows whose body cannot be parsed into ActionDefinition are logged and
        skipped — the list never fails due to a single corrupted row.

        Results are ordered newest-first (created_at DESC).
        """
        rows = await self._pool.fetch(
            """
            SELECT definition_id, definition_type, version,
                   namespace, body, created_at, deprecated
            FROM definitions
            WHERE definition_type = 'action'
              AND namespace = $1
              AND deprecated = FALSE
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            namespace,
            limit,
            offset,
        )

        results: list[ActionDefinition] = []
        for row in rows:
            try:
                raw = row["body"]
                body = json.loads(raw) if isinstance(raw, str) else raw
                results.append(ActionDefinition.model_validate(body))
            except Exception as exc:
                log.warning(
                    "list_actions_skip_invalid_row",
                    extra={
                        "definition_id": row["definition_id"],
                        "error": str(exc),
                    },
                )
        return results

    # ── count_actions ─────────────────────────────────────────────────────────

    async def count_actions(self, namespace: str) -> int:
        """
        Return the total number of non-deprecated actions for ``namespace``.

        Used by GET /actions to populate the ActionListResponse.total field
        with the DB total, independent of the page limit/offset.
        """
        count = await self._pool.fetchval(
            """
            SELECT COUNT(*)
            FROM definitions
            WHERE definition_type = 'action'
              AND namespace = $1
              AND deprecated = FALSE
            """,
            namespace,
        )
        return int(count or 0)

    # ── deprecate ─────────────────────────────────────────────────────────────

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
        reason: str,
    ) -> DefinitionResponse:
        """
        Mark a definition version as deprecated.

        Raises NotFoundError if the definition does not exist.
        Deprecation is idempotent — deprecating an already-deprecated version
        updates the replacement_version if a new one is provided.
        """
        row = await self._pool.fetchrow(
            """
            UPDATE definitions
            SET
                deprecated = TRUE,
                deprecated_at = NOW(),
                replacement_version = COALESCE($3, replacement_version),
                updated_at = NOW()
            WHERE definition_id = $1 AND version = $2
            RETURNING
                definition_id, version, definition_type, namespace,
                meaning_hash, ir_hash,
                deprecated, deprecated_at, replacement_version,
                created_at, updated_at
            """,
            definition_id,
            version,
            replacement_version,
        )
        if row is None:
            raise MemintelError(
                ErrorType.NOT_FOUND,
                f"Definition '{definition_id}' version '{version}' not found.",
                location=f"{definition_id}:{version}",
            )
        log.info(
            "definition_deprecated",
            extra={
                "definition_id": definition_id,
                "version": version,
                "replacement_version": replacement_version,
                "reason": reason,
            },
        )
        return _row_to_definition_response(row)

    # ── promote ───────────────────────────────────────────────────────────────

    async def promote(
        self,
        definition_id: str,
        version: str,
        from_namespace: str,
        to_namespace: str,
        elevated_key: bool = False,
    ) -> DefinitionResponse:
        """
        Copy a definition from one namespace to another.

        Promotion inserts a new row with the target namespace — the original
        row is retained unchanged. Promoting to 'global' requires elevated_key=True.

        Raises:
          MemintelError(AUTH_ERROR)   — promoting to 'global' without elevated_key.
          MemintelError(NOT_FOUND)    — source definition does not exist.
          ConflictError               — target (definition_id, version) already exists
                                        in to_namespace (should not happen in normal flow).
        """
        if to_namespace in ELEVATED_NAMESPACES and not elevated_key:
            raise MemintelError(
                ErrorType.AUTH_ERROR,
                f"Promoting to namespace '{to_namespace}' requires elevated privileges.",
                suggestion="Pass elevated_key=True with a valid admin API key.",
            )

        # Fetch the source row body and metadata
        source = await self._pool.fetchrow(
            """
            SELECT
                definition_id, version, definition_type,
                body, meaning_hash, ir_hash
            FROM definitions
            WHERE definition_id = $1 AND version = $2 AND namespace = $3
            """,
            definition_id,
            version,
            from_namespace,
        )
        if source is None:
            raise MemintelError(
                ErrorType.NOT_FOUND,
                f"Definition '{definition_id}' version '{version}' "
                f"not found in namespace '{from_namespace}'.",
                location=f"{definition_id}:{version}",
            )

        try:
            row = await self._pool.fetchrow(
                """
                INSERT INTO definitions (
                    definition_id, version, definition_type, namespace,
                    body, meaning_hash, ir_hash
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING
                    definition_id, version, definition_type, namespace,
                    meaning_hash, ir_hash,
                    deprecated, deprecated_at, replacement_version,
                    created_at, updated_at
                """,
                source["definition_id"],
                source["version"],
                source["definition_type"],
                to_namespace,
                source["body"],
                source["meaning_hash"],
                source["ir_hash"],
            )
        except asyncpg.UniqueViolationError:
            raise ConflictError(
                f"Definition '{definition_id}' version '{version}' "
                f"already exists in namespace '{to_namespace}'.",
                location=f"{definition_id}:{version}",
            )

        log.info(
            "definition_promoted",
            extra={
                "definition_id": definition_id,
                "version": version,
                "from_namespace": from_namespace,
                "to_namespace": to_namespace,
            },
        )
        return _row_to_definition_response(row)


# ── Row mapping helper ────────────────────────────────────────────────────────

def _row_to_definition_response(row: asyncpg.Record) -> DefinitionResponse:
    """Convert an asyncpg Record from the definitions table into a DefinitionResponse."""
    return DefinitionResponse(
        definition_id=row["definition_id"],
        version=row["version"],
        definition_type=row["definition_type"],
        namespace=Namespace(row["namespace"]),
        meaning_hash=row["meaning_hash"],
        ir_hash=row["ir_hash"],
        deprecated=row["deprecated"],
        deprecated_at=row["deprecated_at"],
        replacement_version=row["replacement_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
