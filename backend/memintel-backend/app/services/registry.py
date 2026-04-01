"""
app/services/registry.py
──────────────────────────────────────────────────────────────────────────────
RegistryService — definition lifecycle management.

Manages the full lifecycle of definitions (concepts, conditions, actions,
primitives) in the definitions table:
  - Register (create or update) definitions
  - List and search with cursor-based pagination
  - Version history and lineage traversal
  - Semantic diff between versions
  - Deprecation and namespace promotion

Also manages the feature registry (registered_features table):
  - Register features with semantic deduplication
  - Get feature by id
  - List usages (definitions that depend on a feature)

Architecture
────────────
RegistryService wraps two sub-layers:
  DefinitionStore    — raw asyncpg persistence (app/stores/definition.py)
  DefinitionRegistry — governance logic: freeze checks, promotion rules,
                       semantic diff (app/registry/definitions.py)

Feature storage uses DefinitionStore with definition_type='feature'.
"""
from __future__ import annotations

import hashlib
import json
import structlog
from typing import Any

import asyncpg

from app.models.concept import (
    DefinitionResponse,
    LineageResult,
    SearchResult,
    SemanticDiffResult,
    VersionSummary,
)
from app.models.errors import ConflictError, NotFoundError
from app.registry.definitions import DefinitionRegistry
from app.stores.definition import DefinitionStore, _row_to_definition_response

log = structlog.get_logger(__name__)


class RegistryService:
    """
    Manages definition registration, versioning, and lifecycle.

    register()           — creates or returns existing definition; HTTP 409 on
                           body mismatch (immutability).
    list_definitions()   — paginated list with optional type/namespace filter.
    search()             — keyword search over definition_id and namespace.
    find_similar()       — finds definitions sharing the same meaning_hash.
    get_versions()       — returns VersionSummary list for a definition.
    get_lineage()        — returns LineageResult (version chain).
    get_semantic_diff()  — compares two versions; returns SemanticDiffResult.
    deprecate()          — marks a version deprecated; returns DefinitionResponse.
    promote()            — promotes to a higher namespace; returns DefinitionResponse.

    register_feature()   — registers a feature; returns FeatureRegistrationResult.
    get_feature()        — returns RegisteredFeature by id.
    get_feature_usages() — returns UsageResult for a feature.
    list_features()      — returns FeatureSearchResult.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._store = DefinitionStore(pool)
        self._registry = DefinitionRegistry(store=self._store)

    # ── register ──────────────────────────────────────────────────────────────

    async def register(self, req: Any) -> DefinitionResponse:
        """
        Register a definition and return a DefinitionResponse.

        Idempotent: if (definition_id, version) already exists with an identical
        body the existing record is returned unchanged (HTTP 200).
        Raises ConflictError (HTTP 409) if the body differs — immutability
        violation.
        """
        try:
            return await self._store.register(
                definition_id=req.definition_id,
                version=req.version,
                definition_type=req.definition_type,
                namespace=req.namespace,
                body=req.body,
            )
        except ConflictError:
            existing_body = await self._store.get(req.definition_id, req.version)
            if existing_body == req.body:
                meta = await self._store.get_metadata(req.definition_id, req.version)
                if meta is not None:
                    return meta
            raise

    # ── list_definitions ──────────────────────────────────────────────────────

    async def list_definitions(
        self,
        definition_type: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        return await self._store.list(
            definition_type=definition_type,
            namespace=namespace,
            limit=limit,
            cursor=cursor,
        )

    # ── search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        q: str,
        definition_type: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchResult:
        """
        Keyword search across definition_id and namespace columns (ILIKE).

        cursor is the definition_id of the last item seen on the previous page.
        """
        limit = min(limit, 100)
        pattern = f"%{q}%"

        conditions: list[str] = ["(definition_id ILIKE $1 OR namespace ILIKE $1)"]
        args: list[Any] = [pattern]
        arg_idx = 2

        if definition_type is not None:
            conditions.append(f"definition_type = ${arg_idx}")
            args.append(definition_type)
            arg_idx += 1

        if cursor is not None:
            conditions.append(
                f"created_at < ("
                f"SELECT created_at FROM definitions "
                f"WHERE definition_id = ${arg_idx} "
                f"ORDER BY created_at DESC LIMIT 1)"
            )
            args.append(cursor)
            arg_idx += 1

        where = "WHERE " + " AND ".join(conditions)
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

        # Count: same filter but without the cursor condition
        count_conditions = [c for c in conditions if "created_at <" not in c]
        count_where = "WHERE " + " AND ".join(count_conditions) if count_conditions else ""
        count_args = args[: len(count_conditions)]
        total_count = await self._pool.fetchval(
            f"SELECT COUNT(*) FROM definitions {count_where}",
            *count_args,
        )

        return SearchResult(
            items=items,
            has_more=has_more,
            next_cursor=next_cursor,
            total_count=total_count or 0,
        )

    # ── find_similar ──────────────────────────────────────────────────────────

    async def find_similar(self, req: Any) -> SearchResult:
        """
        Return definitions that share the same meaning_hash as (definition_id, version).

        Returns an empty SearchResult when the source definition has no meaning_hash
        (non-concept types typically have no hash).
        Raises NotFoundError if the source definition does not exist.
        """
        meta = await self._store.get_metadata(req.definition_id, req.version)
        if meta is None:
            raise NotFoundError(
                f"Definition '{req.definition_id}' version '{req.version}' not found.",
                location=f"{req.definition_id}:{req.version}",
            )

        if meta.meaning_hash is None:
            return SearchResult(items=[], has_more=False, next_cursor=None, total_count=0)

        limit = getattr(req, "limit", 10)
        rows = await self._pool.fetch(
            """
            SELECT
                definition_id, version, definition_type, namespace,
                meaning_hash, ir_hash,
                deprecated, deprecated_at, replacement_version,
                created_at, updated_at
            FROM definitions
            WHERE meaning_hash = $1
              AND NOT (definition_id = $2 AND version = $3)
            ORDER BY created_at DESC
            LIMIT $4
            """,
            meta.meaning_hash,
            req.definition_id,
            req.version,
            limit,
        )

        items = [_row_to_definition_response(r) for r in rows]
        return SearchResult(
            items=items,
            has_more=False,
            next_cursor=None,
            total_count=len(items),
        )

    # ── get_versions ──────────────────────────────────────────────────────────

    async def get_versions(self, definition_id: str) -> list[VersionSummary]:
        """Return all versions newest-first. Returns empty list if none found."""
        return await self._store.versions(definition_id)

    # ── get_lineage ───────────────────────────────────────────────────────────

    async def get_lineage(self, definition_id: str) -> LineageResult:
        """
        Return the version chain for a definition as a LineageResult.

        promoted_to is not currently tracked at the DB level — returned as
        an empty dict. Version chain is ordered newest-first.

        Raises NotFoundError if the definition has no registered versions.
        """
        summaries = await self._store.versions(definition_id)
        if not summaries:
            raise NotFoundError(
                f"No versions found for definition '{definition_id}'.",
                location=definition_id,
            )
        return LineageResult(
            definition_id=definition_id,
            chain=summaries,
            promoted_to={},
        )

    # ── get_semantic_diff ─────────────────────────────────────────────────────

    async def get_semantic_diff(
        self,
        definition_id: str,
        version_from: str,
        version_to: str,
    ) -> SemanticDiffResult:
        """Delegate to DefinitionRegistry which owns the diff logic."""
        return await self._registry.semantic_diff(definition_id, version_from, version_to)

    # ── deprecate ─────────────────────────────────────────────────────────────

    async def deprecate(
        self,
        definition_id: str,
        version: str,
        replacement_version: str | None,
    ) -> DefinitionResponse:
        return await self._store.deprecate(
            definition_id=definition_id,
            version=version,
            replacement_version=replacement_version,
            reason="",
        )

    # ── promote ───────────────────────────────────────────────────────────────

    async def promote(
        self,
        definition_id: str,
        version: str,
        target_namespace: str,
    ) -> DefinitionResponse:
        """
        Promote a definition to a higher namespace.

        from_namespace is resolved from the stored record — the caller only
        supplies the target. DefinitionRegistry enforces the promotion path
        (personal → team → org → global) and blocks on breaking semantic diffs.

        Raises NotFoundError if the definition does not exist.
        """
        meta = await self._store.get_metadata(definition_id, version)
        if meta is None:
            raise NotFoundError(
                f"Definition '{definition_id}' version '{version}' not found.",
                location=f"{definition_id}:{version}",
            )
        return await self._registry.promote(
            definition_id=definition_id,
            version=version,
            from_namespace=meta.namespace.value,
            to_namespace=target_namespace,
        )

    # ── register_feature ──────────────────────────────────────────────────────

    async def register_feature(self, req: Any) -> dict[str, Any]:
        """
        Register a feature. semantic_hash is computed from the body.

        Deduplication: if an existing feature already has the same semantic_hash,
        deduplicated=True is returned and the canonical feature_id is used.

        Returns a FeatureRegistrationResult-compatible dict.
        """
        semantic_hash = _compute_body_hash(req.body)

        # Scan for hash collisions
        existing = await self._store.list(definition_type="feature")
        duplicate_id = next(
            (
                item.definition_id
                for item in existing.items
                if item.meaning_hash == semantic_hash
                and item.definition_id != req.feature_id
            ),
            None,
        )

        if duplicate_id is not None:
            return {
                "feature_id": duplicate_id,
                "version": req.version,
                "semantic_hash": semantic_hash,
                "deduplicated": True,
            }

        await self._store.register(
            definition_id=req.feature_id,
            version=req.version,
            definition_type="feature",
            namespace="org",
            body=req.body,
            meaning_hash=semantic_hash,
        )

        return {
            "feature_id": req.feature_id,
            "version": req.version,
            "semantic_hash": semantic_hash,
            "deduplicated": False,
        }

    # ── get_feature ───────────────────────────────────────────────────────────

    async def get_feature(self, feature_id: str) -> dict[str, Any]:
        """
        Return a feature by feature_id (most recent version).

        Raises NotFoundError if no feature with that id exists.
        Returns a RegisteredFeature-compatible dict.
        """
        row = await self._pool.fetchrow(
            """
            SELECT definition_id, version, meaning_hash, body, created_at
            FROM definitions
            WHERE definition_id = $1 AND definition_type = 'feature'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            feature_id,
        )
        if row is None:
            raise NotFoundError(
                f"Feature '{feature_id}' not found.",
                location=feature_id,
            )
        raw = row["body"]
        body = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return {
            "feature_id": row["definition_id"],
            "version": row["version"],
            "semantic_hash": row["meaning_hash"] or "",
            "body": body,
            "created_at": str(row["created_at"]) if row["created_at"] else None,
        }

    # ── get_feature_usages ────────────────────────────────────────────────────

    async def get_feature_usages(self, feature_id: str) -> dict[str, Any]:
        """
        Return all definitions that reference the given feature.

        Verifies the feature exists and returns a UsageResult-compatible dict.
        Usage scanning across definition bodies is not yet implemented —
        returns an empty usages list.
        """
        exists = await self._pool.fetchval(
            "SELECT 1 FROM definitions WHERE definition_id = $1 AND definition_type = 'feature'",
            feature_id,
        )
        if not exists:
            raise NotFoundError(
                f"Feature '{feature_id}' not found.",
                location=feature_id,
            )
        return {
            "feature_id": feature_id,
            "usages": [],
            "total_count": 0,
        }

    # ── list_features ─────────────────────────────────────────────────────────

    async def list_features(
        self,
        q: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """
        List features with optional keyword filter on feature_id.

        Returns a FeatureSearchResult-compatible dict.
        """
        rows = await self._pool.fetch(
            """
            SELECT definition_id, version, meaning_hash, body, created_at
            FROM definitions
            WHERE definition_type = 'feature'
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit + 1,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]

        if q:
            rows = [r for r in rows if q.lower() in r["definition_id"].lower()]

        feature_items = [
            {
                "feature_id": row["definition_id"],
                "version": row["version"],
                "semantic_hash": row["meaning_hash"] or "",
                "body": (
                    json.loads(row["body"])
                    if isinstance(row["body"], str)
                    else (row["body"] or {})
                ),
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            }
            for row in rows
        ]

        return {
            "items": feature_items,
            "has_more": has_more,
            "next_cursor": None,
            "total_count": len(feature_items),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_body_hash(body: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a raw body dict (used for feature deduplication)."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
