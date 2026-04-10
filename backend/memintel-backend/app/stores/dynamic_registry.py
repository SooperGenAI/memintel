"""
app/stores/dynamic_registry.py
──────────────────────────────────────────────────────────────────────────────
DynamicRegistryStore — asyncpg-backed persistence for dynamically registered
connectors and primitives.

Tables:  registered_connectors, registered_primitives
         (created by migration 0013_add_dynamic_registrations)

Connector params are stored encrypted (Fernet).  This store writes/reads the
raw ciphertext; callers are responsible for encrypt/decrypt via
app.utils.encryption.
"""
from __future__ import annotations

import asyncpg


class DynamicRegistryStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Connectors ─────────────────────────────────────────────────────────────

    async def create_connector(
        self,
        name: str,
        connector_type: str,
        params_encrypted: str,
    ) -> dict:
        """
        Insert a new connector row.  Raises asyncpg.UniqueViolationError if
        a connector with the same name already exists.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO registered_connectors (name, connector_type, params_encrypted)
                VALUES ($1, $2, $3)
                RETURNING id, name, connector_type, created_at
                """,
                name,
                connector_type,
                params_encrypted,
            )
            return dict(row)

    async def get_connector(self, name: str) -> dict | None:
        """Return the full row (including params_encrypted) or None."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, connector_type, params_encrypted, created_at
                FROM registered_connectors
                WHERE name = $1
                """,
                name,
            )
            return dict(row) if row else None

    async def list_connectors(self) -> list[dict]:
        """Return all connectors (no params_encrypted — metadata only)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, connector_type, created_at
                FROM registered_connectors
                ORDER BY created_at
                """
            )
            return [dict(r) for r in rows]

    async def list_connectors_with_params(self) -> list[dict]:
        """
        Return all connectors including params_encrypted.

        Used at startup to reconstruct live connector instances from DB.
        Do NOT expose this in API responses.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, connector_type, params_encrypted, created_at
                FROM registered_connectors
                ORDER BY created_at
                """
            )
            return [dict(r) for r in rows]

    async def delete_connector(self, name: str) -> bool:
        """Delete by name.  Returns True if a row was deleted, False if not found."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM registered_connectors WHERE name = $1",
                name,
            )
            # asyncpg returns 'DELETE N' — extract the count
            return int(result.split()[-1]) > 0

    # ── Primitives ─────────────────────────────────────────────────────────────

    async def create_primitive(
        self,
        name: str,
        primitive_type: str,
        connector_name: str | None,
        query: str | None,
        json_path: str | None,
    ) -> dict:
        """
        Insert a new primitive row.  Raises asyncpg.UniqueViolationError if
        a primitive with the same name already exists.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO registered_primitives
                    (name, primitive_type, connector_name, query, json_path)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, name, primitive_type, connector_name, query, json_path, created_at
                """,
                name,
                primitive_type,
                connector_name,
                query,
                json_path,
            )
            return dict(row)

    async def list_primitives(self) -> list[dict]:
        """Return all registered primitives."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, primitive_type, connector_name, query, json_path, created_at
                FROM registered_primitives
                ORDER BY created_at
                """
            )
            return [dict(r) for r in rows]

    async def list_primitives_for_connector(self, connector_name: str) -> list[dict]:
        """Return all primitives that reference the named connector."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, primitive_type, connector_name, query, json_path, created_at
                FROM registered_primitives
                WHERE connector_name = $1
                ORDER BY created_at
                """,
                connector_name,
            )
            return [dict(r) for r in rows]

    async def delete_primitive(self, name: str) -> bool:
        """Delete by name.  Returns True if a row was deleted, False if not found."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM registered_primitives WHERE name = $1",
                name,
            )
            return int(result.split()[-1]) > 0
