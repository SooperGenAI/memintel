"""
app/connectors/postgres.py
──────────────────────────────────────────────────────────────────────────────
PostgresConnector — asyncpg-backed primitive data connector.

Implements async fetch() for SQL-configured primitives. Queries are configured
per-primitive in memintel_config.yaml primitives: section and resolved through
PrimitiveSourceConfig at construction time.

SQL placeholder convention:
  :entity_id  → $1 (asyncpg positional parameter)
  :as_of      → $2 (asyncpg positional parameter)

Forward/backward fill limitation:
  The day-walking fill strategies are a simplification. Production SQL queries
  should use the pattern:
    SELECT ... WHERE event_date <= :as_of ORDER BY event_date DESC LIMIT 1
  directly in their configured query, rather than relying on day-walking retries.
  The day-walking approach is O(N_missing_days) and should only be used when
  the underlying query cannot express a range search.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from app.models.config import ConnectorConfig, PrimitiveSourceConfig
from app.runtime.data_resolver import ConnectorError, PrimitiveValue

log = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
_MAX_FILL_DAYS = 90


def _resolve_env_var_str(value: str) -> str:
    """Resolve a single ${ENV_VAR} reference. Returns value unchanged if not a reference."""
    import os
    m = _ENV_VAR_RE.fullmatch(value)
    if m:
        resolved = os.environ.get(m.group(1))
        if resolved is None:
            raise ConnectorError(f"Environment variable {m.group(1)!r} is not set")
        return resolved
    return value


def _translate_placeholders(query: str) -> str:
    """Translate :entity_id → $1 and :as_of → $2 for asyncpg positional params."""
    query = query.replace(':entity_id', '$1')
    query = query.replace(':as_of', '$2')
    return query


class PostgresConnector:
    """
    asyncpg-backed connector for SQL-configured primitives.

    Connection is lazy — the pool is created on the first fetch() call.
    primitive_sources maps primitive_name → PrimitiveSourceConfig (connector + query).
    Only primitives present in primitive_sources can be fetched; others raise ConnectorError.

    All methods are async. This connector does NOT implement the sync ConnectorBase
    interface — use it via DataResolver.afetch() or test directly with asyncio.run().
    """

    def __init__(
        self,
        config: ConnectorConfig,
        primitive_sources: dict[str, PrimitiveSourceConfig],
    ) -> None:
        self._host = config.host or "localhost"
        self._port = config.port or 5432
        self._database = config.database or ""
        self._user = config.user or ""
        # Resolve password from env var if it is an ${ENV_VAR} reference
        raw_password = config.password or ""
        self._password = _resolve_env_var_str(raw_password) if raw_password else ""
        self._pool_min = max(1, config.pool_min)
        self._pool_max = max(1, config.pool_max)
        self._primitive_sources = primitive_sources
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """
        Create the asyncpg connection pool.

        Called lazily on the first fetch() call. May also be called explicitly
        at startup (e.g. for health checks). Raises ConnectorError if the
        database is unreachable or credentials are wrong.
        """
        try:
            dsn = (
                f"postgresql://{self._user}:{self._password}"
                f"@{self._host}:{self._port}/{self._database}"
            )
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
        except Exception as exc:
            raise ConnectorError(f"PostgresConnector: connection failed: {exc}") from exc

    async def _ensure_connected(self) -> None:
        if self._pool is None:
            await self.connect()

    async def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """
        Fetch the point-in-time value of primitive_name for entity_id.

        Translates :entity_id → $1 and :as_of → $2 in the configured SQL query.
        Returns PrimitiveValue(value=None) when the query returns no rows.
        Returns PrimitiveValue(value=row[0]) for single and multi-row results;
        logs a warning when multiple rows are returned.

        Raises ConnectorError when:
          - primitive_name has no configured query in primitive_sources
          - the database query fails for any reason
        """
        source = self._primitive_sources.get(primitive_name)
        if source is None:
            raise ConnectorError(f"No query configured for primitive '{primitive_name}'")

        await self._ensure_connected()

        query = _translate_placeholders(source.query)

        try:
            rows = await self._pool.fetch(query, entity_id, timestamp)  # type: ignore[union-attr]
        except Exception as exc:
            raise ConnectorError(str(exc)) from exc

        if not rows:
            return PrimitiveValue(value=None)

        if len(rows) > 1:
            log.warning(
                "postgres_connector_multiple_rows primitive=%s row_count=%d",
                primitive_name,
                len(rows),
            )

        return PrimitiveValue(value=rows[0][0])

    async def fetch_forward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """
        Return the last known value at or before timestamp by walking backwards.

        Retries with timestamp - 1 day up to _MAX_FILL_DAYS (90) times.
        Returns PrimitiveValue(value=None) if no non-null value is found within
        the lookback window.

        LIMITATION: This day-walking approach is a simplification. For
        production use, the configured SQL query should express the range search
        directly:
            SELECT value FROM table
            WHERE entity_id = :entity_id AND event_date <= :as_of
            ORDER BY event_date DESC LIMIT 1
        """
        if timestamp is None:
            return PrimitiveValue(value=None)

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return PrimitiveValue(value=None)

        for _ in range(_MAX_FILL_DAYS):
            pv = await self.fetch(primitive_name, entity_id, ts.isoformat())
            if pv.value is not None:
                return pv
            ts = ts - timedelta(days=1)

        return PrimitiveValue(value=None)

    async def fetch_backward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """
        Return the next known value at or after timestamp by walking forwards.

        Retries with timestamp + 1 day up to _MAX_FILL_DAYS (90) times.
        Returns PrimitiveValue(value=None) if no non-null value is found.
        """
        if timestamp is None:
            return PrimitiveValue(value=None)

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return PrimitiveValue(value=None)

        for _ in range(_MAX_FILL_DAYS):
            pv = await self.fetch(primitive_name, entity_id, ts.isoformat())
            if pv.value is not None:
                return pv
            ts = ts + timedelta(days=1)

        return PrimitiveValue(value=None)

    async def close(self) -> None:
        """Close the asyncpg pool if it is open."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health_check(self) -> bool:
        """Execute SELECT 1. Returns True on success, False on any failure."""
        try:
            await self._ensure_connected()
            val = await self._pool.fetchval("SELECT 1")  # type: ignore[union-attr]
            return val == 1
        except Exception:
            return False
