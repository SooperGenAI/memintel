"""
tests/unit/test_connectors.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for PostgresConnector, RestConnector, and DataResolver async dispatch.

Coverage:
  PostgresConnector:
    1. fetch: single row returned → correct value
    2. fetch: no rows → PrimitiveValue(None)
    3. fetch: multiple rows → first row used, warning logged
    4. fetch: no query configured → ConnectorError
    5. fetch: database error → ConnectorError
    6. health_check: SELECT 1 succeeds → True

  RestConnector:
    7. fetch: 200 response with json_path → correct value
    8. fetch: HTTP 404 → ConnectorError
    9. fetch_forward_fill → ConnectorError (not supported)

  DataResolver async dispatch:
    10. primitive in sources → uses async connector, not MockConnector
    11. primitive not in sources → falls back to MockConnector

All tests use asyncio.run() — no pytest-asyncio dependency needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.config import ConnectorConfig, PrimitiveSourceConfig
from app.models.result import MissingDataPolicy
from app.runtime.data_resolver import ConnectorError, DataResolver, MockConnector, PrimitiveValue


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _postgres_config() -> ConnectorConfig:
    return ConnectorConfig.model_validate(
        {
            "type": "postgres",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        },
        context={"resolved": True},
    )


def _rest_config() -> ConnectorConfig:
    return ConnectorConfig.model_validate(
        {
            "type": "rest",
            "base_url": "http://api.example.com",
        },
        context={"resolved": True},
    )


def _prim_sources(
    prim_name: str = "user.score",
    query: str = "SELECT score FROM t WHERE user_id = $1 AND ts <= $2",
    json_path: str | None = None,
    connector: str = "postgres.analytics",
) -> dict[str, PrimitiveSourceConfig]:
    return {
        prim_name: PrimitiveSourceConfig(
            connector=connector,
            query=query,
            json_path=json_path,
        )
    }


def _make_asyncpg_mock(rows) -> MagicMock:
    """Build a mock asyncpg pool whose fetch() returns rows."""
    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=rows)
    mock_pool.fetchval = AsyncMock(return_value=1)
    return mock_pool


def _make_asyncpg_row(value: Any) -> MagicMock:
    """Build a minimal asyncpg Row-like object where row[0] returns value."""
    row = MagicMock()
    row.__getitem__ = MagicMock(return_value=value)
    return row


# ── PostgresConnector tests ───────────────────────────────────────────────────

def test_postgres_fetch_single_row_returns_correct_value():
    """fetch: single row → PrimitiveValue.value == row[0]."""
    from app.connectors.postgres import PostgresConnector

    prim_name = "user.score"
    expected = 0.85
    row = _make_asyncpg_row(expected)
    mock_pool = _make_asyncpg_mock([row])

    sources = _prim_sources(prim_name)
    connector = PostgresConnector(_postgres_config(), sources)
    connector._pool = mock_pool  # inject mock pool (skip connect())

    result = _run(connector.fetch(prim_name, "user_1", "2024-01-01T00:00:00Z"))

    assert isinstance(result, PrimitiveValue)
    assert result.value == expected


def test_postgres_fetch_no_rows_returns_none():
    """fetch: empty result set → PrimitiveValue(value=None)."""
    from app.connectors.postgres import PostgresConnector

    prim_name = "user.score"
    mock_pool = _make_asyncpg_mock([])

    connector = PostgresConnector(_postgres_config(), _prim_sources(prim_name))
    connector._pool = mock_pool

    result = _run(connector.fetch(prim_name, "user_1", "2024-01-01T00:00:00Z"))

    assert isinstance(result, PrimitiveValue)
    assert result.value is None


def test_postgres_fetch_multiple_rows_uses_first_row_and_logs_warning(caplog):
    """fetch: multiple rows → uses row[0], logs a warning."""
    from app.connectors.postgres import PostgresConnector

    prim_name = "user.score"
    rows = [_make_asyncpg_row(0.9), _make_asyncpg_row(0.5)]
    mock_pool = _make_asyncpg_mock(rows)

    connector = PostgresConnector(_postgres_config(), _prim_sources(prim_name))
    connector._pool = mock_pool

    with caplog.at_level(logging.WARNING):
        result = _run(connector.fetch(prim_name, "user_1", "2024-01-01T00:00:00Z"))

    assert result.value == 0.9
    assert "multiple" in caplog.text.lower()


def test_postgres_fetch_no_query_configured_raises_connector_error():
    """fetch: primitive not in primitive_sources → ConnectorError."""
    from app.connectors.postgres import PostgresConnector

    connector = PostgresConnector(_postgres_config(), {})  # empty sources

    with pytest.raises(ConnectorError, match="No query configured"):
        _run(connector.fetch("user.missing", "user_1", None))


def test_postgres_fetch_database_error_raises_connector_error():
    """fetch: asyncpg raises exception → ConnectorError."""
    from app.connectors.postgres import PostgresConnector

    prim_name = "user.score"
    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(side_effect=Exception("connection reset"))

    connector = PostgresConnector(_postgres_config(), _prim_sources(prim_name))
    connector._pool = mock_pool

    with pytest.raises(ConnectorError, match="connection reset"):
        _run(connector.fetch(prim_name, "user_1", "2024-01-01T00:00:00Z"))


def test_postgres_health_check_returns_true_on_success():
    """health_check: SELECT 1 returns 1 → True."""
    from app.connectors.postgres import PostgresConnector

    mock_pool = AsyncMock()
    mock_pool.fetchval = AsyncMock(return_value=1)

    connector = PostgresConnector(_postgres_config(), {})
    connector._pool = mock_pool

    result = _run(connector.health_check())

    assert result is True
    mock_pool.fetchval.assert_awaited_once_with("SELECT 1")


# ── RestConnector tests ────────────────────────────────────────────────────────

def _make_httpx_mock(status_code: int = 200, json_body: Any = None, text: str = "") -> MagicMock:
    """Return a mock httpx.AsyncClient context manager with a canned response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    if json_body is not None:
        mock_resp.json = MagicMock(return_value=json_body)
    else:
        mock_resp.json = MagicMock(side_effect=Exception("not JSON"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    return mock_client


def test_rest_fetch_200_with_json_path_returns_correct_value():
    """fetch: 200 response with json_path → extracts nested value."""
    from app.connectors.rest import RestConnector

    prim_name = "payment.rate"
    sources = _prim_sources(
        prim_name=prim_name,
        query="/payments/{entity_id}/stats",
        json_path="data.failure_rate",
        connector="rest.billing_api",
    )
    connector = RestConnector(_rest_config(), sources)

    json_body = {"data": {"failure_rate": 0.03, "total": 100}}
    mock_client = _make_httpx_mock(200, json_body)

    with patch("app.connectors.rest.httpx.AsyncClient", return_value=mock_client):
        result = _run(connector.fetch(prim_name, "acct_42", "2024-01-01T00:00:00Z"))

    assert isinstance(result, PrimitiveValue)
    assert result.value == 0.03


def test_rest_fetch_404_raises_connector_error():
    """fetch: HTTP 404 → ConnectorError with status code in message."""
    from app.connectors.rest import RestConnector

    prim_name = "payment.rate"
    sources = _prim_sources(
        prim_name=prim_name,
        query="/payments/{entity_id}/stats",
        connector="rest.billing_api",
    )
    connector = RestConnector(_rest_config(), sources)

    mock_client = _make_httpx_mock(404)

    with patch("app.connectors.rest.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ConnectorError, match="404"):
            _run(connector.fetch(prim_name, "acct_42", "2024-01-01T00:00:00Z"))


def test_rest_fetch_forward_fill_raises_connector_error():
    """fetch_forward_fill: not supported → ConnectorError."""
    from app.connectors.rest import RestConnector

    connector = RestConnector(_rest_config(), {})

    with pytest.raises(ConnectorError, match="forward_fill not supported"):
        _run(connector.fetch_forward_fill("payment.rate", "acct_42", None))


# ── DataResolver async dispatch tests ─────────────────────────────────────────

def test_data_resolver_afetch_uses_async_connector_when_configured():
    """
    DataResolver.afetch: primitive in primitive_sources with matching async connector
    → calls async connector, NOT the default MockConnector.
    """
    prim_name = "user.score"
    entity = "user_99"
    ts = "2024-01-01T00:00:00Z"

    expected_pv = PrimitiveValue(value=0.75)

    # Build an async mock connector
    async_mock_conn = AsyncMock()
    async_mock_conn.fetch = AsyncMock(return_value=expected_pv)

    sources = {prim_name: PrimitiveSourceConfig(connector="postgres.analytics", query="SELECT 1")}
    async_registry = {"postgres.analytics": async_mock_conn}

    mock_default = MockConnector(data={})

    resolver = DataResolver(
        connector=mock_default,
        primitive_sources=sources,
        async_connector_registry=async_registry,
        backoff_base=0.0,
    )

    result = _run(resolver.afetch(prim_name, entity, ts))

    assert result.value == 0.75
    async_mock_conn.fetch.assert_awaited_once_with(prim_name, entity, ts)
    assert mock_default.fetch_call_count == 0  # MockConnector was NOT called


def test_data_resolver_afetch_falls_back_to_mock_when_no_async_connector():
    """
    DataResolver.afetch: primitive NOT in primitive_sources → falls back to
    default MockConnector (backward-compatible with existing tests).
    """
    prim_name = "user.score"
    entity = "user_99"
    ts = "2024-01-01T00:00:00Z"
    mock_value = 0.5

    mock_connector = MockConnector(data={(prim_name, entity, ts): mock_value})
    resolver = DataResolver(
        connector=mock_connector,
        async_connector_registry={},
        backoff_base=0.0,
    )

    result = _run(resolver.afetch(prim_name, entity, ts))

    assert result.value == mock_value
    assert mock_connector.fetch_call_count == 1
