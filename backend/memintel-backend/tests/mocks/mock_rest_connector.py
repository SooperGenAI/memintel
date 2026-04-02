"""
tests/mocks/mock_rest_connector.py
──────────────────────────────────────────────────────────────────────────────
MockAsyncRestConnector — async test double for RestConnector.

Models the async connector interface used by PostgresConnector and
RestConnector: fetch() is an async coroutine that returns PrimitiveValue
directly (not a raw value).  This lets it be registered in the
async_connector_registry of DataResolver so the afetch() async path
is exercised end-to-end, including the source-to-connector routing.

Data table
──────────
Same nested dict structure as MockTableConnector:

    data = {
        "account.active_user_rate_30d": {
            "account_001": 0.42,
        },
    }

Failure injection
─────────────────
transient_failures  int   — raises TransientConnectorError N times then succeeds.
auth_failure        bool  — always raises AuthConnectorError (permanent).

Note: unlike the sync path, async connector errors propagate up through
DataResolver.afetch() which catches ConnectorError and returns a
PrimitiveValue(fetch_error=True) — the pipeline never raises HTTP 500.

Wiring into tests
─────────────────
Use the make_async_connector_override() factory in conftest.py to inject
MockAsyncRestConnector into a TestClient:

    connector = MockAsyncRestConnector(data, connector_name="rest_mock")
    with mock_connector_e2e_client(async_connector=connector) as (client, pool, run_db):
        ...

This registers the connector under connector_name in the async_connector_registry
and patches primitive_sources so the DataResolver routes the primitive through it.
"""
from __future__ import annotations

from typing import Any

from app.runtime.data_resolver import (
    AuthConnectorError,
    PrimitiveValue,
    TransientConnectorError,
)


class MockAsyncRestConnector:
    """
    Async test double for the REST connector interface.

    Implements the same interface as RestConnector:
      async fetch(primitive_name, entity_id, timestamp) → PrimitiveValue
    """

    def __init__(
        self,
        data: dict[str, dict[str, Any]] | None = None,
        connector_name: str = "rest_mock",
        transient_failures: int = 0,
        auth_failure: bool = False,
    ) -> None:
        self._data: dict[str, dict[str, Any]] = data or {}
        self.connector_name: str = connector_name
        self._transient_remaining: int = transient_failures
        self._auth_failure: bool = auth_failure
        self.call_log: list[tuple[str, str, str | None]] = []
        self.fetch_call_count: int = 0

    def _check_failures(self) -> None:
        if self._auth_failure:
            raise AuthConnectorError("MockAsyncRestConnector: auth failed")
        if self._transient_remaining > 0:
            self._transient_remaining -= 1
            raise TransientConnectorError("MockAsyncRestConnector: transient error")

    async def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """Return PrimitiveValue from table (async — matches RestConnector interface)."""
        self.fetch_call_count += 1
        self.call_log.append((primitive_name, entity_id, timestamp))
        self._check_failures()
        value = self._data.get(primitive_name, {}).get(entity_id)
        return PrimitiveValue(value=value, nullable=(value is None))

    async def fetch_forward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """Forward fill is not supported — returns null."""
        return PrimitiveValue(value=None, nullable=True)

    async def fetch_backward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> PrimitiveValue:
        """Backward fill is not supported — returns null."""
        return PrimitiveValue(value=None, nullable=True)

    async def health_check(self) -> None:
        """No-op health check for test double."""
        pass

    async def close(self) -> None:
        """No-op close for test double."""
        pass
