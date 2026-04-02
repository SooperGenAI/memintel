"""
tests/mocks/mock_connector.py
──────────────────────────────────────────────────────────────────────────────
MockTableConnector — test double for ConnectorBase with a (primitive, entity)
lookup table.

Differences from the production MockConnector in data_resolver.py
──────────────────────────────────────────────────────────────────
The built-in MockConnector uses a 3-tuple key (primitive_name, entity_id,
timestamp) which makes test data tables verbose when the specific timestamp
does not matter.  MockTableConnector uses a 2-level nested dict so test data
is readable:

    data = {
        "account.active_user_rate_30d": {
            "account_001": 0.25,   # below threshold → fires
            "account_002": 0.85,   # above threshold → no fire
            "account_null": None,  # null → null_input policy
        },
        "account.days_to_renewal": {
            "account_001": 30,
            "account_002": 90,
        },
        "account.payment_failed_flag": {
            "account_failed": "true",
            "account_ok": "false",
        },
        "user.session_frequency_trend_8w": {
            "user_trend": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        },
    }

Missing (primitive_name, entity_id) pairs return None (triggers missing_data_policy).

Failure injection
─────────────────
transient_failures  int   — raises TransientConnectorError N times then succeeds.
auth_failure        bool  — always raises AuthConnectorError (permanent, never retried).

Call log
────────
call_log — list[(primitive_name, entity_id, timestamp)] for every fetch call.
           Inspect this to assert that as_of timestamp is passed correctly (Test 9).

fetch_call_count / batch_call_count — counters for quantity assertions.
"""
from __future__ import annotations

from typing import Any

from app.runtime.data_resolver import (
    AuthConnectorError,
    ConnectorBase,
    TransientConnectorError,
)


class MockTableConnector(ConnectorBase):
    """
    Timestamp-agnostic test double for ConnectorBase.

    Looks up (primitive_name, entity_id) in a nested dict and returns the
    associated value regardless of the timestamp argument.  The timestamp IS
    recorded in call_log so tests that care about it can assert it there.

    Data table is mutable — tests can update self._data between calls to
    simulate different values per execution (e.g. Test 10 history test).
    """

    def __init__(
        self,
        data: dict[str, dict[str, Any]] | None = None,
        transient_failures: int = 0,
        auth_failure: bool = False,
    ) -> None:
        self._data: dict[str, dict[str, Any]] = data or {}
        self._transient_remaining: int = transient_failures
        self._auth_failure: bool = auth_failure
        # (primitive_name, entity_id, timestamp) for every fetch call
        self.call_log: list[tuple[str, str, str | None]] = []
        self.fetch_call_count: int = 0
        self.batch_call_count: int = 0

    def _check_failures(self) -> None:
        """Raise configured failure exception, decrementing transient counter."""
        if self._auth_failure:
            raise AuthConnectorError("MockTableConnector: authentication failed")
        if self._transient_remaining > 0:
            self._transient_remaining -= 1
            raise TransientConnectorError("MockTableConnector: transient error")

    def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> Any:
        """Return value from table or None when (primitive, entity) is absent."""
        self.fetch_call_count += 1
        self.call_log.append((primitive_name, entity_id, timestamp))
        self._check_failures()
        return self._data.get(primitive_name, {}).get(entity_id)

    def fetch_batch(
        self,
        primitive_names: list[str],
        entity_id: str,
        timestamp: str | None,
    ) -> dict[str, Any]:
        """Batch-fetch multiple primitives for the same entity."""
        self.batch_call_count += 1
        self._check_failures()
        result: dict[str, Any] = {}
        for name in primitive_names:
            self.call_log.append((name, entity_id, timestamp))
            result[name] = self._data.get(name, {}).get(entity_id)
        return result
