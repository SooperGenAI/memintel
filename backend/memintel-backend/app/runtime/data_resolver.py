"""
app/runtime/data_resolver.py
──────────────────────────────────────────────────────────────────────────────
DataResolver — fetches primitive values for concept execution.

Responsibilities:
  1. Fetch a primitive value for an entity at a given timestamp (point-in-time).
  2. Apply the missing_data_policy when the connector returns no data:
       null          → return None (T? — nullable downstream)
       zero          → return 0.0 (non-nullable)
       forward_fill  → return the last known value before timestamp
       backward_fill → return the next known value after timestamp
  3. Retry transient connector failures with exponential backoff.
  4. Do NOT retry permanent failures (auth errors).
  5. Rate-limit errors are treated as transient and retried with backoff.
  6. Maintain a request-scoped cache so the same (primitive, entity, timestamp)
     is fetched at most once per DataResolver instance.  Create a new instance
     per execute() call to reset the cache between requests.
  7. Batch fetch: fetch_batch() calls the connector once for multiple primitives
     of the same entity (one network round-trip per batch, minus cache hits).

Connector interface
───────────────────
ConnectorBase is the abstract interface.  MockConnector is the in-process
test double.  Real connectors (SQL, REST) raise NotImplementedError on the
methods they have not yet implemented.

Retry policy
────────────
  Max retries:       3  (configurable)
  Backoff base:      0.1 s  (configurable; pass backoff_base=0.0 in tests)
  Backoff formula:   base * 2^attempt  (capped at 30 s)
  Retryable:         TransientConnectorError, RateLimitConnectorError
  Non-retryable:     AuthConnectorError, and any non-ConnectorError exception
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.models.config import PrimitiveSourceConfig
from app.models.result import MissingDataPolicy


# ── Error hierarchy ────────────────────────────────────────────────────────────

class ConnectorError(Exception):
    """Base class for all primitive connector errors."""


class TransientConnectorError(ConnectorError):
    """Transient failure — safe to retry (network blip, temp unavailability)."""


class RateLimitConnectorError(TransientConnectorError):
    """Rate limit hit — retried after backoff (subclass of transient)."""


class AuthConnectorError(ConnectorError):
    """Authentication or authorisation failure — MUST NOT retry."""


# ── PrimitiveValue ─────────────────────────────────────────────────────────────

@dataclass
class PrimitiveValue:
    """
    The resolved value of a single primitive fetch after policy application.

    value    — the primitive's data value, or None when policy=null was
               applied on a missing result.
    nullable — True when the raw data was absent and the policy preserved
               null.  Downstream operators must handle nullable inputs.
    """
    value: float | int | bool | str | list | None
    nullable: bool = False

    @property
    def is_missing(self) -> bool:
        """True when the connector returned no data for the requested key."""
        return self.value is None


# ── Connector interface ────────────────────────────────────────────────────────

class ConnectorBase(ABC):
    """
    Abstract primitive data connector.

    All methods raise ConnectorError subclasses on failure.  Implementations
    must wrap non-ConnectorError exceptions before re-raising.
    """

    @abstractmethod
    def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> Any:
        """
        Fetch the point-in-time value of ``primitive_name`` for ``entity_id``.

        Returns the raw value or None when no data exists for this triple.
        Raises ConnectorError subclass on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_batch(
        self,
        primitive_names: list[str],
        entity_id: str,
        timestamp: str | None,
    ) -> dict[str, Any]:
        """
        Fetch multiple primitives for ``entity_id`` in a single request.

        Returns {primitive_name: raw_value | None} for every requested name.
        Raises ConnectorError subclass on failure.
        """
        raise NotImplementedError

    def fetch_forward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> Any:
        """Return the last known value at or before ``timestamp``.  Optional."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fetch_forward_fill(). "
            "Override this method to support forward_fill policy."
        )

    def fetch_backward_fill(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
    ) -> Any:
        """Return the next known value at or after ``timestamp``.  Optional."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fetch_backward_fill(). "
            "Override this method to support backward_fill policy."
        )


# ── StaticDataConnector ───────────────────────────────────────────────────────

class StaticDataConnector(ConnectorBase):
    """
    In-memory connector that returns values from a flat {primitive_name: value}
    dict, ignoring entity_id and timestamp.

    Intended for POST /execute/static (local testing only) — no network calls,
    no connector configuration required.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def fetch(self, primitive_name: str, entity_id: str, timestamp: str | None) -> Any:
        return self._data.get(primitive_name)

    def fetch_batch(
        self, primitive_names: list[str], entity_id: str, timestamp: str | None
    ) -> dict[str, Any]:
        return {name: self._data.get(name) for name in primitive_names}


# ── MockConnector ─────────────────────────────────────────────────────────────

class MockConnector(ConnectorBase):
    """
    In-memory test double for ConnectorBase.

    Lookup tables (all optional):
      data               — {(primitive_name, entity_id, timestamp): value}
      forward_fill_data  — {(primitive_name, entity_id): value}
                           returned by fetch_forward_fill() when primary data absent.
      backward_fill_data — {(primitive_name, entity_id): value}
                           returned by fetch_backward_fill() when primary data absent.

    Failure simulation (counted down per call to _check_failures()):
      transient_failures  — raises TransientConnectorError N times, then succeeds.
      rate_limit_failures — raises RateLimitConnectorError N times, then succeeds.
      auth_failure        — always raises AuthConnectorError (permanent).

    Counters for test assertions:
      fetch_call_count    — incremented on every fetch() call.
      batch_call_count    — incremented on every fetch_batch() call.
    """

    def __init__(
        self,
        data: dict[tuple[str, str, str | None], Any] | None = None,
        forward_fill_data: dict[tuple[str, str], Any] | None = None,
        backward_fill_data: dict[tuple[str, str], Any] | None = None,
        transient_failures: int = 0,
        rate_limit_failures: int = 0,
        auth_failure: bool = False,
    ) -> None:
        self._data: dict[tuple[str, str, str | None], Any] = data or {}
        self._fwd: dict[tuple[str, str], Any] = forward_fill_data or {}
        self._bwd: dict[tuple[str, str], Any] = backward_fill_data or {}
        self._transient_remaining = transient_failures
        self._rate_limit_remaining = rate_limit_failures
        self._auth_failure = auth_failure
        self.fetch_call_count: int = 0
        self.batch_call_count: int = 0

    def _check_failures(self) -> None:
        if self._auth_failure:
            raise AuthConnectorError("Mock: authentication failed")
        if self._rate_limit_remaining > 0:
            self._rate_limit_remaining -= 1
            raise RateLimitConnectorError("Mock: rate limit exceeded")
        if self._transient_remaining > 0:
            self._transient_remaining -= 1
            raise TransientConnectorError("Mock: transient error")

    def fetch(self, primitive_name: str, entity_id: str, timestamp: str | None) -> Any:
        self.fetch_call_count += 1
        self._check_failures()
        return self._data.get((primitive_name, entity_id, timestamp))

    def fetch_batch(
        self, primitive_names: list[str], entity_id: str, timestamp: str | None
    ) -> dict[str, Any]:
        self.batch_call_count += 1
        self._check_failures()
        return {
            name: self._data.get((name, entity_id, timestamp))
            for name in primitive_names
        }

    def fetch_forward_fill(
        self, primitive_name: str, entity_id: str, timestamp: str | None
    ) -> Any:
        self._check_failures()
        return self._fwd.get((primitive_name, entity_id))

    def fetch_backward_fill(
        self, primitive_name: str, entity_id: str, timestamp: str | None
    ) -> Any:
        self._check_failures()
        return self._bwd.get((primitive_name, entity_id))


# ── Retry helper ──────────────────────────────────────────────────────────────

_MAX_RETRIES  = 3
_BACKOFF_BASE = 0.1    # seconds
_BACKOFF_CAP  = 30.0   # seconds


def _with_retry(fn, max_retries: int = _MAX_RETRIES, backoff_base: float = _BACKOFF_BASE) -> Any:
    """
    Call ``fn()`` with exponential-backoff retry on transient failures.

    Retried errors:     TransientConnectorError (includes RateLimitConnectorError).
    Never-retried:      AuthConnectorError, any non-ConnectorError exception.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except AuthConnectorError:
            raise  # permanent — propagate immediately, no retry
        except TransientConnectorError as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(backoff_base * (2 ** attempt), _BACKOFF_CAP)
                if delay > 0:
                    time.sleep(delay)
        except ConnectorError:
            raise  # unknown connector error — propagate immediately
    raise last_exc  # type: ignore[misc]


# ── DataResolver ──────────────────────────────────────────────────────────────

class DataResolver:
    """
    Fetches and resolves primitive values for a single concept execution.

    Each instance is request-scoped.  Create a new DataResolver per execute()
    call so the request-scoped cache resets between requests.

    Parameters
    ──────────
    connector      — ConnectorBase implementation (MockConnector in tests).
    missing_policy — default policy applied when data is absent.
                     May be overridden per-fetch via the policy= argument.
    max_retries    — maximum retry attempts for transient failures.
    backoff_base   — initial backoff in seconds.  Pass 0.0 in tests to avoid
                     real sleeping.
    """

    def __init__(
        self,
        connector: ConnectorBase,
        missing_policy: MissingDataPolicy = MissingDataPolicy.NULL,
        max_retries: int = _MAX_RETRIES,
        backoff_base: float = _BACKOFF_BASE,
        primitive_sources: dict[str, PrimitiveSourceConfig] | None = None,
        connector_registry: dict[str, ConnectorBase] | None = None,
    ) -> None:
        self._connector = connector
        self._default_policy = missing_policy
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._primitive_sources: dict[str, PrimitiveSourceConfig] = primitive_sources or {}
        self._connector_registry: dict[str, ConnectorBase] = connector_registry or {}
        # Request-scoped cache: dict keyed by (primitive_name, entity_id, timestamp).
        # Discarded when the DataResolver instance goes out of scope.
        self._cache: dict[tuple[str, str, str | None], PrimitiveValue] = {}

    def _get_connector(self, primitive_name: str) -> ConnectorBase:
        """
        Return the connector to use for a named primitive.

        If the primitive has an entry in primitive_sources and the referenced
        connector name is registered in connector_registry, that connector is
        returned. Otherwise, falls back to the default connector passed at
        construction time (MockConnector / StaticConnector behaviour).
        """
        source = self._primitive_sources.get(primitive_name)
        if source is not None:
            conn = self._connector_registry.get(source.connector)
            if conn is not None:
                return conn
        return self._connector

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
        policy: MissingDataPolicy | None = None,
    ) -> PrimitiveValue:
        """
        Return the value of ``primitive_name`` for ``entity_id`` at ``timestamp``.

        Hits the request-scoped cache first; only calls the connector on a miss.
        Missing data is resolved via the effective missing_data_policy.
        """
        effective_policy = policy if policy is not None else self._default_policy
        cache_key = (primitive_name, entity_id, timestamp)

        if cache_key in self._cache:
            return self._cache[cache_key]

        conn = self._get_connector(primitive_name)
        raw = _with_retry(
            lambda: conn.fetch(primitive_name, entity_id, timestamp),
            max_retries=self._max_retries,
            backoff_base=self._backoff_base,
        )

        resolved = self._apply_policy(raw, primitive_name, entity_id, timestamp, effective_policy)
        self._cache[cache_key] = resolved
        return resolved

    def fetch_batch(
        self,
        primitive_names: list[str],
        entity_id: str,
        timestamp: str | None,
        policy: MissingDataPolicy | None = None,
    ) -> dict[str, PrimitiveValue]:
        """
        Fetch multiple primitives for ``entity_id`` in one connector call.

        Names already in the request cache are not re-fetched.  Only cache
        misses are sent to the connector as a single batch request.
        Returns {primitive_name: PrimitiveValue} for every name.
        """
        effective_policy = policy if policy is not None else self._default_policy
        result: dict[str, PrimitiveValue] = {}
        to_fetch: list[str] = []

        for name in primitive_names:
            cache_key = (name, entity_id, timestamp)
            if cache_key in self._cache:
                result[name] = self._cache[cache_key]
            else:
                to_fetch.append(name)

        if to_fetch:
            # Group cache misses by their resolved connector so each connector
            # receives a single batch call (preserving the one-round-trip guarantee
            # when all primitives use the same connector).
            by_connector: dict[ConnectorBase, list[str]] = {}
            for name in to_fetch:
                conn = self._get_connector(name)
                if conn not in by_connector:
                    by_connector[conn] = []
                by_connector[conn].append(name)

            for conn, names in by_connector.items():
                raw_batch = _with_retry(
                    lambda _c=conn, _ns=names: _c.fetch_batch(_ns, entity_id, timestamp),
                    max_retries=self._max_retries,
                    backoff_base=self._backoff_base,
                )
                for name in names:
                    raw = raw_batch.get(name)
                    resolved = self._apply_policy(
                        raw, name, entity_id, timestamp, effective_policy
                    )
                    self._cache[(name, entity_id, timestamp)] = resolved
                    result[name] = resolved

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_policy(
        self,
        raw: Any,
        primitive_name: str,
        entity_id: str,
        timestamp: str | None,
        policy: MissingDataPolicy,
    ) -> PrimitiveValue:
        """Apply the missing_data_policy when raw is None (data absent)."""
        if raw is not None:
            return PrimitiveValue(value=raw, nullable=False)

        if policy == MissingDataPolicy.NULL:
            return PrimitiveValue(value=None, nullable=True)

        if policy == MissingDataPolicy.ZERO:
            return PrimitiveValue(value=0.0, nullable=False)

        if policy == MissingDataPolicy.FORWARD_FILL:
            filled = _with_retry(
                lambda: self._connector.fetch_forward_fill(
                    primitive_name, entity_id, timestamp
                ),
                max_retries=self._max_retries,
                backoff_base=self._backoff_base,
            )
            return (
                PrimitiveValue(value=filled, nullable=False)
                if filled is not None
                else PrimitiveValue(value=None, nullable=True)
            )

        if policy == MissingDataPolicy.BACKWARD_FILL:
            filled = _with_retry(
                lambda: self._connector.fetch_backward_fill(
                    primitive_name, entity_id, timestamp
                ),
                max_retries=self._max_retries,
                backoff_base=self._backoff_base,
            )
            return (
                PrimitiveValue(value=filled, nullable=False)
                if filled is not None
                else PrimitiveValue(value=None, nullable=True)
            )

        return PrimitiveValue(value=None, nullable=True)  # fallback
