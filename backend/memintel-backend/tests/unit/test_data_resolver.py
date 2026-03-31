"""
tests/unit/test_data_resolver.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for DataResolver, MockConnector, StaticDataConnector, and retry.

Coverage:
  1. timestamp fetch returns point-in-time data (not current state)
  2. forward_fill: returns last known value when current is null
  3. backward_fill: returns next known value when current is null
  4. null policy: returns T? (nullable) when data missing
  5. zero policy: returns 0 (non-nullable) when data missing
  6. retry logic: transient connector failure triggers backoff + retry
  7. auth failure: does NOT retry, fails immediately
  8. rate limit: retries with delay (treated as transient)
  9. batch: multiple primitives for same entity fetched in one request
 10. request-scoped cache: same primitive+entity fetched only once per request
 11. StaticDataConnector: fetch() returns value for known primitive
 12. StaticDataConnector: fetch() returns None for unknown primitive
 13. StaticDataConnector: ignores entity_id and timestamp
 14. StaticDataConnector: fetch_batch() returns full dict
 15. StaticDataConnector: fetch_batch() returns None for unknown names

Test isolation: every test creates its own MockConnector and DataResolver.
No shared mutable state between tests.
"""
from __future__ import annotations

import pytest

from app.models.result import MissingDataPolicy
from app.runtime.data_resolver import (
    AuthConnectorError,
    DataResolver,
    MockConnector,
    RateLimitConnectorError,
    StaticDataConnector,
    TransientConnectorError,
)


# ── Constants ──────────────────────────────────────────────────────────────────

_ENTITY = "user_42"
_PRIM   = "churn_probability"
_TS     = "2024-03-15T09:00:00Z"
_TS2    = "2024-03-16T09:00:00Z"


def _resolver(connector: MockConnector, policy: MissingDataPolicy = MissingDataPolicy.NULL) -> DataResolver:
    return DataResolver(connector=connector, missing_policy=policy, backoff_base=0.0)


# ── 1. Timestamp fetch returns point-in-time data ────────────────────────────

class TestPointInTimeFetch:
    def test_timestamp_fetch_returns_correct_value(self):
        connector = MockConnector(data={
            (_PRIM, _ENTITY, _TS):  0.9,
            (_PRIM, _ENTITY, _TS2): 0.2,
        })
        resolver = _resolver(connector)
        result = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert result.value == 0.9

    def test_different_timestamps_return_different_values(self):
        connector = MockConnector(data={
            (_PRIM, _ENTITY, _TS):  0.9,
            (_PRIM, _ENTITY, _TS2): 0.2,
        })
        resolver = _resolver(connector)
        r1 = resolver.fetch(_PRIM, _ENTITY, _TS)
        r2 = resolver.fetch(_PRIM, _ENTITY, _TS2)
        assert r1.value == 0.9
        assert r2.value == 0.2

    def test_snapshot_fetch_uses_none_timestamp(self):
        connector = MockConnector(data={
            (_PRIM, _ENTITY, None): 0.55,
            (_PRIM, _ENTITY, _TS):  0.9,
        })
        resolver = _resolver(connector)
        result = resolver.fetch(_PRIM, _ENTITY, None)
        assert result.value == 0.55

    def test_fetch_returns_primitive_value_with_nullable_false_when_present(self):
        connector = MockConnector(data={(_PRIM, _ENTITY, _TS): 0.7})
        resolver = _resolver(connector)
        result = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert result.nullable is False


# ── 2. forward_fill: last known value when current is null ────────────────────

class TestForwardFill:
    def test_forward_fill_returns_last_known_value(self):
        connector = MockConnector(
            data={},
            forward_fill_data={(_PRIM, _ENTITY): 0.75},
        )
        resolver = _resolver(connector, MissingDataPolicy.FORWARD_FILL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.FORWARD_FILL)
        assert result.value == 0.75
        assert result.nullable is False

    def test_forward_fill_null_when_no_fill_available(self):
        connector = MockConnector(data={}, forward_fill_data={})
        resolver = _resolver(connector, MissingDataPolicy.FORWARD_FILL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.FORWARD_FILL)
        assert result.value is None
        assert result.nullable is True

    def test_forward_fill_not_used_when_current_data_present(self):
        connector = MockConnector(
            data={(_PRIM, _ENTITY, _TS): 0.9},
            forward_fill_data={(_PRIM, _ENTITY): 0.75},
        )
        resolver = _resolver(connector, MissingDataPolicy.FORWARD_FILL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.FORWARD_FILL)
        # Current data exists → no fill needed.
        assert result.value == 0.9


# ── 3. backward_fill: next known value when current is null ───────────────────

class TestBackwardFill:
    def test_backward_fill_returns_next_known_value(self):
        connector = MockConnector(
            data={},
            backward_fill_data={(_PRIM, _ENTITY): 0.3},
        )
        resolver = _resolver(connector, MissingDataPolicy.BACKWARD_FILL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.BACKWARD_FILL)
        assert result.value == 0.3
        assert result.nullable is False

    def test_backward_fill_null_when_no_fill_available(self):
        connector = MockConnector(data={}, backward_fill_data={})
        resolver = _resolver(connector, MissingDataPolicy.BACKWARD_FILL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.BACKWARD_FILL)
        assert result.value is None
        assert result.nullable is True


# ── 4. null policy: returns T? (nullable) when data missing ───────────────────

class TestNullPolicy:
    def test_null_policy_returns_none_when_missing(self):
        connector = MockConnector(data={})
        resolver = _resolver(connector, MissingDataPolicy.NULL)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.NULL)
        assert result.value is None
        assert result.nullable is True

    def test_null_policy_is_missing_true(self):
        connector = MockConnector(data={})
        resolver = _resolver(connector)
        result = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert result.is_missing is True


# ── 5. zero policy: returns 0 (non-nullable) when data missing ────────────────

class TestZeroPolicy:
    def test_zero_policy_returns_zero_when_missing(self):
        connector = MockConnector(data={})
        resolver = _resolver(connector, MissingDataPolicy.ZERO)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.ZERO)
        assert result.value == 0.0
        assert result.nullable is False

    def test_zero_policy_is_missing_false(self):
        connector = MockConnector(data={})
        resolver = _resolver(connector, MissingDataPolicy.ZERO)
        result = resolver.fetch(_PRIM, _ENTITY, _TS, policy=MissingDataPolicy.ZERO)
        assert result.is_missing is False


# ── 6. retry: transient failures trigger retry ───────────────────────────────

class TestTransientRetry:
    def test_transient_failure_retried_and_eventually_succeeds(self):
        connector = MockConnector(
            data={(_PRIM, _ENTITY, _TS): 0.8},
            transient_failures=2,   # fails twice, then succeeds
        )
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        result = resolver.fetch(_PRIM, _ENTITY, _TS)
        # After 2 transient failures the connector returns the real value.
        assert result.value == 0.8

    def test_transient_failures_exhaust_retries_then_raise(self):
        connector = MockConnector(
            data={},
            transient_failures=10,  # more than max_retries=3
        )
        resolver = DataResolver(connector=connector, backoff_base=0.0, max_retries=3)
        pv = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert pv.fetch_error is True
        assert pv.value is None

    def test_fetch_call_count_reflects_retries(self):
        connector = MockConnector(
            data={(_PRIM, _ENTITY, _TS): 1.0},
            transient_failures=2,
        )
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        resolver.fetch(_PRIM, _ENTITY, _TS)
        # 2 failures + 1 success = 3 calls total.
        assert connector.fetch_call_count == 3


# ── 7. auth failure: no retry, fails immediately ─────────────────────────────

class TestAuthFailure:
    def test_auth_failure_raises_immediately(self):
        connector = MockConnector(data={}, auth_failure=True)
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        pv = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert pv.fetch_error is True
        assert pv.value is None

    def test_auth_failure_only_one_attempt(self):
        connector = MockConnector(data={}, auth_failure=True)
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        try:
            resolver.fetch(_PRIM, _ENTITY, _TS)
        except AuthConnectorError:
            pass
        # Auth error must not trigger retries — exactly 1 call.
        assert connector.fetch_call_count == 1


# ── 8. rate limit: retried with backoff (treated as transient) ────────────────

class TestRateLimitRetry:
    def test_rate_limit_retried_and_succeeds(self):
        connector = MockConnector(
            data={(_PRIM, _ENTITY, _TS): 0.6},
            rate_limit_failures=1,
        )
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        result = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert result.value == 0.6

    def test_rate_limit_exhausted_raises(self):
        connector = MockConnector(
            data={},
            rate_limit_failures=10,
        )
        resolver = DataResolver(connector=connector, backoff_base=0.0, max_retries=3)
        pv = resolver.fetch(_PRIM, _ENTITY, _TS)
        assert pv.fetch_error is True
        assert pv.value is None


# ── 9. batch: multiple primitives fetched in one connector call ───────────────

class TestBatchFetch:
    def test_batch_uses_single_connector_call(self):
        connector = MockConnector(data={
            ("prim_a", _ENTITY, _TS): 1.0,
            ("prim_b", _ENTITY, _TS): 2.0,
            ("prim_c", _ENTITY, _TS): 3.0,
        })
        resolver = _resolver(connector)
        results = resolver.fetch_batch(["prim_a", "prim_b", "prim_c"], _ENTITY, _TS)
        # All three returned correctly.
        assert results["prim_a"].value == 1.0
        assert results["prim_b"].value == 2.0
        assert results["prim_c"].value == 3.0
        # Only ONE connector batch call.
        assert connector.batch_call_count == 1

    def test_batch_applies_missing_policy_per_primitive(self):
        connector = MockConnector(data={
            ("prim_a", _ENTITY, _TS): 1.0,
            # prim_b is absent
        })
        resolver = _resolver(connector, MissingDataPolicy.ZERO)
        results = resolver.fetch_batch(["prim_a", "prim_b"], _ENTITY, _TS,
                                       policy=MissingDataPolicy.ZERO)
        assert results["prim_a"].value == 1.0
        assert results["prim_b"].value == 0.0
        assert results["prim_b"].nullable is False

    def test_batch_cache_hits_excluded_from_connector_call(self):
        connector = MockConnector(data={
            ("prim_a", _ENTITY, _TS): 1.0,
            ("prim_b", _ENTITY, _TS): 2.0,
        })
        resolver = _resolver(connector)
        # Pre-warm prim_a via individual fetch.
        resolver.fetch("prim_a", _ENTITY, _TS)
        # Batch with prim_a already in cache — only prim_b sent to connector.
        results = resolver.fetch_batch(["prim_a", "prim_b"], _ENTITY, _TS)
        assert results["prim_a"].value == 1.0
        assert results["prim_b"].value == 2.0
        # 1 individual fetch + 1 batch (only prim_b) = batch_call_count=1
        assert connector.batch_call_count == 1
        # prim_a was fetched individually, not batched again.
        assert connector.fetch_call_count == 1


# ── 10. request-scoped cache: same primitive fetched only once ────────────────

class TestRequestScopedCache:
    def test_same_primitive_fetched_only_once_per_resolver(self):
        connector = MockConnector(data={(_PRIM, _ENTITY, _TS): 0.5})
        resolver = _resolver(connector)

        resolver.fetch(_PRIM, _ENTITY, _TS)
        resolver.fetch(_PRIM, _ENTITY, _TS)
        resolver.fetch(_PRIM, _ENTITY, _TS)

        # The connector is called only once; subsequent fetches hit the local cache.
        assert connector.fetch_call_count == 1

    def test_new_resolver_instance_resets_cache(self):
        connector = MockConnector(data={(_PRIM, _ENTITY, _TS): 0.5})

        resolver1 = _resolver(connector)
        resolver1.fetch(_PRIM, _ENTITY, _TS)

        resolver2 = _resolver(connector)
        resolver2.fetch(_PRIM, _ENTITY, _TS)

        # Each resolver made its own call — cache is instance-scoped, not global.
        assert connector.fetch_call_count == 2

    def test_different_timestamps_are_cached_independently(self):
        connector = MockConnector(data={
            (_PRIM, _ENTITY, _TS):  0.9,
            (_PRIM, _ENTITY, _TS2): 0.2,
        })
        resolver = _resolver(connector)
        resolver.fetch(_PRIM, _ENTITY, _TS)
        resolver.fetch(_PRIM, _ENTITY, _TS)   # cache hit
        resolver.fetch(_PRIM, _ENTITY, _TS2)
        resolver.fetch(_PRIM, _ENTITY, _TS2)  # cache hit

        # Two distinct keys → 2 connector calls total.
        assert connector.fetch_call_count == 2


# ── 11-15. StaticDataConnector ────────────────────────────────────────────────

class TestStaticDataConnector:
    """StaticDataConnector — in-memory connector for /execute/static tests."""

    def test_fetch_returns_value_for_known_primitive(self):
        connector = StaticDataConnector({"revenue": 15000.0, "score": 0.9})
        assert connector.fetch("revenue", "any_entity", None) == 15000.0

    def test_fetch_returns_none_for_unknown_primitive(self):
        connector = StaticDataConnector({"revenue": 15000.0})
        assert connector.fetch("missing_prim", "any_entity", None) is None

    def test_fetch_ignores_entity_id_and_timestamp(self):
        connector = StaticDataConnector({"revenue": 42.0})
        # Different entity_id and timestamp — value must still be returned.
        assert connector.fetch("revenue", "entity_A", "2024-01-01T00:00:00Z") == 42.0
        assert connector.fetch("revenue", "entity_B", None) == 42.0
        assert connector.fetch("revenue", "entity_C", "2099-12-31T23:59:59Z") == 42.0

    def test_fetch_batch_returns_dict_of_all_requested_names(self):
        connector = StaticDataConnector({"revenue": 100.0, "score": 0.5, "tier": "gold"})
        result = connector.fetch_batch(["revenue", "score"], "ent", None)
        assert result == {"revenue": 100.0, "score": 0.5}

    def test_fetch_batch_returns_none_for_unknown_names(self):
        connector = StaticDataConnector({"revenue": 100.0})
        result = connector.fetch_batch(["revenue", "no_such_prim"], "ent", None)
        assert result["revenue"] == 100.0
        assert result["no_such_prim"] is None

    def test_fetch_batch_empty_request_returns_empty_dict(self):
        connector = StaticDataConnector({"revenue": 100.0})
        assert connector.fetch_batch([], "ent", None) == {}

    def test_works_with_data_resolver_fetch(self):
        """DataResolver wraps StaticDataConnector correctly."""
        connector = StaticDataConnector({"revenue": 5000.0})
        resolver = DataResolver(connector, backoff_base=0.0)
        pv = resolver.fetch("revenue", "acme", None)
        assert pv.value == 5000.0
        assert pv.nullable is False

    def test_works_with_data_resolver_fetch_missing_uses_null_policy(self):
        connector = StaticDataConnector({})   # no data
        resolver = DataResolver(connector, backoff_base=0.0)
        pv = resolver.fetch("revenue", "acme", None, policy=MissingDataPolicy.NULL)
        assert pv.value is None
        assert pv.nullable is True

    def test_works_with_data_resolver_fetch_missing_uses_zero_policy(self):
        connector = StaticDataConnector({})
        resolver = DataResolver(connector, backoff_base=0.0)
        pv = resolver.fetch("revenue", "acme", None, policy=MissingDataPolicy.ZERO)
        assert pv.value == 0.0
        assert pv.nullable is False
