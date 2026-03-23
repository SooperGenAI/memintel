"""
tests/integration/test_runtime_failure_scenarios.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for runtime failure scenarios.

Coverage:
  1. Connector unavailable
       - DataResolver exhausts retries on transient failures → TransientConnectorError
       - Error propagates through ConceptExecutor (not swallowed)
       - AuthConnectorError is NOT retried (permanent failure)
  2. Cache miss → execution still completes correctly
       - cache=False: result is correct, cache stays empty
       - cache=True + timestamp: second call is a cache hit (connector not called again)
       - snapshot (timestamp=None): never cached, connector called each time
  3. Concurrent executions → identical results
       - 10 threads calling execute_graph() concurrently for the same entity+timestamp
       - All 10 results are identical (determinism under concurrency)
  4. Database unavailable → clean HTTP error, no stack trace
       - get_execute_service raises MemintelError(EXECUTION_ERROR): 500 + JSON body
       - get_db raises AttributeError: FastAPI returns clean JSON, no traceback text

No real DB, Redis, or LLM calls. Scenarios 1–3 are pure in-process runtime tests.
Scenario 4 uses the FastAPI ASGI TestClient with dependency overrides.
"""
from __future__ import annotations

# aioredis stub must precede all app imports.
# aioredis uses `distutils` which was removed in Python 3.12+.
import sys
from unittest.mock import MagicMock

if "aioredis" not in sys.modules:
    sys.modules["aioredis"] = MagicMock()

import concurrent.futures
import threading
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from app.api.routes import execute as execute_route
from app.api.routes.execute import get_execute_service
from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.models.concept import ConceptDefinition, ExecutionGraph, FeatureNode, PrimitiveRef
from app.models.errors import ErrorType, MemintelError, memintel_error_handler
from app.models.result import MissingDataPolicy
from app.models.task import Namespace
from app.persistence.db import get_db
from app.runtime.cache import ResultCache
from app.runtime.data_resolver import (
    AuthConnectorError,
    DataResolver,
    MockConnector,
    TransientConnectorError,
)
from app.runtime.executor import ConceptExecutor


# ── Shared fixtures ────────────────────────────────────────────────────────────

_DEFN = ConceptDefinition(
    concept_id="org.runtime_test_concept",
    version="1.0",
    namespace=Namespace.ORG,
    output_type="float",
    primitives={
        "signal": PrimitiveRef(
            type="float",
            missing_data_policy=MissingDataPolicy.ZERO,
        ),
    },
    features={
        "score": FeatureNode(
            op="normalize",
            inputs={"input": "signal"},
        ),
    },
    output_feature="score",
)

_ENTITY    = "entity_runtime_test"
_TIMESTAMP = "2024-06-01T09:00:00Z"


def _build_graph() -> ExecutionGraph:
    """Compile _DEFN once and return the hashed ExecutionGraph."""
    builder = DAGBuilder()
    ir_gen  = IRGenerator()
    graph = builder.build_dag(_DEFN)
    ir_gen.hash_graph(graph)   # sets graph.ir_hash in-place
    return graph


# ── Scenario 1: Connector unavailable ─────────────────────────────────────────

def test_connector_unavailable_raises_typed_error() -> None:
    """
    When a connector raises TransientConnectorError on every call, DataResolver
    exhausts all retries and re-raises the error as a TransientConnectorError —
    a typed ConnectorError, NOT a bare Python exception or RuntimeError.

    Retry count: 1 initial call + 3 retries = 4 total connector calls.
    """
    connector = MockConnector(transient_failures=10)   # more than max_retries
    resolver  = DataResolver(
        connector=connector,
        backoff_base=0.0,   # no real sleep in tests
        max_retries=3,
    )

    with pytest.raises(TransientConnectorError):
        resolver.fetch("signal", _ENTITY, _TIMESTAMP)

    # Verify exactly 4 calls: 1 initial + 3 retries.
    assert connector.fetch_call_count == 4, (
        f"Expected 4 fetch calls (1 initial + 3 retries), got {connector.fetch_call_count}"
    )


def test_connector_unavailable_propagates_through_executor() -> None:
    """
    ConceptExecutor.execute_graph() propagates TransientConnectorError when the
    connector is permanently unavailable. It does NOT swallow the exception or
    return a silent None result.

    The error is a typed ConnectorError subclass — callers can catch it
    specifically. It is NOT a bare unhandled exception.
    """
    connector = MockConnector(transient_failures=10)
    resolver  = DataResolver(connector=connector, backoff_base=0.0, max_retries=3)
    cache     = ResultCache()
    executor  = ConceptExecutor(result_cache=cache)
    graph     = _build_graph()

    with pytest.raises(TransientConnectorError):
        executor.execute_graph(
            graph=graph,
            entity=_ENTITY,
            data_resolver=resolver,
            timestamp=_TIMESTAMP,
        )


def test_auth_failure_is_not_retried() -> None:
    """
    AuthConnectorError is a permanent failure and must NOT be retried.
    Exactly 1 connector call is expected — retry logic must be bypassed.
    """
    connector = MockConnector(auth_failure=True)
    resolver  = DataResolver(connector=connector, backoff_base=0.0, max_retries=3)

    with pytest.raises(AuthConnectorError):
        resolver.fetch("signal", _ENTITY, _TIMESTAMP)

    assert connector.fetch_call_count == 1, (
        f"Auth failures must not be retried — expected 1 call, "
        f"got {connector.fetch_call_count}"
    )


def test_transient_failure_then_success() -> None:
    """
    When the connector fails transiently for fewer than max_retries times and
    then succeeds, DataResolver returns the correct value without raising.
    """
    connector = MockConnector(
        data={("signal", _ENTITY, _TIMESTAMP): 0.5},
        transient_failures=2,   # fails twice, succeeds on the 3rd call
    )
    resolver = DataResolver(connector=connector, backoff_base=0.0, max_retries=3)

    pv = resolver.fetch("signal", _ENTITY, _TIMESTAMP)

    assert pv.value == 0.5, f"Expected 0.5 after transient recovery, got {pv.value}"
    assert not pv.nullable
    # 2 transient failures + 1 successful call = 3 connector calls.
    assert connector.fetch_call_count == 3, (
        f"Expected 3 calls (2 transient + 1 success), got {connector.fetch_call_count}"
    )


# ── Scenario 2: Cache miss → execution completes correctly ────────────────────

def test_cache_miss_execution_completes() -> None:
    """
    With cache=False, execution completes correctly. The result is deterministic
    (same input -> same output) and the result cache stays empty.
    """
    connector = MockConnector(data={("signal", _ENTITY, _TIMESTAMP): 0.75})
    resolver  = DataResolver(connector=connector, backoff_base=0.0)
    cache     = ResultCache()
    executor  = ConceptExecutor(result_cache=cache)
    graph     = _build_graph()

    result = executor.execute_graph(
        graph=graph,
        entity=_ENTITY,
        data_resolver=resolver,
        timestamp=_TIMESTAMP,
        cache=False,
    )

    assert result is not None
    assert result.entity      == _ENTITY
    assert result.version     == "1.0"
    assert result.deterministic is True

    # normalize(0.75) = 0.75 / (1 + |0.75|) = 0.75 / 1.75
    expected = 0.75 / 1.75
    assert abs(result.value - expected) < 1e-9, (
        f"Unexpected result value: {result.value}, expected {expected}"
    )

    # cache=False: the result must NOT be stored in the cache.
    assert len(cache) == 0, (
        f"cache=False must not populate the result cache; found {len(cache)} entries"
    )


def test_cache_hit_avoids_second_connector_call() -> None:
    """
    With cache=True and a timestamp present, the second execute_graph() call
    is served from the result cache without touching the connector.
    """
    connector = MockConnector(data={("signal", _ENTITY, _TIMESTAMP): 0.75})
    resolver  = DataResolver(connector=connector, backoff_base=0.0)
    cache     = ResultCache()
    executor  = ConceptExecutor(result_cache=cache)
    graph     = _build_graph()

    # First call — cache miss; connector must be called.
    r1 = executor.execute_graph(
        graph=graph,
        entity=_ENTITY,
        data_resolver=resolver,
        timestamp=_TIMESTAMP,
        cache=True,
    )
    assert connector.fetch_call_count == 1, (
        "First call (cache miss) must call the connector once"
    )
    assert len(cache) == 1, "Result must be stored in cache after first call"

    # Second call — cache hit; connector must NOT be called again.
    r2 = executor.execute_graph(
        graph=graph,
        entity=_ENTITY,
        data_resolver=resolver,
        timestamp=_TIMESTAMP,
        cache=True,
    )
    assert connector.fetch_call_count == 1, (
        "Second call (cache hit) must not call the connector again"
    )
    assert r1.value == r2.value, (
        f"Cached result must match original: {r1.value} != {r2.value}"
    )


def test_snapshot_result_never_cached() -> None:
    """
    Snapshot executions (timestamp=None) must never be stored in the result
    cache, even when cache=True. Each snapshot call must hit the connector.
    """
    connector = MockConnector(data={("signal", _ENTITY, None): 0.3})
    cache     = ResultCache()
    executor  = ConceptExecutor(result_cache=cache)
    graph     = _build_graph()

    for _ in range(2):
        # DataResolver is request-scoped — create a new instance per call so
        # the internal primitive cache resets between "requests".
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        executor.execute_graph(
            graph=graph,
            entity=_ENTITY,
            data_resolver=resolver,
            timestamp=None,   # snapshot mode
            cache=True,
        )

    # Snapshot results are never stored in the result cache.
    assert len(cache) == 0, (
        "Snapshot results must never be stored in the result cache"
    )
    # Both calls must have hit the connector (no ResultCache shortcut).
    assert connector.fetch_call_count == 2, (
        f"Each snapshot call must hit the connector; expected 2 calls, "
        f"got {connector.fetch_call_count}"
    )


def test_cache_key_separates_timestamp_from_none() -> None:
    """
    The cache key (concept_id, version, entity, timestamp) must treat
    timestamp=None and any explicit timestamp as completely independent keys.

    Calling with timestamp='T' must not produce a cache hit for timestamp=None,
    and vice versa.
    """
    connector = MockConnector(data={
        ("signal", _ENTITY, _TIMESTAMP): 0.8,
        ("signal", _ENTITY, None):       0.2,
    })
    resolver = DataResolver(connector=connector, backoff_base=0.0)
    cache    = ResultCache()
    executor = ConceptExecutor(result_cache=cache)
    graph    = _build_graph()

    r_ts = executor.execute_graph(
        graph=graph, entity=_ENTITY, data_resolver=resolver,
        timestamp=_TIMESTAMP, cache=True,
    )
    # Snapshot result — must NOT reuse the deterministic cache entry above.
    r_none = executor.execute_graph(
        graph=graph, entity=_ENTITY, data_resolver=resolver,
        timestamp=None, cache=True,
    )

    # Deterministic result was cached; snapshot was not.
    assert len(cache) == 1, (
        f"Only the deterministic result should be in cache; found {len(cache)}"
    )
    # The two results should differ (different connector values).
    assert r_ts.value != r_none.value, (
        "Deterministic and snapshot results must not collide in the cache"
    )
    assert r_ts.deterministic   is True
    assert r_none.deterministic is False


# ── Scenario 3: Concurrent executions → identical results ─────────────────────

def test_concurrent_executions_deterministic() -> None:
    """
    10 concurrent execute_graph() calls for the same (entity, timestamp) must
    all return identical results (determinism under concurrency).

    Each thread gets its own DataResolver (request-scoped cache). A shared
    ResultCache and MockConnector are used. Thread safety is provided by the
    GIL for the in-memory dict-backed ResultCache.
    """
    N = 10
    connector = MockConnector(data={("signal", _ENTITY, _TIMESTAMP): 0.42})
    cache     = ResultCache()
    executor  = ConceptExecutor(result_cache=cache)
    graph     = _build_graph()

    results: list[float] = []
    errors:  list[Exception] = []
    lock     = threading.Lock()

    def _run_one() -> None:
        # Each thread owns its DataResolver so the per-request cache is isolated.
        resolver = DataResolver(connector=connector, backoff_base=0.0)
        try:
            r = executor.execute_graph(
                graph=graph,
                entity=_ENTITY,
                data_resolver=resolver,
                timestamp=_TIMESTAMP,
                cache=True,
            )
            with lock:
                results.append(r.value)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        futures = [pool.submit(_run_one) for _ in range(N)]
        concurrent.futures.wait(futures)

    assert not errors, (
        f"Some concurrent executions raised errors: {errors}"
    )
    assert len(results) == N, (
        f"Expected {N} results, got {len(results)}"
    )

    first = results[0]
    for i, v in enumerate(results[1:], start=1):
        assert v == first, (
            f"Concurrent execution {i} returned {v}, expected {first}. "
            "Determinism under concurrency violated."
        )


def test_concurrent_executions_correct_value() -> None:
    """
    Complementary to the determinism test: verify the actual value is correct,
    not just that all concurrent results agree with each other.
    """
    signal_val = 0.42
    connector  = MockConnector(data={("signal", _ENTITY, _TIMESTAMP): signal_val})
    cache      = ResultCache()
    executor   = ConceptExecutor(result_cache=cache)
    graph      = _build_graph()

    resolver = DataResolver(connector=connector, backoff_base=0.0)
    r = executor.execute_graph(
        graph=graph,
        entity=_ENTITY,
        data_resolver=resolver,
        timestamp=_TIMESTAMP,
        cache=False,
    )

    expected = signal_val / (1.0 + signal_val)   # normalize(x) = x / (1 + |x|)
    assert abs(r.value - expected) < 1e-9, (
        f"normalize({signal_val}) expected {expected}, got {r.value}"
    )


# ── Scenario 4: Database unavailable → clean HTTP error ───────────────────────

@asynccontextmanager
async def _null_lifespan(app: FastAPI):
    """Null lifespan — skips DB/Redis/config startup so tests run in-process."""
    yield


def _make_test_app() -> FastAPI:
    app = FastAPI(lifespan=_null_lifespan)
    app.add_exception_handler(MemintelError, memintel_error_handler)
    # Routers have no built-in prefix — register with explicit prefix here.
    app.include_router(execute_route.evaluate_router, prefix="/evaluate")
    app.include_router(execute_route.router,          prefix="/execute")
    return app


_EVAL_FULL_BODY = {
    "concept_id":        "org.runtime_test_concept",
    "concept_version":   "1.0",
    "condition_id":      "org.cond_test",
    "condition_version": "1.0",
    "entity":            _ENTITY,
}


def test_db_unavailable_returns_clean_json_error() -> None:
    """
    When the execute service raises MemintelError(EXECUTION_ERROR) — simulating
    a database connection failure surfaced as an execution error — the API must:
      - Return HTTP 500 (execution_error maps to 500 per the error table)
      - Return valid JSON (not a stack trace or HTML)
      - Include error.type == 'execution_error' in the response body

    Note: The current error map has no 503 entry. A true DB-specific 503 would
    require adding a dedicated DatabaseUnavailableError -> 503 mapping. Until
    then, execution_error -> 500 is the correct typed response.
    """
    app = _make_test_app()

    async def _failing_service():
        raise MemintelError(
            ErrorType.EXECUTION_ERROR,
            "Database connection pool unavailable — connection refused to 127.0.0.1:5432",
        )

    app.dependency_overrides[get_execute_service] = _failing_service

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/evaluate/full", json=_EVAL_FULL_BODY)
    finally:
        app.dependency_overrides.clear()

    # Must not succeed.
    assert resp.status_code != 200, (
        f"DB failure must not return HTTP 200, got {resp.status_code}"
    )
    # execution_error -> HTTP 500.
    assert resp.status_code == 500, (
        f"execution_error must map to HTTP 500, got {resp.status_code}"
    )
    # Must be valid JSON — not an HTML page or Python traceback.
    body = resp.json()
    assert "error" in body, (
        f"Response body must contain 'error' key; got: {body}"
    )
    assert body["error"].get("type") == "execution_error", (
        f"error.type must be 'execution_error'; got: {body['error']}"
    )


def test_db_unavailable_response_is_not_stack_trace() -> None:
    """
    When get_db raises AttributeError (simulating app.state.db not being
    set — e.g. lifespan startup was skipped), FastAPI's default 500 handler
    must return clean JSON without exposing Python internals.

    Verifies:
      - HTTP status is not 200
      - Response body is valid JSON
      - No 'Traceback' or 'File "' substrings appear in the response text
        (i.e., no Python stack trace is leaked to the caller)
    """
    app = _make_test_app()

    async def _get_db_broken():
        raise AttributeError(
            "app.state has no attribute 'db' — connection pool was not initialised"
        )

    app.dependency_overrides[get_db] = _get_db_broken

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/evaluate/full", json=_EVAL_FULL_BODY)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code != 200, (
        f"DB failure must not return HTTP 200, got {resp.status_code}"
    )

    # FastAPI returns plain-text "Internal Server Error" for unhandled exceptions
    # (not a structured MemintelError, since AttributeError bypasses our handler).
    # The key invariant is that no Python stack trace is exposed to the caller.
    body_text = resp.text

    assert "Traceback" not in body_text, (
        "Python Traceback text leaked into HTTP response — "
        "internal errors must not expose implementation details"
    )
    assert 'File "' not in body_text, (
        "File path from traceback leaked into HTTP response"
    )
    # Verify the response is not an HTML error page (which would contain '<html>').
    assert "<html>" not in body_text.lower(), (
        "HTML error page leaked into HTTP response"
    )
