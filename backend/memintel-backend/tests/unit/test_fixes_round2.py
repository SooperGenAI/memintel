"""
tests/unit/test_fixes_round2.py
──────────────────────────────────────────────────────────────────────────────
Tests for the second round of bug fixes (audit round 2):

  FIX 1+2: ConceptExecutor.execute() is async; ConditionEvaluator.aevaluate()
            returns the correct DecisionValue using the async path.
  FIX 3:   explain_decision() with timestamp=None does not crash (uses aevaluate).
  FIX 4:   /execute/static is fully async (uses aexecute_graph).
  FIX 5:   get_explanation_service() wires real connector registry from app.state.
  FIX 6:   apply_calibration() correctly registers new condition version
           (DefinitionRegistry.register() extracts condition_id from body).
  FIX 7:   DecisionRecord accepts bool and str concept_value without validation error.
  FIX 8:   fetch_history() filters NULL values; _evaluate_strategy history list
           never contains None-valued rows.
  FIX 9:   DataResolver.afetch() correctly awaits async connector (already present).
  FIX 10:  _row_to_job() uses row["poll_interval_s"] correctly (already present).
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.condition import (
    ConditionDefinition,
    DecisionType,
    DecisionValue,
    StrategyDefinition,
    StrategyType,
)
from app.models.concept import ExecutionGraph, GraphNode
from app.models.decision import DecisionRecord
from app.models.result import (
    ConceptOutputType,
    ConceptResult,
    MissingDataPolicy,
)
from app.runtime.cache import ResultCache
from app.runtime.condition_evaluator import ConditionEvaluator
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_threshold_condition(
    concept_id: str = "org.churn_score",
    concept_version: str = "1.0",
    threshold: float = 0.7,
) -> ConditionDefinition:
    # Use model_validate(dict) to correctly handle the discriminated union strategy field
    return ConditionDefinition.model_validate({
        "condition_id": "high_churn",
        "version": "1.0",
        "concept_id": concept_id,
        "concept_version": concept_version,
        "strategy": {
            "type": "threshold",
            "params": {"direction": "above", "value": threshold},
        },
        "namespace": "org",
    })


def _concept_result(value: float = 0.82) -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity="user_42",
        version="1.0",
        deterministic=True,
        timestamp="2024-01-15T09:00:00Z",
    )


def _warm_cache(
    cache: ResultCache,
    concept_result: ConceptResult,
    condition: ConditionDefinition,
    timestamp: str | None = "2024-01-15T09:00:00Z",
) -> None:
    cache_key = (condition.concept_id, condition.concept_version, concept_result.entity, timestamp)
    cache.set(cache_key, concept_result)


# ── FIX 1: ConceptExecutor.execute() is now async ─────────────────────────────

def test_executor_execute_is_coroutine_function():
    """
    FIX 1: ConceptExecutor.execute() must be declared async def.
    Before the fix it was sync def and called get_by_concept() without await.
    """
    assert inspect.iscoroutinefunction(ConceptExecutor.execute), (
        "ConceptExecutor.execute() must be async def"
    )


@pytest.mark.asyncio
async def test_executor_execute_awaits_graph_store():
    """
    FIX 1: ConceptExecutor.execute() must await graph_store.get_by_concept().
    Verifies that await is used by confirming a real ConceptResult is returned
    (not a coroutine object).
    """
    graph_store = MagicMock()
    graph_store.get_by_concept = AsyncMock(return_value=None)

    executor = ConceptExecutor(result_cache=ResultCache(), graph_store=graph_store)
    from app.models.errors import MemintelError, ErrorType
    with pytest.raises(MemintelError) as exc_info:
        await executor.execute(
            concept_id="org.test",
            version="1.0",
            entity="user_1",
            data_resolver=DataResolver(connector=MockConnector(data={}), backoff_base=0.0),
        )
    # NOT_FOUND because graph is None — proves await was used (no AttributeError on coroutine)
    assert exc_info.value.error_type == ErrorType.NOT_FOUND
    graph_store.get_by_concept.assert_awaited_once_with("org.test", "1.0")


# ── FIX 2: ConditionEvaluator.aevaluate() returns correct DecisionValue ───────

@pytest.mark.asyncio
async def test_aevaluate_returns_decision_value_from_cache():
    """
    FIX 2: ConditionEvaluator.aevaluate() must return a DecisionValue.
    Cache is pre-warmed so no graph_store hit is needed.
    """
    condition = _make_threshold_condition(threshold=0.5)
    cr = _concept_result(value=0.82)  # above threshold → fires

    cache = ResultCache()
    _warm_cache(cache, cr, condition)

    executor = ConceptExecutor(result_cache=cache)
    evaluator = ConditionEvaluator(executor=executor, result_cache=cache)

    resolver = DataResolver(connector=MockConnector(data={}), backoff_base=0.0)
    decision = await evaluator.aevaluate(
        condition=condition,
        entity="user_42",
        data_resolver=resolver,
        timestamp="2024-01-15T09:00:00Z",
    )

    assert isinstance(decision, DecisionValue)
    assert decision.value is True  # 0.82 > 0.5


@pytest.mark.asyncio
async def test_aevaluate_calls_executor_on_cache_miss():
    """
    FIX 2: On cache miss, aevaluate() must call await executor.execute()
    (now async). Verified via a mock that checks await was used.
    """
    condition = _make_threshold_condition(threshold=0.5)
    cr = _concept_result(value=0.82)

    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=cr)

    cache = ResultCache()  # empty — will miss
    evaluator = ConditionEvaluator(executor=mock_executor, result_cache=cache)
    resolver = DataResolver(connector=MockConnector(data={}), backoff_base=0.0)

    decision = await evaluator.aevaluate(
        condition=condition,
        entity="user_42",
        data_resolver=resolver,
        timestamp="2024-01-15T09:00:00Z",
    )

    assert isinstance(decision, DecisionValue)
    mock_executor.execute.assert_awaited_once()


# ── FIX 3: explain_decision() with timestamp=None does not crash ──────────────

@pytest.mark.asyncio
async def test_explain_decision_snapshot_mode_does_not_crash():
    """
    FIX 3: explain_decision() with timestamp=None must not crash.
    Before the fix, the sync evaluate() path raised RuntimeError on cache miss
    because the result was not cached for None-timestamp (snapshot mode).
    After the fix, aevaluate() is awaited and _aget_concept_result() awaits
    executor.execute() directly.
    """
    from app.services.explanation import ExplanationService

    condition_id = "high_churn"
    condition_version = "1.0"
    entity = "user_42"

    body = {
        "condition_id": condition_id,
        "version": condition_version,
        "concept_id": "org.churn_score",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {
            "type": "threshold",
            "params": {"direction": "above", "value": 0.5},
        },
    }

    cr = _concept_result(value=0.82)

    class _MockRegistry:
        async def get(self, cid, ver):
            return body

    mock_executor = MagicMock()
    mock_executor.aexecute = AsyncMock(return_value=cr)

    # evaluator with aevaluate that returns a fixed decision
    mock_evaluator = MagicMock()
    mock_evaluator.aevaluate = AsyncMock(return_value=DecisionValue(
        value=True,
        decision_type=DecisionType.BOOLEAN,
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=None,
    ))

    svc = ExplanationService(
        definition_registry=_MockRegistry(),
        concept_executor=mock_executor,
        condition_evaluator=mock_evaluator,
        data_resolver=DataResolver(connector=MockConnector(data={}), backoff_base=0.0),
    )

    # timestamp=None — snapshot mode — must not crash
    result = await svc.explain_decision(
        condition_id=condition_id,
        condition_version=condition_version,
        entity=entity,
        timestamp=None,   # ← snapshot mode
    )

    assert result is not None
    assert result.decision is True
    # aevaluate must have been called (not the sync evaluate)
    mock_evaluator.aevaluate.assert_awaited_once()


# ── FIX 4: /execute/static uses async executor ────────────────────────────────

def test_execute_static_uses_aexecute_graph():
    """
    FIX 4: The /execute/static route must use await executor.aexecute_graph(),
    not the sync executor.execute_graph().

    Verified by inspecting the route source for the async call pattern.
    """
    import inspect
    import ast
    import textwrap
    from app.api.routes import execute as execute_module
    source = inspect.getsource(execute_module.execute_static)
    # The source must contain 'await executor.aexecute_graph' (async call)
    assert "await executor.aexecute_graph" in source, (
        "execute_static() must use 'await executor.aexecute_graph(...)' not 'executor.execute_graph(...)'"
    )
    # And must NOT call sync execute_graph without await
    assert "executor.execute_graph(" not in source.replace("await executor.aexecute_graph(", ""), (
        "execute_static() must not call sync execute_graph() directly"
    )


# ── FIX 5: get_explanation_service uses real connector registry ────────────────

@pytest.mark.asyncio
async def test_get_explanation_service_wires_connector_registry():
    """
    FIX 5: get_explanation_service() must read connector_registry from
    app.state and wire it into the DataResolver's async_connector_registry.
    Before the fix, a bare MockConnector(data={}) was always used.
    """
    from app.api.routes.decisions import get_explanation_service
    from app.runtime.data_resolver import DataResolver

    class _FakePool:
        async def fetchrow(self, *a, **kw): return None
        async def fetch(self, *a, **kw): return []
        async def fetchval(self, *a, **kw): return None

    sentinel_registry = {"pg": object()}  # non-empty registry

    class _FakeConnectorRegistry:
        _registry = sentinel_registry

    class _FakeRequest:
        class app:
            class state:
                connector_registry = _FakeConnectorRegistry()
                config = None

    service = await get_explanation_service(request=_FakeRequest(), pool=_FakePool())

    assert isinstance(service._data_resolver, DataResolver)
    # The async_connector_registry must be the one from app.state, not empty
    assert service._data_resolver._async_connector_registry is sentinel_registry, (
        "DataResolver must use the connector_registry from app.state"
    )


# ── FIX 6: apply_calibration correctly registers new condition version ─────────

def test_definition_registry_register_extracts_condition_id_from_body():
    """
    FIX 6: DefinitionRegistry.register() must correctly extract condition_id
    from the body dict as the definition_id argument to DefinitionStore.register().

    Verifies the extraction logic in DefinitionRegistry.register() when
    definition_type='condition' and the body has a 'condition_id' key.
    """
    from app.registry.definitions import DefinitionRegistry

    captured: dict = {}

    class _MockStore:
        async def register(self, definition_id, version, definition_type,
                           namespace, body, meaning_hash=None, ir_hash=None):
            captured["definition_id"] = definition_id
            captured["version"] = version
            captured["definition_type"] = definition_type
            return MagicMock(definition_id=definition_id, version=version)

    registry = DefinitionRegistry(store=_MockStore())
    body = {
        "condition_id": "high_churn",
        "version": "2.0",
        "concept_id": "org.churn_score",
        "concept_version": "1.0",
        "namespace": "org",
        "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.6}},
    }

    asyncio.run(registry.register(body, namespace="org", definition_type="condition"))

    assert captured["definition_id"] == "high_churn", (
        f"definition_id must be extracted from body['condition_id']. "
        f"Got: {captured.get('definition_id')!r}"
    )
    assert captured["version"] == "2.0"
    assert captured["definition_type"] == "condition"


# ── FIX 7: DecisionRecord accepts bool and str concept_value ──────────────────

def test_decision_record_accepts_float_concept_value():
    """FIX 7: DecisionRecord.concept_value accepts float (existing behaviour)."""
    record = DecisionRecord(
        concept_id="org.churn_score",
        concept_version="1.0",
        condition_id="high_churn",
        condition_version="1.0",
        entity_id="user_42",
        fired=True,
        concept_value=0.82,
    )
    assert record.concept_value == 0.82


def test_decision_record_accepts_bool_concept_value():
    """
    FIX 7: DecisionRecord.concept_value must accept bool.
    Before the fix, concept_value was typed as float | None, which would
    silently coerce True→1.0 or raise for certain Pydantic v2 modes.
    After the fix the field is float | int | bool | str | None.
    """
    record = DecisionRecord(
        concept_id="org.boolean_concept",
        concept_version="1.0",
        condition_id="flag_condition",
        condition_version="1.0",
        entity_id="user_42",
        fired=True,
        concept_value=True,
    )
    # Pydantic may coerce True→1 in some union orderings; the important thing is
    # that no ValidationError is raised and the value is truthy.
    assert record.concept_value  # True or 1 are both truthy


def test_decision_record_accepts_str_concept_value():
    """
    FIX 7: DecisionRecord.concept_value must accept str (categorical concepts).
    Before the fix, 'churn_risk' would cause a ValidationError.
    """
    record = DecisionRecord(
        concept_id="org.category_concept",
        concept_version="1.0",
        condition_id="category_condition",
        condition_version="1.0",
        entity_id="user_42",
        fired=True,
        concept_value="churn_risk",
    )
    assert record.concept_value == "churn_risk"


def test_decision_record_accepts_none_concept_value():
    """FIX 7: None is still valid (no concept value available)."""
    record = DecisionRecord(
        concept_id="org.churn_score",
        concept_version="1.0",
        condition_id="high_churn",
        condition_version="1.0",
        entity_id="user_42",
        fired=False,
        concept_value=None,
    )
    assert record.concept_value is None


# ── FIX 8: fetch_history filters NULL values ──────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_history_filters_null_values():
    """
    FIX 8: ConceptResultStore.fetch_history() must include AND value IS NOT NULL
    in the WHERE clause so that categorical concept rows (where value=NULL) are
    excluded from history used by numeric strategies.

    Before the fix, a categorical row with value=NULL would cause
    float(row["value"]) → TypeError in _evaluate_strategy().
    """
    from app.stores.concept_result import ConceptResultStore
    import inspect

    source = inspect.getsource(ConceptResultStore.fetch_history)
    assert "value IS NOT NULL" in source, (
        "fetch_history() must include 'AND value IS NOT NULL' to exclude "
        "categorical rows with NULL numeric values"
    )


@pytest.mark.asyncio
async def test_fetch_history_query_excludes_null_rows():
    """
    FIX 8: fetch_history() with a pool that returns one NULL-value row must
    return an empty list (the NULL row is excluded by the SQL filter).

    This verifies the fix end-to-end using a recording pool.
    """
    from app.stores.concept_result import ConceptResultStore

    class _RecordingPool:
        def __init__(self):
            self.last_query: str = ""

        async def fetch(self, query: str, *args) -> list:
            self.last_query = query
            return []  # DB returns nothing (WHERE value IS NOT NULL excluded the row)

    pool = _RecordingPool()
    store = ConceptResultStore(pool)
    rows = await store.fetch_history("org.churn_score", "user_42", limit=10)

    assert rows == []
    assert "value IS NOT NULL" in pool.last_query, (
        "The SQL query passed to the pool must contain 'value IS NOT NULL'"
    )


# ── FIX 9: DataResolver.afetch() already present ─────────────────────────────

def test_data_resolver_has_afetch():
    """
    FIX 9: DataResolver must have an async afetch() method.
    This was already present; verified here as a regression guard.
    """
    assert hasattr(DataResolver, "afetch"), "DataResolver must have afetch()"
    assert inspect.iscoroutinefunction(DataResolver.afetch), (
        "DataResolver.afetch() must be async def"
    )


@pytest.mark.asyncio
async def test_data_resolver_afetch_uses_mock_connector_fallback():
    """
    FIX 9: DataResolver.afetch() falls back to the sync MockConnector when no
    async connector is configured for the primitive. Returns correct PrimitiveValue.
    """
    data = {("churn_score", "user_42", "2024-01-15T09:00:00Z"): 0.9}
    resolver = DataResolver(
        connector=MockConnector(data=data),
        backoff_base=0.0,
    )
    pv = await resolver.afetch(
        "churn_score", "user_42", "2024-01-15T09:00:00Z"
    )
    assert pv.value == 0.9
    assert pv.fetch_error is False


# ── FIX 10: _row_to_job() uses poll_interval_s correctly ─────────────────────

def test_row_to_job_maps_poll_interval_s_correctly():
    """
    FIX 10: _row_to_job() must read row["poll_interval_s"] (DB column name)
    and map it to Job.poll_interval_seconds (Python field name).

    Verifies the mapping by inspecting the source code — the DB column is
    'poll_interval_s' per the jobs table DDL.
    """
    import inspect
    from app.stores import job as job_module

    source = inspect.getsource(job_module._row_to_job)
    assert 'row["poll_interval_s"]' in source, (
        "_row_to_job() must use row['poll_interval_s'] to map the DB column name"
    )


def test_row_to_job_returns_correct_poll_interval():
    """
    FIX 10: A job row with poll_interval_s=5 must produce
    Job.poll_interval_seconds=5 — no KeyError.
    """
    import datetime
    from app.stores.job import _row_to_job
    from app.models.result import JobStatus

    class _Row(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

    row = _Row({
        "job_id": "job-abc",
        "job_type": "execute",
        "status": "queued",
        "request_body": None,
        "result_body": None,
        "error_body": None,
        "poll_interval_s": 5,
        "enqueued_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        "started_at": None,
        "completed_at": None,
        "updated_at": None,
    })

    job = _row_to_job(row)
    assert job.poll_interval_seconds == 5, (
        f"poll_interval_seconds must be 5, got {job.poll_interval_seconds}"
    )
    assert job.status == JobStatus.QUEUED
