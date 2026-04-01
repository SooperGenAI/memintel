"""
tests/unit/test_executor.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the ConceptExecutor, ConditionEvaluator, and ActionTrigger.

Coverage:
  1. Same inputs produce same output (determinism — run 3× assert identical)
  2. Timestamp present → deterministic=True
  3. Timestamp absent  → deterministic=False
  4. Snapshot result NOT stored in cache across requests
  5. dry_run → all actions have status='would_trigger'
  6. Action failure does NOT fail the pipeline (HTTP 200 analogue)
  7. Actions do NOT retry on failure (at-most-once per evaluation call)
  8. Same evaluation called twice fires actions twice (no cross-call dedup)

Test isolation: every test creates its own MockConnector, DataResolver,
ResultCache, and ConceptExecutor. No shared state between tests.
"""
from __future__ import annotations

import asyncio
import pytest

from app.models.action import ActionDefinition, FireOn, NotificationActionConfig, TriggerConfig
from app.models.concept import ExecutionGraph, GraphEdge, GraphNode
from app.models.condition import (
    ConditionDefinition,
    DecisionType,
    DecisionValue,
    StrategyType,
    ThresholdStrategy as ThresholdStrategyModel,
    ThresholdParams,
)
from app.models.result import ActionTriggeredStatus, ConceptOutputType, MissingDataPolicy
from app.models.task import Namespace
from app.runtime.action_trigger import ActionTrigger
from app.runtime.cache import ResultCache
from app.runtime.condition_evaluator import ConditionEvaluator
from app.runtime.data_resolver import DataResolver, MockConnector
from app.runtime.executor import ConceptExecutor


# ── Shared helpers ─────────────────────────────────────────────────────────────

_ENTITY   = "user_1"
_TS       = "2024-03-15T09:00:00Z"
_PRIM_ID  = "churn_probability"
_PRIM_VAL = 0.9


def _make_graph() -> ExecutionGraph:
    """
    Minimal concept graph:
      prim:churn_probability  (primitive_fetch, float)
      feat:churn_score        (normalize, input → prim)
    """
    prim_node = GraphNode(
        node_id="prim:churn_probability",
        op="primitive_fetch",
        inputs={},
        params={
            "source_name": _PRIM_ID,
            "declared_type": "float",
            "missing_data_policy": "zero",
        },
        output_type="float",
    )
    feat_node = GraphNode(
        node_id="feat:churn_score",
        op="normalize",
        inputs={"input": "prim:churn_probability"},
        params={},
        output_type="float",
    )
    edge = GraphEdge(
        from_node_id="prim:churn_probability",
        to_node_id="feat:churn_score",
        input_slot="input",
    )
    return ExecutionGraph(
        graph_id="graph_001",
        concept_id="org.churn_risk_score",
        version="1.0",
        ir_hash="a" * 64,
        nodes=[prim_node, feat_node],
        edges=[edge],
        topological_order=["prim:churn_probability", "feat:churn_score"],
        parallelizable_groups=[["prim:churn_probability"], ["feat:churn_score"]],
        output_node_id="feat:churn_score",
        output_type="float",
    )


def _make_connector(snapshot_val: float | None = None) -> MockConnector:
    """
    Connector with fixed deterministic data for timestamp _TS and a
    configurable snapshot value for timestamp=None.
    """
    data = {
        (_PRIM_ID, _ENTITY, _TS): _PRIM_VAL,
    }
    if snapshot_val is not None:
        data[(_PRIM_ID, _ENTITY, None)] = snapshot_val
    return MockConnector(data=data)


def _make_executor(connector: MockConnector | None = None) -> tuple[ConceptExecutor, ResultCache]:
    cache = ResultCache()
    exec_ = ConceptExecutor(result_cache=cache)
    return exec_, cache


def _resolver(connector: MockConnector) -> DataResolver:
    return DataResolver(
        connector=connector,
        missing_policy=MissingDataPolicy.ZERO,
        backoff_base=0.0,
    )


def _run(executor, graph, connector, ts=_TS):
    return executor.execute_graph(
        graph=graph,
        entity=_ENTITY,
        data_resolver=_resolver(connector),
        timestamp=ts,
    )


# ── 1. Determinism: same inputs → same output ─────────────────────────────────

class TestDeterminism:
    def test_same_inputs_three_runs_produce_identical_results(self):
        graph     = _make_graph()
        connector = _make_connector()
        executor, _cache = _make_executor(connector)

        results = [
            _run(executor, graph, connector).value
            for _ in range(3)
        ]
        assert results[0] == results[1] == results[2]

    def test_different_entities_produce_independent_results(self):
        graph  = _make_graph()
        data   = {
            (_PRIM_ID, "user_1", _TS): 0.9,
            (_PRIM_ID, "user_2", _TS): 0.2,
        }
        connector = MockConnector(data=data)
        executor, _ = _make_executor()

        r1 = executor.execute_graph(graph=graph, entity="user_1",
                                    data_resolver=_resolver(connector), timestamp=_TS)
        r2 = executor.execute_graph(graph=graph, entity="user_2",
                                    data_resolver=_resolver(connector), timestamp=_TS)
        # Different primitives → different outputs.
        assert r1.value != r2.value

    def test_output_is_float_for_float_output_type(self):
        graph    = _make_graph()
        conn     = _make_connector()
        executor, _ = _make_executor()
        result = _run(executor, graph, conn)
        assert isinstance(result.value, float)
        assert result.type == ConceptOutputType.FLOAT


# ── 2. Timestamp present → deterministic=True ────────────────────────────────

class TestTimestampPresent:
    def test_deterministic_true_when_timestamp_present(self):
        graph = _make_graph()
        conn  = _make_connector()
        executor, _ = _make_executor()
        result = _run(executor, graph, conn, ts=_TS)
        assert result.deterministic is True

    def test_timestamp_propagated_to_result(self):
        graph = _make_graph()
        conn  = _make_connector()
        executor, _ = _make_executor()
        result = _run(executor, graph, conn, ts=_TS)
        assert result.timestamp == _TS

    def test_deterministic_result_stored_in_cache(self):
        graph = _make_graph()
        conn  = _make_connector()
        executor, cache = _make_executor(conn)
        _run(executor, graph, conn, ts=_TS)
        cache_key = ("org.churn_risk_score", "1.0", _ENTITY, _TS)
        assert cache.get(cache_key) is not None

    def test_cache_hit_returns_same_result(self):
        graph = _make_graph()
        conn  = _make_connector()
        executor, _ = _make_executor(conn)
        r1 = _run(executor, graph, conn, ts=_TS)
        r2 = _run(executor, graph, conn, ts=_TS)
        assert r1.value == r2.value
        assert r1.deterministic is True


# ── 3. Timestamp absent → deterministic=False ────────────────────────────────

class TestTimestampAbsent:
    def test_deterministic_false_when_no_timestamp(self):
        graph = _make_graph()
        conn  = _make_connector(snapshot_val=0.5)
        executor, _ = _make_executor()
        result = _run(executor, graph, conn, ts=None)
        assert result.deterministic is False

    def test_timestamp_is_none_on_result(self):
        graph = _make_graph()
        conn  = _make_connector(snapshot_val=0.5)
        executor, _ = _make_executor()
        result = _run(executor, graph, conn, ts=None)
        assert result.timestamp is None


# ── 4. Snapshot result NOT stored in cache ────────────────────────────────────

class TestSnapshotNotCached:
    def test_snapshot_result_not_in_cache_after_execution(self):
        graph = _make_graph()
        conn  = _make_connector(snapshot_val=0.5)
        executor, cache = _make_executor(conn)

        _run(executor, graph, conn, ts=None)

        cache_key = ("org.churn_risk_score", "1.0", _ENTITY, None)
        assert cache.get(cache_key) is None  # must not be cached

    def test_cache_size_unchanged_after_snapshot(self):
        graph = _make_graph()
        conn  = _make_connector(snapshot_val=0.5)
        executor, cache = _make_executor(conn)

        _run(executor, graph, conn, ts=None)
        _run(executor, graph, conn, ts=None)
        _run(executor, graph, conn, ts=None)

        assert len(cache) == 0  # cache must remain empty for snapshots

    def test_snapshot_and_deterministic_keys_are_independent(self):
        graph = _make_graph()
        data  = {
            (_PRIM_ID, _ENTITY, _TS):  0.9,
            (_PRIM_ID, _ENTITY, None): 0.5,
        }
        conn = MockConnector(data=data)
        executor, cache = _make_executor(conn)

        _run(executor, graph, conn, ts=_TS)   # deterministic → cached
        _run(executor, graph, conn, ts=None)  # snapshot     → NOT cached

        assert len(cache) == 1  # only the deterministic entry
        assert cache.get(("org.churn_risk_score", "1.0", _ENTITY, None)) is None
        assert cache.get(("org.churn_risk_score", "1.0", _ENTITY, _TS))  is not None


# ── 5. dry_run → actions have status='would_trigger' ─────────────────────────

class TestDryRun:
    def _make_actions(self) -> list[ActionDefinition]:
        return [
            ActionDefinition(
                action_id="org.notify_team",
                version="1.0",
                namespace=Namespace.ORG,
                config=NotificationActionConfig(type="notification", channel="slack"),
                trigger=TriggerConfig(
                    fire_on=FireOn.TRUE,
                    condition_id="org.high_churn_risk",
                    condition_version="1.0",
                ),
            )
        ]

    def _true_decision(self) -> DecisionValue:
        return DecisionValue(
            value=True,
            decision_type=DecisionType.BOOLEAN,
            condition_id="org.high_churn_risk",
            condition_version="1.0",
            entity=_ENTITY,
        )

    def test_dry_run_all_actions_would_trigger(self):
        trigger = ActionTrigger()
        decision = self._true_decision()
        results = asyncio.run(trigger.trigger_bound_actions(decision, self._make_actions(), dry_run=True))
        assert all(r.status == ActionTriggeredStatus.WOULD_TRIGGER for r in results)

    def test_dry_run_no_actual_dispatch(self):
        dispatch_called = []

        def dispatcher(action, decision):
            dispatch_called.append(action.action_id)
            return {}

        trigger = ActionTrigger(dispatcher=dispatcher)
        decision = self._true_decision()
        asyncio.run(trigger.trigger_bound_actions(decision, self._make_actions(), dry_run=True))
        assert dispatch_called == []  # dispatcher must NOT be invoked in dry_run


# ── 6. Action failure does NOT fail the pipeline ─────────────────────────────

class TestActionFailureIsolation:
    def _make_action(self) -> ActionDefinition:
        return ActionDefinition(
            action_id="org.flaky_action",
            version="1.0",
            namespace=Namespace.ORG,
            config=NotificationActionConfig(type="notification", channel="slack"),
            trigger=TriggerConfig(
                fire_on=FireOn.TRUE,
                condition_id="org.high_churn_risk",
                condition_version="1.0",
            ),
        )

    def _true_decision(self) -> DecisionValue:
        return DecisionValue(
            value=True,
            decision_type=DecisionType.BOOLEAN,
            condition_id="org.high_churn_risk",
            condition_version="1.0",
            entity=_ENTITY,
        )

    def test_failed_action_does_not_raise(self):
        def failing_dispatcher(action, decision):
            raise RuntimeError("Simulated delivery failure")

        trigger  = ActionTrigger(dispatcher=failing_dispatcher)
        decision = self._true_decision()

        # Must NOT raise — best-effort contract.
        results = asyncio.run(trigger.trigger_bound_actions(decision, [self._make_action()], dry_run=False))
        assert results[0].status == ActionTriggeredStatus.FAILED

    def test_failed_action_error_is_captured(self):
        def failing_dispatcher(action, decision):
            raise RuntimeError("Delivery timeout")

        trigger  = ActionTrigger(dispatcher=failing_dispatcher)
        decision = self._true_decision()
        results  = asyncio.run(trigger.trigger_bound_actions(decision, [self._make_action()]))
        assert results[0].error is not None
        assert "Delivery timeout" in results[0].error.error.message

    def test_one_failed_action_does_not_block_subsequent_actions(self):
        actions = [
            ActionDefinition(
                action_id="org.failing",
                version="1.0",
                namespace=Namespace.ORG,
                config=NotificationActionConfig(type="notification", channel="ch"),
                trigger=TriggerConfig(
                    fire_on=FireOn.TRUE,
                    condition_id="c1", condition_version="1.0",
                ),
            ),
            ActionDefinition(
                action_id="org.succeeding",
                version="1.0",
                namespace=Namespace.ORG,
                config=NotificationActionConfig(type="notification", channel="ch"),
                trigger=TriggerConfig(
                    fire_on=FireOn.TRUE,
                    condition_id="c1", condition_version="1.0",
                ),
            ),
        ]

        calls = []

        def dispatcher(action, decision):
            calls.append(action.action_id)
            if action.action_id == "org.failing":
                raise RuntimeError("fail")
            return {}

        trigger  = ActionTrigger(dispatcher=dispatcher)
        decision = DecisionValue(
            value=True, decision_type=DecisionType.BOOLEAN,
            condition_id="c1", condition_version="1.0", entity=_ENTITY,
        )
        results = asyncio.run(trigger.trigger_bound_actions(decision, actions))

        assert results[0].status == ActionTriggeredStatus.FAILED
        assert results[1].status == ActionTriggeredStatus.TRIGGERED
        assert "org.succeeding" in calls


# ── 7. Actions do NOT retry on failure (at-most-once) ────────────────────────

class TestAtMostOnce:
    def test_action_attempted_exactly_once_on_failure(self):
        call_count = []

        def dispatcher(action, decision):
            call_count.append(1)
            raise RuntimeError("always fails")

        action = ActionDefinition(
            action_id="org.once",
            version="1.0",
            namespace=Namespace.ORG,
            config=NotificationActionConfig(type="notification", channel="ch"),
            trigger=TriggerConfig(
                fire_on=FireOn.ANY,
                condition_id="c1", condition_version="1.0",
            ),
        )
        trigger  = ActionTrigger(dispatcher=dispatcher)
        decision = DecisionValue(
            value=True, decision_type=DecisionType.BOOLEAN,
            condition_id="c1", condition_version="1.0", entity=_ENTITY,
        )
        asyncio.run(trigger.trigger_bound_actions(decision, [action]))

        assert len(call_count) == 1  # exactly one attempt, no retry


# ── 8. Same evaluation twice fires actions twice (no cross-call dedup) ────────

class TestNoCrossCallDedup:
    def test_same_entity_timestamp_fires_actions_twice(self):
        call_count = []

        def dispatcher(action, decision):
            call_count.append(1)
            return {}

        action = ActionDefinition(
            action_id="org.count",
            version="1.0",
            namespace=Namespace.ORG,
            config=NotificationActionConfig(type="notification", channel="ch"),
            trigger=TriggerConfig(
                fire_on=FireOn.TRUE,
                condition_id="c1", condition_version="1.0",
            ),
        )
        trigger  = ActionTrigger(dispatcher=dispatcher)
        decision = DecisionValue(
            value=True, decision_type=DecisionType.BOOLEAN,
            condition_id="c1", condition_version="1.0", entity=_ENTITY,
        )

        asyncio.run(trigger.trigger_bound_actions(decision, [action]))
        asyncio.run(trigger.trigger_bound_actions(decision, [action]))

        # Each call fires once — total 2 (no cross-call deduplication)
        assert len(call_count) == 2

    def test_skipped_action_not_counted_as_triggered(self):
        call_count = []

        def dispatcher(action, decision):
            call_count.append(1)
            return {}

        action = ActionDefinition(
            action_id="org.fires_on_false",
            version="1.0",
            namespace=Namespace.ORG,
            config=NotificationActionConfig(type="notification", channel="ch"),
            trigger=TriggerConfig(
                fire_on=FireOn.FALSE,   # fires when decision=False
                condition_id="c1", condition_version="1.0",
            ),
        )
        trigger  = ActionTrigger(dispatcher=dispatcher)
        true_decision = DecisionValue(
            value=True, decision_type=DecisionType.BOOLEAN,
            condition_id="c1", condition_version="1.0", entity=_ENTITY,
        )
        results = asyncio.run(trigger.trigger_bound_actions(true_decision, [action]))
        assert results[0].status == ActionTriggeredStatus.SKIPPED
        assert len(call_count) == 0  # not dispatched


# ── 9. None-input guard on numeric operators ──────────────────────────────────

class TestNoneInputGuard:
    """
    Operators must not crash when an upstream primitive_fetch returns None
    (e.g. missing_data_policy='null' and the connector found no data).

    Each test calls the operator function directly so we can check the
    returned value without building a full graph.
    """

    def test_z_score_op_none_input_propagates_none(self):
        """z_score_op with None input and no policy → returns None (null propagation)."""
        from app.runtime.executor import _op_z_score_op
        result = _op_z_score_op({"input": None}, {})
        assert result is None

    def test_normalize_none_input_zero_policy_substitutes_zero(self):
        """normalize with None input and missing_data_policy='zero' → normalize(0.0) = 0.0."""
        from app.runtime.executor import _op_normalize
        # _guard_none sees policy='zero' → substitutes 0.0 → normalize(0.0) = 0.0/(1+0.0) = 0.0
        result = _op_normalize({"input": None}, {"missing_data_policy": "zero"})
        assert result == 0.0

    def test_add_none_input_propagates_none(self):
        """Binary operator with one None operand propagates None regardless of other input."""
        from app.runtime.executor import _op_add
        assert _op_add({"a": None, "b": 5.0}, {}) is None
        assert _op_add({"a": 3.0,  "b": None}, {}) is None


# ── FIX 2: ConceptExecutor.aexecute() properly awaits get_by_concept ─────────

class TestAexecute:
    """
    aexecute() is the async entry point that properly awaits
    GraphStore.get_by_concept() — fixing the missing-await bug in execute().
    """

    def test_aexecute_returns_concept_result_not_coroutine(self):
        """
        aexecute() with a mock async graph store returns a real ConceptResult,
        not a coroutine object. This confirms get_by_concept() is awaited.
        """
        import asyncio

        graph = _make_graph()
        connector = _make_connector()
        cache = ResultCache()

        class _AsyncGraphStore:
            """Minimal async graph store stub."""
            async def get_by_concept(self, concept_id: str, version: str):
                return graph

        executor = ConceptExecutor(result_cache=cache, graph_store=_AsyncGraphStore())
        resolver = DataResolver(connector=connector, backoff_base=0.0)

        result = asyncio.run(executor.aexecute(
            concept_id="org.churn_risk_score",
            version="1.0",
            entity=_ENTITY,
            data_resolver=resolver,
            timestamp=_TS,
        ))

        # Must be a real ConceptResult, not a coroutine.
        from app.models.result import ConceptResult
        assert isinstance(result, ConceptResult), (
            f"Expected ConceptResult, got {type(result).__name__}. "
            "This means get_by_concept() was not awaited."
        )
        assert result.entity == _ENTITY
        assert result.deterministic is True

    def test_aexecute_raises_not_found_when_graph_missing(self):
        """aexecute() raises MemintelError(NOT_FOUND) when graph_store returns None."""
        import asyncio
        from app.models.errors import MemintelError, ErrorType

        cache = ResultCache()

        class _EmptyGraphStore:
            async def get_by_concept(self, concept_id, version):
                return None

        executor = ConceptExecutor(result_cache=cache, graph_store=_EmptyGraphStore())
        resolver = DataResolver(connector=MockConnector(data={}), backoff_base=0.0)

        with pytest.raises(MemintelError) as exc_info:
            asyncio.run(executor.aexecute(
                concept_id="missing.concept",
                version="1.0",
                entity=_ENTITY,
                data_resolver=resolver,
            ))

        assert exc_info.value.error_type == ErrorType.NOT_FOUND

    def test_aexecute_raises_runtime_error_without_graph_store(self):
        """aexecute() raises RuntimeError if no graph_store was injected."""
        import asyncio

        cache = ResultCache()
        executor = ConceptExecutor(result_cache=cache)  # no graph_store
        resolver = DataResolver(connector=MockConnector(data={}), backoff_base=0.0)

        with pytest.raises(RuntimeError, match="graph_store"):
            asyncio.run(executor.aexecute(
                concept_id="org.x",
                version="1.0",
                entity=_ENTITY,
                data_resolver=resolver,
            ))
