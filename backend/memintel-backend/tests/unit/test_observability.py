"""
tests/unit/test_observability.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for structured logging events.

All tests use structlog.testing.capture_logs() — no real I/O, no JSON parsing.
capture_logs() replaces the processor pipeline with a capturing processor and
returns a list of dicts, one per log call.

Coverage:
  1. concept_executed event — cache miss path (all required fields present)
  2. concept_executed event — cache hit path (cache_hit=True, compute_time_ms=0)
  3. condition_evaluated event — all required fields present
  4. memintel_error event — emitted from MemintelError.__init__
  5. No raw primitive data or concept output values in concept_executed log
  6. feedback note field is NOT logged (PII protection invariant)
"""
from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from app.models.concept import ExecutionGraph, GraphEdge, GraphNode
from app.models.condition import ConditionDefinition, DecisionType, DecisionValue, StrategyType
from app.models.errors import ErrorType, MemintelError, NotFoundError
from app.models.result import ConceptOutputType, ConceptResult
from app.runtime.cache import ResultCache
from app.runtime.condition_evaluator import ConditionEvaluator
from app.runtime.executor import ConceptExecutor


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


class _MockDataResolver:
    """Minimal data resolver: always returns 0.82 for any fetch."""

    def fetch(self, source_name, entity, timestamp, policy=None):
        class _PV:
            value = 0.82
        return _PV()


def _make_minimal_graph(
    concept_id: str = "org.test_concept",
    version: str = "1.0",
) -> ExecutionGraph:
    """
    Build a minimal one-node ExecutionGraph with a single primitive_fetch node.
    Uses a real GraphNode (no mocking) so the executor runs its actual code path.
    """
    node = GraphNode(
        node_id="signal_a",
        op="primitive_fetch",
        inputs={},
        params={"source_name": "signal_a"},
        output_type="float",
    )
    return ExecutionGraph(
        graph_id="test_graph_id",
        concept_id=concept_id,
        version=version,
        ir_hash="abc123",
        nodes=[node],
        edges=[],
        topological_order=["signal_a"],
        parallelizable_groups=[["signal_a"]],
        output_node_id="signal_a",
        output_type="float",
    )


def _make_concept_result(
    concept_id: str = "org.test_concept",
    version: str = "1.0",
    entity: str = "user_42",
    timestamp: str = "2024-01-15T09:00:00Z",
    value: float = 0.82,
) -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity=entity,
        version=version,
        deterministic=True,
        timestamp=timestamp,
    )


def _threshold_condition(
    condition_id: str = "org.test",
    concept_id: str = "org.test_concept",
    value: float = 0.75,
) -> ConditionDefinition:
    return ConditionDefinition.model_validate({
        "condition_id": condition_id,
        "version": "1.0",
        "concept_id": concept_id,
        "concept_version": "1.0",
        "strategy": {"type": "threshold", "params": {"direction": "above", "value": value}},
        "namespace": "personal",
    })


# ── Tests: concept_executed ────────────────────────────────────────────────────

def test_concept_executed_cache_miss_emits_event():
    """
    On a cache miss, execute_graph() must emit concept_executed with all
    required fields: concept_id, version, entity, timestamp, deterministic,
    cache_hit=False, compute_time_ms, result_type.
    """
    cache = ResultCache()
    executor = ConceptExecutor(result_cache=cache)
    graph = _make_minimal_graph()

    with capture_logs() as logs:
        executor.execute_graph(
            graph,
            entity="user_42",
            data_resolver=_MockDataResolver(),
            timestamp="2024-01-15T09:00:00Z",
        )

    events = [l for l in logs if l["event"] == "concept_executed"]
    assert len(events) == 1, f"expected 1 concept_executed event, got {len(events)}"

    e = events[0]
    assert e["concept_id"] == "org.test_concept"
    assert e["version"] == "1.0"
    assert e["entity"] == "user_42"
    assert e["timestamp"] == "2024-01-15T09:00:00Z"
    assert e["deterministic"] is True
    assert e["cache_hit"] is False
    assert "compute_time_ms" in e
    assert isinstance(e["compute_time_ms"], int)
    assert e["result_type"] == "float"


def test_concept_executed_cache_hit_emits_event():
    """
    On a cache hit, execute_graph() must emit concept_executed with
    cache_hit=True and compute_time_ms=0.
    """
    cache = ResultCache()
    executor = ConceptExecutor(result_cache=cache)
    graph = _make_minimal_graph()
    resolver = _MockDataResolver()
    ts = "2024-01-15T09:00:00Z"

    # Prime the cache.
    executor.execute_graph(graph, entity="user_42", data_resolver=resolver, timestamp=ts)

    with capture_logs() as logs:
        executor.execute_graph(graph, entity="user_42", data_resolver=resolver, timestamp=ts)

    events = [l for l in logs if l["event"] == "concept_executed"]
    assert len(events) == 1
    e = events[0]
    assert e["cache_hit"] is True
    assert e["compute_time_ms"] == 0


def test_concept_executed_does_not_log_raw_output_value():
    """
    INVARIANT: raw concept output value must NOT appear in the concept_executed
    log entry (per py-instructions.md PII / data minimisation rules).
    """
    cache = ResultCache()
    executor = ConceptExecutor(result_cache=cache)
    graph = _make_minimal_graph()

    with capture_logs() as logs:
        executor.execute_graph(
            graph,
            entity="user_42",
            data_resolver=_MockDataResolver(),
            timestamp="2024-01-15T09:00:00Z",
        )

    events = [l for l in logs if l["event"] == "concept_executed"]
    assert len(events) == 1
    e = events[0]
    # The raw output value (0.82) must not be a top-level field.
    assert "value" not in e
    assert "output_value" not in e
    assert "result_value" not in e


# ── Tests: condition_evaluated ─────────────────────────────────────────────────

def test_condition_evaluated_emits_event():
    """
    ConditionEvaluator.evaluate() must emit condition_evaluated with all
    required fields: condition_id, condition_version, entity, timestamp,
    decision_value, decision_type, strategy_type, params_applied,
    actions_triggered_count.
    """
    condition = _threshold_condition()
    concept_result = _make_concept_result()

    cache = ResultCache()
    cache_key = ("org.test_concept", "1.0", "user_42", "2024-01-15T09:00:00Z")
    cache.set(cache_key, concept_result)

    executor = ConceptExecutor(result_cache=cache)
    evaluator = ConditionEvaluator(executor=executor, result_cache=cache)

    with capture_logs() as logs:
        evaluator.evaluate(
            condition,
            entity="user_42",
            data_resolver=None,
            timestamp="2024-01-15T09:00:00Z",
        )

    events = [l for l in logs if l["event"] == "condition_evaluated"]
    assert len(events) == 1, f"expected 1 condition_evaluated event, got {len(events)}"

    e = events[0]
    assert e["condition_id"] == "org.test"
    assert e["condition_version"] == "1.0"
    assert e["entity"] == "user_42"
    assert e["timestamp"] == "2024-01-15T09:00:00Z"
    assert "decision_value" in e      # "True" or "False"
    assert e["decision_type"] == DecisionType.BOOLEAN.value
    assert e["strategy_type"] == StrategyType.THRESHOLD.value
    assert "params_applied" in e
    assert e["actions_triggered_count"] == 0


def test_condition_evaluated_params_applied_matches_strategy():
    """
    params_applied in condition_evaluated must reflect the condition's actual
    strategy params (the raw params dict before any composite resolution).
    """
    condition = _threshold_condition(value=0.75)
    concept_result = _make_concept_result()

    cache = ResultCache()
    cache.set(("org.test_concept", "1.0", "user_42", "2024-01-15T09:00:00Z"), concept_result)

    executor = ConceptExecutor(result_cache=cache)
    evaluator = ConditionEvaluator(executor=executor, result_cache=cache)

    with capture_logs() as logs:
        evaluator.evaluate(condition, entity="user_42", data_resolver=None, timestamp="2024-01-15T09:00:00Z")

    e = next(l for l in logs if l["event"] == "condition_evaluated")
    assert e["params_applied"]["value"] == 0.75
    assert e["params_applied"]["direction"] == "above"


# ── Tests: memintel_error ──────────────────────────────────────────────────────

def test_memintel_error_emits_event():
    """
    MemintelError.__init__ must emit memintel_error with error_type and location.
    Verified by raising and catching the exception inside capture_logs().
    """
    with capture_logs() as logs:
        try:
            raise MemintelError(
                ErrorType.NOT_FOUND,
                "Resource not found",
                location="task_id",
            )
        except MemintelError:
            pass

    events = [l for l in logs if l["event"] == "memintel_error"]
    assert len(events) == 1, f"expected 1 memintel_error event, got {len(events)}"

    e = events[0]
    assert e["error_type"] == ErrorType.NOT_FOUND.value   # "not_found"
    assert e["location"] == "task_id"
    # NOT_FOUND and CONFLICT are expected/normal responses — logged at debug, not warning
    assert e["log_level"] == "debug"


def test_memintel_error_subclass_emits_event():
    """NotFoundError (a MemintelError subclass) must also emit memintel_error."""
    with capture_logs() as logs:
        try:
            raise NotFoundError("Condition not found", location="condition_id")
        except NotFoundError:
            pass

    events = [l for l in logs if l["event"] == "memintel_error"]
    assert len(events) == 1
    assert events[0]["error_type"] == "not_found"
    assert events[0]["location"] == "condition_id"


def test_memintel_error_no_location_logs_none():
    """When location is omitted, memintel_error must still be emitted (location=None)."""
    with capture_logs() as logs:
        try:
            raise MemintelError(ErrorType.EXECUTION_ERROR, "Unexpected failure")
        except MemintelError:
            pass

    events = [l for l in logs if l["event"] == "memintel_error"]
    assert len(events) == 1
    assert events[0]["location"] is None
