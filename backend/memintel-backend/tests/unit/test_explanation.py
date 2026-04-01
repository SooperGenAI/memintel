"""
tests/unit/test_explanation.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ExplanationService.explain_decision().

All tests run synchronously via asyncio.run() — no pytest-asyncio required.
The ConceptExecutor and ConditionEvaluator are stubbed; their sync calls
are wrapped inside the async explain_decision() method.

Coverage:
  1. threshold strategy → threshold_applied == params.value
  2. z_score strategy   → threshold_applied == params.threshold
  3. equals strategy    → threshold_applied is None; label_matched set
  4. composite strategy → both threshold_applied and label_matched are None
  5. drivers[].contribution sums to 1.0 (normalisation invariant)
  6. unnormalised contributions are normalised to 1.0 by the service
"""
from __future__ import annotations

import asyncio

import pytest

from app.models.condition import (
    ConditionDefinition,
    DecisionType,
    DecisionValue,
    StrategyType,
)
from app.models.result import (
    ConceptExplanation,
    ConceptOutputType,
    ConceptResult,
    NodeTrace,
)
from app.models.task import Namespace
from app.services.explanation import ExplanationService


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


# ── Condition body builders ────────────────────────────────────────────────────

def _threshold_body(value: float = 0.75) -> dict:
    return {
        "condition_id": "org.test_threshold",
        "version": "1.0",
        "concept_id": "org.test_concept",
        "concept_version": "1.0",
        "strategy": {
            "type": "threshold",
            "params": {"direction": "above", "value": value},
        },
        "namespace": "personal",
    }


def _z_score_body(threshold: float = 2.5) -> dict:
    return {
        "condition_id": "org.test_zscore",
        "version": "1.0",
        "concept_id": "org.test_concept",
        "concept_version": "1.0",
        "strategy": {
            "type": "z_score",
            "params": {"threshold": threshold, "direction": "above", "window": "30d"},
        },
        "namespace": "personal",
    }


def _equals_body(value: str = "premium") -> dict:
    return {
        "condition_id": "org.test_equals",
        "version": "1.0",
        "concept_id": "org.test_concept",
        "concept_version": "1.0",
        "strategy": {
            "type": "equals",
            "params": {"value": value},
        },
        "namespace": "personal",
    }


def _composite_body() -> dict:
    return {
        "condition_id": "org.test_composite",
        "version": "1.0",
        "concept_id": "org.test_concept",
        "concept_version": "1.0",
        "strategy": {
            "type": "composite",
            "params": {"operator": "AND", "operands": ["cond_a", "cond_b"]},
        },
        "namespace": "personal",
    }


# ── Mock executor and evaluator ────────────────────────────────────────────────

class MockConceptExecutor:
    """Returns a preset ConceptResult on every execute() / aexecute() call."""

    def __init__(self, concept_result: ConceptResult):
        self._result = concept_result

    def execute(
        self,
        concept_id,
        version,
        entity,
        data_resolver,
        timestamp=None,
        explain=False,
        **kwargs,
    ) -> ConceptResult:
        return self._result

    async def aexecute(
        self,
        concept_id,
        version,
        entity,
        data_resolver,
        timestamp=None,
        explain=False,
        **kwargs,
    ) -> ConceptResult:
        return self._result


class MockConditionEvaluator:
    """Returns a preset DecisionValue on every evaluate() call."""

    def __init__(self, decision: DecisionValue):
        self._decision = decision

    def evaluate(self, condition, entity, data_resolver, timestamp=None, **kwargs) -> DecisionValue:
        return self._decision


class MockDefinitionRegistry:
    """Registry stub that serves pre-seeded condition bodies."""

    def __init__(self, bodies: dict[tuple, dict] | None = None):
        self._bodies: dict[tuple, dict] = bodies or {}

    def seed(self, condition_id: str, version: str, body: dict) -> None:
        self._bodies[(condition_id, version)] = body

    async def get(self, definition_id: str, version: str) -> dict:
        from app.models.errors import NotFoundError
        key = (definition_id, version)
        if key not in self._bodies:
            raise NotFoundError(f"Condition '{definition_id}' version '{version}' not found.")
        return self._bodies[key]


# ── ConceptResult builders ────────────────────────────────────────────────────

def _make_concept_result(
    value: float = 0.82,
    contributions: dict[str, float] | None = None,
    nodes: list[NodeTrace] | None = None,
) -> ConceptResult:
    """Build a ConceptResult with an explanation."""
    if contributions is None:
        contributions = {"signal_a": 0.6, "signal_b": 0.4}

    explanation = ConceptExplanation(
        output=value,
        contributions=contributions,
        nodes=nodes or [],
    )
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity="user_42",
        version="1.0",
        deterministic=True,
        timestamp="2024-01-15T09:00:00Z",
        explanation=explanation,
    )


def _make_categorical_result(label: str = "premium") -> ConceptResult:
    """Build a ConceptResult for a categorical concept."""
    explanation = ConceptExplanation(
        output=label,
        contributions={"signal_a": 0.7, "signal_b": 0.3},
    )
    return ConceptResult(
        value=label,
        type=ConceptOutputType.CATEGORICAL,
        entity="user_42",
        version="1.0",
        deterministic=True,
        timestamp="2024-01-15T09:00:00Z",
        explanation=explanation,
    )


def _make_service(
    *,
    condition_id: str,
    condition_version: str,
    body: dict,
    concept_result: ConceptResult,
    decision: DecisionValue,
) -> ExplanationService:
    registry = MockDefinitionRegistry()
    registry.seed(condition_id, condition_version, body)

    executor = MockConceptExecutor(concept_result)
    evaluator = MockConditionEvaluator(decision)

    return ExplanationService(
        definition_registry=registry,
        concept_executor=executor,
        condition_evaluator=evaluator,
        data_resolver=None,
    )


def _boolean_decision(
    condition_id: str = "org.test_threshold",
    value: bool = True,
) -> DecisionValue:
    return DecisionValue(
        value=value,
        decision_type=DecisionType.BOOLEAN,
        condition_id=condition_id,
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    )


def _categorical_decision(
    condition_id: str = "org.test_equals",
    label: str = "premium",
) -> DecisionValue:
    return DecisionValue(
        value=label,
        decision_type=DecisionType.CATEGORICAL,
        condition_id=condition_id,
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_threshold_strategy_threshold_applied_equals_params_value():
    """
    For threshold strategy, threshold_applied must equal the params.value field.
    """
    body = _threshold_body(value=0.75)
    concept_result = _make_concept_result()
    decision = _boolean_decision(condition_id="org.test_threshold")

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.threshold_applied == 0.75, (
        f"threshold strategy: expected threshold_applied=0.75, got {result.threshold_applied}"
    )
    assert result.label_matched is None
    assert result.strategy_type == StrategyType.THRESHOLD


def test_z_score_strategy_threshold_applied_equals_params_threshold():
    """
    For z_score strategy, threshold_applied must equal params.threshold
    (NOT params.value — z_score uses a different key).
    """
    body = _z_score_body(threshold=2.5)
    concept_result = _make_concept_result()
    decision = _boolean_decision(condition_id="org.test_zscore")

    svc = _make_service(
        condition_id="org.test_zscore",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_zscore",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.threshold_applied == 2.5, (
        f"z_score strategy: expected threshold_applied=2.5, got {result.threshold_applied}"
    )
    assert result.label_matched is None
    assert result.strategy_type == StrategyType.Z_SCORE


def test_equals_strategy_threshold_applied_is_none_label_matched_set():
    """
    For equals strategy:
      - threshold_applied must be None (no numeric threshold)
      - label_matched must equal the matched label from the decision value
    """
    body = _equals_body(value="premium")
    concept_result = _make_categorical_result(label="premium")
    decision = _categorical_decision(condition_id="org.test_equals", label="premium")

    svc = _make_service(
        condition_id="org.test_equals",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_equals",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.threshold_applied is None, (
        f"equals strategy: threshold_applied must be None, got {result.threshold_applied}"
    )
    assert result.label_matched == "premium", (
        f"equals strategy: expected label_matched='premium', got {result.label_matched}"
    )
    assert result.strategy_type == StrategyType.EQUALS


def test_composite_strategy_both_threshold_and_label_are_none():
    """
    For composite strategy, both threshold_applied and label_matched must be None.
    Composite uses AND/OR operator — not a scalar threshold or label match.
    """
    body = _composite_body()
    concept_result = _make_concept_result()
    decision = _boolean_decision(condition_id="org.test_composite")

    svc = _make_service(
        condition_id="org.test_composite",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_composite",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.threshold_applied is None, (
        f"composite strategy: threshold_applied must be None, got {result.threshold_applied}"
    )
    assert result.label_matched is None, (
        f"composite strategy: label_matched must be None, got {result.label_matched}"
    )
    assert result.strategy_type == StrategyType.COMPOSITE


def test_drivers_contribution_sums_to_1():
    """
    INVARIANT: sum of all drivers[].contribution must equal 1.0 (within floating-
    point tolerance) regardless of what the executor returns.
    """
    body = _threshold_body()
    concept_result = _make_concept_result(
        contributions={"signal_a": 0.6, "signal_b": 0.4}
    )
    decision = _boolean_decision()

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    total = sum(d.contribution for d in result.drivers)
    assert abs(total - 1.0) < 1e-6, (
        f"drivers contribution sum must be 1.0, got {total}"
    )
    assert len(result.drivers) == 2


def test_unnormalised_contributions_are_normalised_to_1():
    """
    When executor returns contributions that do not sum to 1.0
    (e.g. {a: 0.3, b: 0.45} = 0.75 total), the service normalises them.
    """
    body = _threshold_body()
    # Contributions that sum to 0.75 — service must scale them to 1.0.
    concept_result = _make_concept_result(
        contributions={"signal_a": 0.3, "signal_b": 0.45}
    )
    decision = _boolean_decision()

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    total = sum(d.contribution for d in result.drivers)
    assert abs(total - 1.0) < 1e-6, (
        f"normalised drivers contribution sum must be 1.0, got {total}"
    )


def test_get_explanation_service_injects_non_none_data_resolver():
    """
    get_explanation_service() (the FastAPI dependency in decisions.py) must
    construct ExplanationService with a real DataResolver — not None.

    Calling explain_decision() with data_resolver=None would raise
    AttributeError the moment the executor calls resolver.fetch(); this test
    verifies the dependency factory wires a DataResolver instance.
    """
    from app.runtime.data_resolver import DataResolver
    from app.api.routes.decisions import get_explanation_service

    # Simulate a minimal asyncpg pool that satisfies ExplanationService's stores.
    class _FakePool:
        async def fetchrow(self, *a, **kw): return None
        async def fetch(self, *a, **kw): return []
        async def fetchval(self, *a, **kw): return None

    # get_explanation_service is an async dependency — call it directly.
    service = asyncio.run(get_explanation_service(pool=_FakePool()))

    assert service._data_resolver is not None, (
        "ExplanationService.data_resolver must not be None"
    )
    assert isinstance(service._data_resolver, DataResolver), (
        f"Expected DataResolver, got {type(service._data_resolver).__name__}"
    )


def test_drivers_sorted_highest_first():
    """
    Drivers are listed highest contribution first.
    """
    body = _threshold_body()
    concept_result = _make_concept_result(
        contributions={"signal_b": 0.2, "signal_a": 0.8}
    )
    decision = _boolean_decision()

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    contributions = [d.contribution for d in result.drivers]
    assert contributions == sorted(contributions, reverse=True), (
        "Drivers must be sorted descending by contribution"
    )
    assert result.drivers[0].signal == "signal_a"


def test_empty_contributions_returns_empty_drivers():
    """When concept returns no contributions, drivers list is empty."""
    body = _threshold_body()
    concept_result = _make_concept_result(contributions={})
    decision = _boolean_decision()

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.drivers == []


def test_node_trace_values_used_for_driver_value():
    """
    When ConceptExplanation.nodes contains NodeTrace entries, their output_value
    is used as the driver value (matched by node_id == signal_name).
    """
    nodes = [
        NodeTrace(
            node_id="signal_a",
            op="primitive_fetch",
            inputs={},
            params={},
            output_value=0.92,
            output_type="float",
        ),
        NodeTrace(
            node_id="signal_b",
            op="primitive_fetch",
            inputs={},
            params={},
            output_value=0.41,
            output_type="float",
        ),
    ]
    body = _threshold_body()
    concept_result = _make_concept_result(
        contributions={"signal_a": 0.7, "signal_b": 0.3},
        nodes=nodes,
    )
    decision = _boolean_decision()

    svc = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=body,
        concept_result=concept_result,
        decision=decision,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    driver_map = {d.signal: d.value for d in result.drivers}
    assert driver_map["signal_a"] == 0.92
    assert driver_map["signal_b"] == 0.41


# ── FIX 1 + FIX 2: explain_condition tests ────────────────────────────────────

def _make_explain_condition_service(body: dict) -> ExplanationService:
    """Build an ExplanationService wired only with a registry (no executor/evaluator)."""
    registry = MockDefinitionRegistry()
    condition_id = body["condition_id"]
    version = body["version"]
    registry.seed(condition_id, version, body)
    return ExplanationService(
        definition_registry=registry,
        concept_executor=None,
        condition_evaluator=None,
        data_resolver=None,
    )


def test_explain_condition_threshold_returns_condition_explanation():
    """
    explain_condition() must return a ConditionExplanation with non-empty
    natural_language_summary and parameter_rationale for a threshold strategy.
    Must not raise TypeError or AttributeError — verifies FIX 1 (wiring) and
    FIX 2 (method existence + _explain_strategy helper).
    """
    from app.models.condition import ConditionExplanation, StrategyType

    body = _threshold_body(value=0.75)
    svc = _make_explain_condition_service(body)

    result = run(svc.explain_condition(
        condition_id="org.test_threshold",
        condition_version="1.0",
    ))

    assert isinstance(result, ConditionExplanation)
    assert result.condition_id == "org.test_threshold"
    assert result.condition_version == "1.0"
    assert result.strategy.type == StrategyType.THRESHOLD
    assert len(result.natural_language_summary) > 0
    assert len(result.parameter_rationale) > 0
    # Verify the threshold value appears in the rationale.
    assert "0.75" in result.natural_language_summary or "0.75" in result.parameter_rationale


def test_explain_condition_z_score_returns_condition_explanation():
    """explain_condition() works for z_score strategy."""
    from app.models.condition import ConditionExplanation, StrategyType

    body = _z_score_body(threshold=2.5)
    svc = _make_explain_condition_service(body)

    result = run(svc.explain_condition(
        condition_id="org.test_zscore",
        condition_version="1.0",
    ))

    assert isinstance(result, ConditionExplanation)
    assert result.strategy.type == StrategyType.Z_SCORE
    assert "2.5" in result.natural_language_summary or "2.5" in result.parameter_rationale


def test_explain_condition_equals_returns_condition_explanation():
    """explain_condition() works for equals strategy."""
    from app.models.condition import ConditionExplanation, StrategyType

    body = _equals_body(value="premium")
    svc = _make_explain_condition_service(body)

    result = run(svc.explain_condition(
        condition_id="org.test_equals",
        condition_version="1.0",
    ))

    assert isinstance(result, ConditionExplanation)
    assert result.strategy.type == StrategyType.EQUALS
    assert "premium" in result.natural_language_summary or "premium" in result.parameter_rationale


def test_explain_condition_not_found_raises_not_found_error():
    """
    explain_condition() must raise NotFoundError (→ HTTP 404) when the condition
    is not in the registry. Must never raise TypeError or AttributeError.
    """
    from app.models.errors import NotFoundError

    registry = MockDefinitionRegistry()  # empty — nothing seeded
    svc = ExplanationService(
        definition_registry=registry,
        concept_executor=None,
        condition_evaluator=None,
        data_resolver=None,
    )

    import pytest
    with pytest.raises(NotFoundError):
        run(svc.explain_condition(
            condition_id="does_not_exist",
            condition_version="9.9",
        ))


def test_explain_condition_service_wiring_from_conditions_route():
    """
    get_explanation_service() in conditions.py (FIX 1) must construct
    ExplanationService without TypeError — i.e. must NOT pass pool=pool
    as a keyword arg to ExplanationService.__init__.

    Calls the factory directly with a minimal fake pool and verifies the
    returned service has a non-None _registry.
    """
    from app.api.routes.conditions import get_explanation_service

    class _FakePool:
        async def fetchrow(self, *a, **kw): return None
        async def fetch(self, *a, **kw): return []
        async def fetchval(self, *a, **kw): return None

    service = run(get_explanation_service(pool=_FakePool()))

    assert service._registry is not None, (
        "ExplanationService._registry must not be None after FIX 1"
    )
    # executor/evaluator are None — explain_condition is definition-only
    assert service._executor is None
    assert service._evaluator is None


# ── FIX 1 + FIX 2: explain_decision() uses aexecute() not execute(explain_mode=…)

class _StrictMockExecutor:
    """
    Mock executor that raises TypeError if called with explain_mode kwarg
    (mimicking the real ConceptExecutor.execute() which has no such param),
    but succeeds on aexecute() calls.
    """

    def __init__(self, concept_result: ConceptResult):
        self._result = concept_result

    def execute(self, *args, **kwargs) -> ConceptResult:
        if "explain_mode" in kwargs:
            raise TypeError(
                f"execute() got an unexpected keyword argument 'explain_mode'"
            )
        return self._result

    async def aexecute(self, *args, **kwargs) -> ConceptResult:
        # aexecute() does NOT accept explain_mode — confirm it's not passed.
        if "explain_mode" in kwargs:
            raise TypeError(
                f"aexecute() got an unexpected keyword argument 'explain_mode'"
            )
        return self._result


def test_explain_decision_does_not_pass_explain_mode_to_executor():
    """
    FIX 1: explain_decision() must NOT pass explain_mode=ExplainMode.FULL
    to the executor. ConceptExecutor.execute() and aexecute() have no such
    parameter — passing it causes a TypeError on every real call.

    Uses _StrictMockExecutor which raises TypeError if explain_mode is
    passed to either execute() or aexecute().
    """
    body = _threshold_body(value=0.75)
    concept_result = _make_concept_result()
    decision = _boolean_decision()

    registry = MockDefinitionRegistry()
    registry.seed("org.test_threshold", "1.0", body)

    executor = _StrictMockExecutor(concept_result)
    evaluator = MockConditionEvaluator(decision)

    svc = ExplanationService(
        definition_registry=registry,
        concept_executor=executor,
        condition_evaluator=evaluator,
        data_resolver=None,
    )

    # Must not raise TypeError — FIX 1 removed the explain_mode kwarg.
    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.threshold_applied == 0.75


def test_explain_decision_calls_aexecute_not_execute():
    """
    FIX 2: explain_decision() must call self._executor.aexecute() (async),
    not self._executor.execute() (sync). Verifies the async path is taken.
    """
    call_log: list[str] = []

    class _TrackingExecutor:
        def execute(self, *args, **kwargs) -> ConceptResult:
            call_log.append("execute")
            return _make_concept_result()

        async def aexecute(self, *args, **kwargs) -> ConceptResult:
            call_log.append("aexecute")
            return _make_concept_result()

    body = _threshold_body()
    registry = MockDefinitionRegistry()
    registry.seed("org.test_threshold", "1.0", body)

    svc = ExplanationService(
        definition_registry=registry,
        concept_executor=_TrackingExecutor(),
        condition_evaluator=MockConditionEvaluator(_boolean_decision()),
        data_resolver=None,
    )

    run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert call_log == ["aexecute"], (
        f"explain_decision() must call aexecute(), not execute(). Got: {call_log}"
    )
