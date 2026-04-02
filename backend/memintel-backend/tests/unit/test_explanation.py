"""
tests/unit/test_explanation.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for ExplanationService.

All tests run synchronously via asyncio.run() — no pytest-asyncio required.

Coverage for explain_decision() (FIX 2 — stored-record approach):
  1. threshold strategy → threshold_applied == stored params.value
  2. z_score strategy   → threshold_applied == stored params.threshold
  3. equals strategy    → threshold_applied is None; label_matched from concept_value
  4. composite strategy → both threshold_applied and label_matched are None
  5. drivers contribution sums to 1.0 (equal-weight, 2 signals)
  6. equal-weight contributions with 3 signals also sum to 1.0
  7. driver.value comes from stored input_primitives values
  8. empty input_primitives returns empty drivers list
  9. driver.signal names preserved from input_primitives keys
  10. executor is NOT called — explanation built from stored record
  11. NotFoundError raised when decision_store returns None (record not found)

Coverage for explain_condition() (definition-level, no entity required):
  12. threshold strategy → ConditionExplanation with summary containing threshold value
  13. z_score strategy   → ConditionExplanation with summary containing z-score value
  14. equals strategy    → ConditionExplanation with summary containing matched value
  15. not-found condition → NotFoundError (404)
  16. wiring from conditions route → ExplanationService._registry is not None

Wiring:
  17. get_explanation_service() (decisions.py) constructs a DataResolver
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
from app.models.decision import DecisionRecord
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


# ── Mock stores ────────────────────────────────────────────────────────────────

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


class MockDecisionStore:
    """Decision store stub for unit tests.

    decision_found=True  → returns a preset DecisionRecord (or a minimal default).
    decision_found=False → returns None → triggers NotFoundError in the service.
    """

    def __init__(
        self,
        record: DecisionRecord | None = None,
        decision_found: bool = True,
    ):
        self._record = record
        self._decision_found = decision_found
        self.find_calls: list[tuple] = []

    async def find_by_condition_entity_timestamp(
        self,
        condition_id: str,
        condition_version: str,
        entity_id: str,
        timestamp: str,
    ) -> DecisionRecord | None:
        self.find_calls.append((condition_id, condition_version, entity_id, timestamp))
        if not self._decision_found:
            return None
        return self._record


# ── DecisionRecord builder ─────────────────────────────────────────────────────

def _make_decision_record(
    *,
    condition_id: str = "org.test_threshold",
    condition_version: str = "1.0",
    entity_id: str = "user_42",
    fired: bool = True,
    concept_value: bool | float | int | str | None = 0.82,
    threshold_applied: dict | None = None,
    input_primitives: dict | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id="test-001",
        concept_id="org.test_concept",
        concept_version="1.0",
        condition_id=condition_id,
        condition_version=condition_version,
        entity_id=entity_id,
        fired=fired,
        concept_value=concept_value,
        threshold_applied=threshold_applied,
        input_primitives=input_primitives,
    )


# ── Service factory ────────────────────────────────────────────────────────────

def _make_service(
    *,
    condition_id: str,
    condition_version: str,
    body: dict,
    record: DecisionRecord | None = None,
    decision_found: bool = True,
) -> tuple[ExplanationService, MockDecisionStore]:
    registry = MockDefinitionRegistry()
    registry.seed(condition_id, condition_version, body)
    decision_store = MockDecisionStore(record=record, decision_found=decision_found)
    svc = ExplanationService(
        definition_registry=registry,
        concept_executor=None,   # not used by explain_decision (FIX 2)
        condition_evaluator=None,
        data_resolver=None,
        decision_store=decision_store,
    )
    return svc, decision_store


# ── Tests: explain_decision() ─────────────────────────────────────────────────

def test_threshold_strategy_threshold_applied_equals_params_value():
    """
    For threshold strategy, threshold_applied must equal the stored params.value.
    The value is extracted from the decision record's threshold_applied dict.
    """
    record = _make_decision_record(
        condition_id="org.test_threshold",
        threshold_applied={"direction": "above", "value": 0.75},
    )
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(value=0.75),
        record=record,
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
    For z_score strategy, threshold_applied must equal stored params.threshold
    (NOT params.value — z_score uses a different key).
    """
    record = _make_decision_record(
        condition_id="org.test_zscore",
        threshold_applied={"threshold": 2.5, "direction": "above", "window": "30d"},
    )
    svc, _ = _make_service(
        condition_id="org.test_zscore",
        condition_version="1.0",
        body=_z_score_body(threshold=2.5),
        record=record,
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
      - label_matched must equal the stored string concept_value
    """
    record = _make_decision_record(
        condition_id="org.test_equals",
        concept_value="premium",  # stored as string
        threshold_applied=None,
    )
    svc, _ = _make_service(
        condition_id="org.test_equals",
        condition_version="1.0",
        body=_equals_body(value="premium"),
        record=record,
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
    record = _make_decision_record(
        condition_id="org.test_composite",
        threshold_applied=None,
    )
    svc, _ = _make_service(
        condition_id="org.test_composite",
        condition_version="1.0",
        body=_composite_body(),
        record=record,
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
    INVARIANT: with 2 signals, equal-weight contributions (0.5 each) sum to 1.0.
    """
    record = _make_decision_record(
        input_primitives={"signal_a": 0.6, "signal_b": 0.4},
    )
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
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


def test_equal_weight_contributions_with_3_signals_sums_to_1():
    """
    INVARIANT: with 3 signals, equal-weight contributions (1/3 each) sum to 1.0.
    Verifies the invariant holds regardless of how many primitives are stored.
    """
    record = _make_decision_record(
        input_primitives={"signal_a": 1.0, "signal_b": 2.0, "signal_c": 3.0},
    )
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    total = sum(d.contribution for d in result.drivers)
    assert abs(total - 1.0) < 2e-6, (
        f"3-signal equal-weight sum must be 1.0, got {total}"
    )
    assert len(result.drivers) == 3


def test_driver_value_comes_from_input_primitives():
    """
    driver.value must equal the stored input_primitives value for each signal.
    The values are raw primitive fetched values, not contributions.
    """
    record = _make_decision_record(
        input_primitives={"signal_a": 0.92, "signal_b": 0.41},
    )
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    driver_map = {d.signal: d.value for d in result.drivers}
    assert driver_map["signal_a"] == 0.92, (
        f"expected signal_a value=0.92, got {driver_map.get('signal_a')}"
    )
    assert driver_map["signal_b"] == 0.41, (
        f"expected signal_b value=0.41, got {driver_map.get('signal_b')}"
    )


def test_empty_contributions_returns_empty_drivers():
    """When input_primitives is None, drivers list is empty."""
    record = _make_decision_record(input_primitives=None)
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.drivers == []


def test_driver_signal_names_preserved_from_input_primitives():
    """
    driver.signal must match the key from input_primitives.
    Verifies signal name fidelity through the stored-record path.
    """
    record = _make_decision_record(
        input_primitives={"session_count": 5, "page_views": 12},
    )
    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
    )

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    signal_names = {d.signal for d in result.drivers}
    assert signal_names == {"session_count", "page_views"}, (
        f"Expected signal names from input_primitives, got {signal_names}"
    )


def test_explain_decision_does_not_call_executor():
    """
    FIX 2: explain_decision() must NOT call concept executor.
    The explanation is built from the stored DecisionRecord.
    concept_executor=None must not cause AttributeError — confirms
    executor is never touched in the stored-record path.
    """
    record = _make_decision_record(
        condition_id="org.test_threshold",
        fired=True,
        concept_value=0.82,
        threshold_applied={"direction": "above", "value": 0.75},
        input_primitives={"signal_a": 0.82},
    )
    svc, ds = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        record=record,
    )

    # executor is None — would raise AttributeError if called
    assert svc._executor is None

    result = run(svc.explain_decision(
        condition_id="org.test_threshold",
        condition_version="1.0",
        entity="user_42",
        timestamp="2024-01-15T09:00:00Z",
    ))

    assert result.decision is True
    assert len(ds.find_calls) == 1, "decision_store.find_by_condition_entity_timestamp must be called exactly once"


def test_explain_decision_raises_not_found_when_record_not_found():
    """
    FIX 2: explain_decision() must raise NotFoundError (→ HTTP 404) when
    decision_store.find_by_condition_entity_timestamp() returns None.
    """
    from app.models.errors import NotFoundError

    svc, _ = _make_service(
        condition_id="org.test_threshold",
        condition_version="1.0",
        body=_threshold_body(),
        decision_found=False,  # store returns None
    )

    with pytest.raises(NotFoundError):
        run(svc.explain_decision(
            condition_id="org.test_threshold",
            condition_version="1.0",
            entity="user_42",
            timestamp="2024-01-15T09:00:00Z",
        ))


# ── Tests: wiring ─────────────────────────────────────────────────────────────

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
    # Provide a minimal request stub so the service can read app.state.
    class _FakeRequest:
        class app:
            class state:
                connector_registry = None
                config = None

    service = asyncio.run(get_explanation_service(request=_FakeRequest(), pool=_FakePool()))

    assert service._data_resolver is not None, (
        "ExplanationService.data_resolver must not be None"
    )
    assert isinstance(service._data_resolver, DataResolver), (
        f"Expected DataResolver, got {type(service._data_resolver).__name__}"
    )


# ── Tests: explain_condition() ─────────────────────────────────────────────────

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
