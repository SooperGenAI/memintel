"""
tests/unit/test_fetch_error_distinction.py
──────────────────────────────────────────────────────────────────────────────
6 tests verifying that connector failure (fetch_error) and legitimate null
input (null_input) are cleanly distinguishable at every layer:
  strategy output, decision record, signal_errors.

This separation is required for regulatory auditability: a failed connector
and a legitimately absent value are categorically different events.

  reason="fetch_error"  — connector raised ConnectorError; data quality issue.
  reason="null_input"   — primitive has no value for this entity at this time.
                          Expected domain behaviour; missing_data_policy applied.

Tests
─────
1. test_fetch_error_returns_fetch_error_reason
2. test_legitimate_null_returns_null_input_reason
3. test_fetch_error_distinguishable_from_null_input
4. test_fetch_error_in_signal_errors
5. test_composite_propagates_fetch_error
6. test_fetch_error_appears_in_all_strategies
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.models.condition import DecisionType, DecisionValue, ConditionDefinition
from app.models.result import ConceptOutputType, ConceptResult, MissingDataPolicy
from app.runtime.data_resolver import (
    AuthConnectorError,
    DataResolver,
    MockConnector,
    PrimitiveValue,
)
from app.services.execute import ExecuteService
from app.strategies.composite import CompositeStrategy


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _float_result(value, entity="e1", version="1.0", ts=None) -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity=entity,
        version=version,
        deterministic=ts is not None,
        timestamp=ts,
    )


def _cat_result(value, entity="e1", version="1.0") -> ConceptResult:
    return ConceptResult(
        value=value,
        type=ConceptOutputType.CATEGORICAL,
        entity=entity,
        version=version,
        deterministic=False,
    )


def _null_float(entity="e1") -> ConceptResult:
    return _float_result(None, entity=entity)


def _null_cat(entity="e1") -> ConceptResult:
    return _cat_result(None, entity=entity)


def _bool_dv(value: bool, reason: str | None = None) -> DecisionValue:
    return DecisionValue(
        value=value,
        decision_type=DecisionType.BOOLEAN,
        condition_id="op",
        condition_version="1.0",
        entity="e1",
        reason=reason,
    )


# ── Minimal condition stubs ────────────────────────────────────────────────────

_CONCEPT_BODY = {
    "concept_id": "org.metric",
    "version": "1.0",
    "namespace": "org",
    "output_type": "float",
    "primitives": {
        "metric": {"type": "float", "missing_data_policy": "null"}
    },
    "features": {
        "score": {
            "op": "normalize",
            "inputs": {"input": "metric"},
            "params": {},
        }
    },
    "output_feature": "score",
}

_THRESHOLD_CONDITION = {
    "condition_id": "metric_above",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.metric",
    "concept_version": "1.0",
    "strategy": {"type": "threshold", "params": {"direction": "above", "value": 0.5}},
}
_PERCENTILE_CONDITION = {
    "condition_id": "metric_percentile",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.metric",
    "concept_version": "1.0",
    "strategy": {"type": "percentile", "params": {"direction": "top", "value": 25.0}},
}
_ZSCORE_CONDITION = {
    "condition_id": "metric_zscore",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.metric",
    "concept_version": "1.0",
    "strategy": {"type": "z_score", "params": {"threshold": 2.0, "direction": "above"}},
}
_CHANGE_CONDITION = {
    "condition_id": "metric_change",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.metric",
    "concept_version": "1.0",
    "strategy": {"type": "change", "params": {"direction": "increase", "value": 0.1}},
}

_CAT_CONCEPT_BODY = {
    "concept_id": "org.label",
    "version": "1.0",
    "namespace": "org",
    "output_type": "categorical",
    "primitives": {
        "label": {"type": "categorical", "missing_data_policy": "null"}
    },
    "features": {
        "out": {
            "op": "passthrough",
            "inputs": {"input": "label"},
            "params": {},
        }
    },
    "output_feature": "out",
}
_EQUALS_CONDITION = {
    "condition_id": "label_equals",
    "version": "1.0",
    "namespace": "org",
    "concept_id": "org.label",
    "concept_version": "1.0",
    "strategy": {"type": "equals", "params": {"value": "high"}},
}


# ── MockPool helper ────────────────────────────────────────────────────────────

class _MockPool:
    """Minimal asyncpg pool stub that serves concept/condition lookups."""

    def __init__(self, fetchrow_map: dict | None = None) -> None:
        self._map = fetchrow_map or {}
        self.decision_inserts: list[tuple] = []

    async def fetchrow(self, query: str, *args: Any) -> dict | None:
        if "INSERT INTO decisions" in query:
            self.decision_inserts.append(args)
            return {"decision_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
        if len(args) >= 2:
            key = (args[0], args[1])
            if key in self._map:
                return self._map[key]
        if len(args) == 1 and args[0] in self._map:
            return self._map[args[0]]
        return None

    async def fetch(self, query: str, *args: Any) -> list:
        if "concept_results" in query:
            return []
        return []

    async def execute(self, query: str, *args: Any) -> None:
        pass


def _make_pool(concept_body: dict | None = None, condition_body: dict | None = None) -> _MockPool:
    cb = concept_body or _CONCEPT_BODY
    cond_b = condition_body or _THRESHOLD_CONDITION
    return _MockPool(fetchrow_map={
        (cb["concept_id"], cb["version"]): {"body": json.dumps(cb)},
        (cond_b["condition_id"], cond_b["version"]): {"body": json.dumps(cond_b)},
    })


def _make_req(condition_id: str, condition_version: str, entity: str = "e1") -> Any:
    class _Req:
        pass
    req = _Req()
    req.condition_id = condition_id
    req.condition_version = condition_version
    req.entity = entity
    req.timestamp = None
    req.explain = False
    return req


def _resolver_with_fetch_error(primitive_name: str = "metric") -> DataResolver:
    """Return a DataResolver whose cache has a fetch_error for the named primitive."""
    connector = MockConnector(auth_failure=True)  # always raises AuthConnectorError
    resolver = DataResolver(connector=connector, backoff_base=0.0)
    # Trigger a fetch so the error is captured in the resolver cache
    resolver.fetch(primitive_name, "e1", None, policy=MissingDataPolicy.NULL)
    assert resolver.has_fetch_errors()  # confirm setup
    return resolver


def _resolver_with_null(primitive_name: str = "metric") -> DataResolver:
    """Return a DataResolver whose cache has a legitimate null for the named primitive."""
    connector = MockConnector(data={})  # returns None for all keys
    resolver = DataResolver(connector=connector, backoff_base=0.0)
    resolver.fetch(primitive_name, "e1", None, policy=MissingDataPolicy.NULL)
    assert not resolver.has_fetch_errors()  # confirm it's a genuine null, not an error
    return resolver


# ── Direct _evaluate_strategy() helper ────────────────────────────────────────

def _run_evaluate_strategy(
    condition_body: dict,
    concept_result: ConceptResult,
    resolver: DataResolver,
) -> DecisionValue:
    """Call ExecuteService._evaluate_strategy() directly without HTTP/DB stack."""
    pool = _make_pool()
    svc = ExecuteService(pool=pool)
    condition = ConditionDefinition(**condition_body)

    # Executor is not called for non-composite strategies inside _evaluate_strategy;
    # pass None — it is only used during composite recursive evaluation.
    class _FakeExecutor:
        pass

    async def _inner():
        return await svc._evaluate_strategy(
            condition,
            concept_result,
            concept_result.entity,
            concept_result.timestamp,
            _FakeExecutor(),  # type: ignore[arg-type]
            resolver,
        )

    return asyncio.run(_inner())


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — fetch_error returns reason="fetch_error"
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchErrorReturnsCorrectReason:
    def test_fetch_error_returns_fetch_error_reason(self):
        """
        MockConnector configured with auth_failure=True (permanent ConnectorError).
        Concept result value is None (connector failed → value absent).
        Strategy evaluation MUST return reason='fetch_error', NOT 'null_input'.
        """
        resolver = _resolver_with_fetch_error("metric")
        result = _null_float()

        dv = _run_evaluate_strategy(_THRESHOLD_CONDITION, result, resolver)

        assert dv.reason == "fetch_error", (
            f"Expected reason='fetch_error' for connector failure, got '{dv.reason}'"
        )
        assert dv.value is False
        assert dv.reason != "null_input"


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — legitimate null returns reason="null_input"
# ══════════════════════════════════════════════════════════════════════════════

class TestLegitimateNullReturnsNullInputReason:
    def test_legitimate_null_returns_null_input_reason(self):
        """
        MockConnector configured to return None (missing_data_policy=null).
        No connector error — the primitive legitimately has no value.
        Strategy evaluation MUST return reason='null_input', NOT 'fetch_error'.
        """
        resolver = _resolver_with_null("metric")
        result = _null_float()

        dv = _run_evaluate_strategy(_THRESHOLD_CONDITION, result, resolver)

        assert dv.reason == "null_input", (
            f"Expected reason='null_input' for legitimate null, got '{dv.reason}'"
        )
        assert dv.value is False
        assert dv.reason != "fetch_error"


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — fetch_error and null_input are distinguishable
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchErrorDistinguishableFromNullInput:
    def test_fetch_error_distinguishable_from_null_input(self):
        """
        Run both scenarios for the same condition.
        The two reasons MUST be different — regulatory auditability requires
        that a data-quality failure is always distinguishable from a
        legitimately absent value in the decision record.
        """
        resolver_fail = _resolver_with_fetch_error("metric")
        resolver_null = _resolver_with_null("metric")

        dv_fail = _run_evaluate_strategy(_THRESHOLD_CONDITION, _null_float(), resolver_fail)
        dv_null = _run_evaluate_strategy(_THRESHOLD_CONDITION, _null_float(), resolver_null)

        assert dv_fail.reason != dv_null.reason, (
            "fetch_error and null_input must produce different reason values "
            f"for regulatory auditability; both returned '{dv_fail.reason}'"
        )
        assert dv_fail.reason == "fetch_error"
        assert dv_null.reason == "null_input"


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — fetch_error appears in signal_errors with error_type
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchErrorInSignalErrors:
    def test_fetch_error_in_signal_errors(self):
        """
        Execute evaluate_full() with a connector that always fails.
        The decision record's signal_errors must contain the failing primitive
        with error_type='fetch_error' — distinguishing it from null_input.
        """
        from unittest.mock import patch

        # Patch _make_connector to return a permanently-failing connector
        failing_connector = MockConnector(auth_failure=True)

        pool = _MockPool(fetchrow_map={
            ("org.metric", "1.0"): {"body": json.dumps(_CONCEPT_BODY)},
            ("metric_above", "1.0"): {"body": json.dumps(_THRESHOLD_CONDITION)},
        })

        class _FullReq:
            concept_id = "org.metric"
            concept_version = "1.0"
            condition_id = "metric_above"
            condition_version = "1.0"
            entity = "e1"
            timestamp = None
            explain = False
            dry_run = False

            def model_dump(self):
                return self.__dict__.copy()

        async def _inner():
            svc = ExecuteService(pool=pool)
            with patch("app.services.execute._make_connector", return_value=failing_connector):
                result = await svc.evaluate_full(_FullReq())
            await asyncio.sleep(0)  # drain background decision-record task
            return result

        asyncio.run(_inner())

        # decision_inserts[0] args:
        # [0] concept_id, [1] concept_version, [2] condition_id, [3] condition_version,
        # [4] entity_id, [5] evaluated_at, [6] fired, [7] concept_value,
        # [8] threshold_applied, [9] ir_hash, [10] input_primitives,
        # [11] signal_errors, [12] reason, [13] action_ids_fired, [14] dry_run
        assert len(pool.decision_inserts) == 1
        args = pool.decision_inserts[0]
        signal_errors_json = args[11]
        assert signal_errors_json is not None, "signal_errors must be set when connector fails"
        signal_errors = json.loads(signal_errors_json)

        # The primitive "metric" failed — must appear in signal_errors with error_type
        assert "metric" in signal_errors, (
            f"Failing primitive 'metric' not found in signal_errors: {signal_errors}"
        )
        entry = signal_errors["metric"]
        assert isinstance(entry, dict), f"signal_errors entry must be a dict, got {type(entry)}"
        assert entry["error_type"] == "fetch_error", (
            f"Expected error_type='fetch_error', got '{entry.get('error_type')}'"
        )
        assert entry["primitive_id"] == "metric"


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — composite propagates fetch_error
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositePropagatesFetchError:
    def test_composite_propagates_fetch_error(self):
        """
        Composite AND where one operand has reason='fetch_error'.
        The composite result must have reason='fetch_error', not 'null_input'.

        This ensures that a data-quality failure (connector down) is not silently
        absorbed into a generic False result — it surfaces at the composite level.
        """
        strategy = CompositeStrategy()
        result = _null_float()

        operand_fetch_error = _bool_dv(False, reason="fetch_error")
        operand_normal_true = _bool_dv(True, reason=None)

        params = {
            "operator": "AND",
            "operand_results": [operand_fetch_error, operand_normal_true],
            "operands": ["op1", "op2"],
        }

        dv = strategy.evaluate(
            result, [], params,
            condition_id="c1", condition_version="1.0",
        )

        assert dv.value is False
        assert dv.reason == "fetch_error", (
            f"Composite AND should propagate reason='fetch_error' from operand, "
            f"got '{dv.reason}'"
        )
        assert dv.reason != "null_input"

    def test_composite_and_null_input_operand_propagates_reason(self):
        """
        Composite AND where one operand has reason='null_input'.
        The composite result must have reason='null_input'.
        """
        strategy = CompositeStrategy()
        result = _null_float()

        operand_null = _bool_dv(False, reason="null_input")
        operand_true = _bool_dv(True, reason=None)

        params = {
            "operator": "AND",
            "operand_results": [operand_null, operand_true],
            "operands": ["op1", "op2"],
        }

        dv = strategy.evaluate(
            result, [], params,
            condition_id="c1", condition_version="1.0",
        )

        assert dv.value is False
        assert dv.reason == "null_input"

    def test_composite_and_all_true_no_reason_propagation(self):
        """
        Composite AND all True → fires. No reason propagation (fired=True).
        """
        strategy = CompositeStrategy()
        result = _null_float()

        params = {
            "operator": "AND",
            "operand_results": [_bool_dv(True), _bool_dv(True)],
            "operands": ["op1", "op2"],
        }

        dv = strategy.evaluate(
            result, [], params,
            condition_id="c1", condition_version="1.0",
        )

        assert dv.value is True
        assert dv.reason is None

    def test_composite_not_still_propagates_fetch_error(self):
        """
        NOT operator with fetch_error operand still propagates reason='fetch_error'.
        (RULE 4 already propagates any reason from NOT; fetch_error included.)
        """
        strategy = CompositeStrategy()
        result = _null_float()

        operand = _bool_dv(False, reason="fetch_error")
        params = {"operator": "NOT", "operand_results": [operand]}

        dv = strategy.evaluate(
            result, [], params,
            condition_id="c1", condition_version="1.0",
        )

        assert dv.value is False
        assert dv.reason == "fetch_error"


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — fetch_error appears in all 5 strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchErrorAppearsInAllStrategies:
    """
    Run the fetch_error scenario through all 5 strategies.
    All must return reason='fetch_error' (not 'null_input') when the connector
    fails — regardless of strategy type or history requirements.

    The fetch_error check fires BEFORE the history guard, so z_score, percentile,
    and change strategies also return 'fetch_error' even with no history.
    """

    def _run(self, condition_body: dict, concept_result: ConceptResult) -> DecisionValue:
        resolver = _resolver_with_fetch_error()
        return _run_evaluate_strategy(condition_body, concept_result, resolver)

    def test_threshold_fetch_error(self):
        dv = self._run(_THRESHOLD_CONDITION, _null_float())
        assert dv.reason == "fetch_error", f"threshold: expected 'fetch_error', got '{dv.reason}'"
        assert dv.reason != "null_input"

    def test_percentile_fetch_error(self):
        dv = self._run(_PERCENTILE_CONDITION, _null_float())
        assert dv.reason == "fetch_error", f"percentile: expected 'fetch_error', got '{dv.reason}'"
        assert dv.reason != "null_input"

    def test_z_score_fetch_error(self):
        dv = self._run(_ZSCORE_CONDITION, _null_float())
        assert dv.reason == "fetch_error", f"z_score: expected 'fetch_error', got '{dv.reason}'"
        assert dv.reason != "null_input"

    def test_change_fetch_error(self):
        dv = self._run(_CHANGE_CONDITION, _null_float())
        assert dv.reason == "fetch_error", f"change: expected 'fetch_error', got '{dv.reason}'"
        assert dv.reason != "null_input"

    def test_equals_fetch_error(self):
        """
        Equals strategy: connector failure → reason='fetch_error'.
        The fetch_error guard fires before the strategy's own null_input check,
        so the result is BOOLEAN False with reason='fetch_error' rather than
        the CATEGORICAL null_input that EqualsStrategy would normally return.
        """
        # Equals uses a categorical concept
        resolver = DataResolver(
            connector=MockConnector(auth_failure=True), backoff_base=0.0
        )
        resolver.fetch("label", "e1", None, policy=MissingDataPolicy.NULL)
        assert resolver.has_fetch_errors()

        pool = _MockPool(fetchrow_map={
            ("org.label", "1.0"): {"body": json.dumps(_CAT_CONCEPT_BODY)},
            ("label_equals", "1.0"): {"body": json.dumps(_EQUALS_CONDITION)},
        })
        svc = ExecuteService(pool=pool)
        condition = ConditionDefinition(**_EQUALS_CONDITION)

        class _FakeExec:
            pass

        async def _inner():
            return await svc._evaluate_strategy(
                condition,
                _null_cat(),
                "e1",
                None,
                _FakeExec(),  # type: ignore[arg-type]
                resolver,
            )

        dv = asyncio.run(_inner())
        assert dv.reason == "fetch_error", f"equals: expected 'fetch_error', got '{dv.reason}'"
        assert dv.reason != "null_input"
