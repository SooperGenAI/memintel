"""
tests/integration/test_v7_missing_data_policy.py
──────────────────────────────────────────────────────────────────────────────
T-5 Part 2 — Missing data policy propagation.

For each missing_data_policy value, assert that when a primitive returns None
(no data for the requested key), the correct substitution is applied and the
downstream decision reflects that substituted value.

"No exception raised" is not a sufficient assertion — every test verifies the
actual outcome value so that a silent substitution bug cannot go undetected.

Policies under test
────────────────────
  null          — concept_value stays None; threshold returns False, reason='null_input'.
  zero          — concept_value becomes 0.0; threshold evaluates 0.0 vs threshold.
  forward_fill  — uses forward_fill_data[(primitive, entity)]; None if absent.
  backward_fill — uses backward_fill_data[(primitive, entity)]; None if absent.

Approach
────────
Uses the same helpers as test_v7_strategy_params.py, reproduced here so this
file can be run independently with pytest.

Connector injection:  monkeypatches _make_connector at the module level.
Connector key absent: guarantees None from MockConnector.fetch().
"""
from __future__ import annotations

import types as _types
from unittest.mock import patch

import pytest

import app.services.execute as _execute_module
from app.runtime.data_resolver import MockConnector
from app.services.execute import ExecuteService
from app.stores.definition import DefinitionStore


# ══════════════════════════════════════════════════════════════════════════════
# Concept / condition factories
# ══════════════════════════════════════════════════════════════════════════════

def _float_concept(concept_id: str, primitive_name: str, policy: str = "null") -> dict:
    """Minimal float concept using z_score_op (transparent passthrough for float)."""
    return {
        "concept_id": concept_id,
        "version": "v1",
        "namespace": "org",
        "output_type": "float",
        "description": f"Policy test concept: {concept_id}",
        "primitives": {
            primitive_name: {"type": "float", "missing_data_policy": policy}
        },
        "features": {
            "output": {
                "op": "z_score_op",
                "inputs": {"input": primitive_name},
                "params": {},
            }
        },
        "output_feature": "output",
    }


def _categorical_concept(
    concept_id: str,
    primitive_name: str,
    labels: list[str],
    policy: str = "null",
) -> dict:
    """Minimal categorical concept using passthrough op."""
    return {
        "concept_id": concept_id,
        "version": "v1",
        "namespace": "org",
        "output_type": "categorical",
        "labels": labels,
        "description": f"Policy test concept: {concept_id}",
        "primitives": {
            primitive_name: {
                "type": "categorical",
                "labels": labels,
                "missing_data_policy": policy,
            }
        },
        "features": {
            "output": {
                "op": "passthrough",
                "inputs": {"input": primitive_name},
                "params": {},
            }
        },
        "output_feature": "output",
    }


def _condition_body(
    condition_id: str,
    concept_id: str,
    strategy_type: str,
    params: dict,
) -> dict:
    return {
        "condition_id": condition_id,
        "version": "v1",
        "concept_id": concept_id,
        "concept_version": "v1",
        "namespace": "org",
        "strategy": {"type": strategy_type, "params": params},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _register(run, db_pool, definition_id: str, body: dict, def_type: str) -> None:
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id=definition_id,
        version="v1",
        definition_type=def_type,
        namespace="org",
        body=body,
    ))


def build_condition(run, db_pool, concept_body: dict, condition_body: dict) -> tuple[str, str]:
    """Register concept + condition; return (condition_id, 'v1')."""
    _register(run, db_pool, concept_body["concept_id"], concept_body, "concept")
    _register(run, db_pool, condition_body["condition_id"], condition_body, "condition")
    return condition_body["condition_id"], "v1"


def evaluate_condition(
    run,
    db_pool,
    condition_id: str,
    entity: str,
    connector_data: dict,
    *,
    forward_fill_data: dict | None = None,
    backward_fill_data: dict | None = None,
    timestamp: str | None = None,
):
    """
    Run ψ → φ with a monkeypatched MockConnector.

    connector_data keys are (primitive_name, entity_id, timestamp).
    Omitting a key guarantees MockConnector.fetch() returns None for it.
    """
    mock_conn = MockConnector(
        data=connector_data,
        forward_fill_data=forward_fill_data or {},
        backward_fill_data=backward_fill_data or {},
    )
    svc = ExecuteService(pool=db_pool)
    req = _types.SimpleNamespace(
        condition_id=condition_id,
        condition_version="v1",
        entity=entity,
        timestamp=timestamp,
        explain=False,
    )
    with patch.object(_execute_module, "_make_connector", return_value=mock_conn):
        return run(svc.evaluate_condition(req))


# ══════════════════════════════════════════════════════════════════════════════
# ZERO policy
# ══════════════════════════════════════════════════════════════════════════════

def test_policy_zero_substitutes_zero_not_triggered(db_pool, run):
    """
    ZERO policy: connector returns None → 0.0 substituted.
    Threshold above 10.0: 0.0 is NOT above 10.0 → not triggered.

    This asserts the substitution propagates to the strategy, not just that
    no exception is raised.
    """
    concept = _float_concept("mdp1.concept", "mdp1.score", policy="zero")
    cond = _condition_body(
        "mdp1.cond", "mdp1.concept",
        strategy_type="threshold",
        params={"direction": "above", "value": 10.0},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp1.cond", "user-1",
        connector_data={},  # absent → None → zero substitution → 0.0
    )

    assert result.value is False, (
        f"ZERO policy: 0.0 is not above 10.0; expected False, "
        f"got value={result.value!r}, reason={result.reason}"
    )


def test_policy_zero_substitution_triggers_when_zero_crosses_threshold(db_pool, run):
    """
    ZERO policy: connector returns None → 0.0 substituted.
    Threshold below 0.5: 0.0 < 0.5 → triggered.

    Confirms zero substitution is treated as a real numeric value by the strategy.
    """
    concept = _float_concept("mdp2.concept", "mdp2.score", policy="zero")
    cond = _condition_body(
        "mdp2.cond", "mdp2.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.5},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp2.cond", "user-1",
        connector_data={},  # absent → None → zero → 0.0
    )

    assert result.value is True, (
        f"ZERO policy: 0.0 < 0.5 should trigger; "
        f"got value={result.value!r}, reason={result.reason}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD FILL policy
# ══════════════════════════════════════════════════════════════════════════════

def test_policy_forward_fill_uses_last_known_no_trigger(db_pool, run):
    """
    FORWARD FILL: primary returns None, forward_fill_data has 0.90.
    Threshold below 0.50: 0.90 is NOT below 0.50 → not triggered.
    """
    concept = _float_concept("mdp3.concept", "mdp3.score", policy="forward_fill")
    cond = _condition_body(
        "mdp3.cond", "mdp3.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp3.cond", "user-1",
        connector_data={},  # primary absent → None
        forward_fill_data={("mdp3.score", "user-1"): 0.90},
    )

    assert result.value is False, (
        f"FORWARD FILL 0.90: should NOT trigger 'below 0.50', "
        f"got value={result.value!r}, reason={result.reason}"
    )


def test_policy_forward_fill_low_value_triggers(db_pool, run):
    """
    FORWARD FILL: primary returns None, forward_fill_data has 0.30.
    Threshold below 0.50: 0.30 < 0.50 → triggered.
    """
    concept = _float_concept("mdp4.concept", "mdp4.score", policy="forward_fill")
    cond = _condition_body(
        "mdp4.cond", "mdp4.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp4.cond", "user-1",
        connector_data={},
        forward_fill_data={("mdp4.score", "user-1"): 0.30},
    )

    assert result.value is True, (
        f"FORWARD FILL 0.30: should trigger 'below 0.50', "
        f"got value={result.value!r}, reason={result.reason}"
    )


def test_policy_forward_fill_no_history_propagates_null(db_pool, run):
    """
    FORWARD FILL: primary returns None, no forward_fill_data available.

    DataResolver returns PrimitiveValue(value=None, nullable=True).
    z_score_op propagates None → concept_value=None.
    ThresholdStrategy returns value=False, reason='null_input'.

    Critical: must NOT raise an exception and must NOT produce a spurious trigger.
    """
    concept = _float_concept("mdp5.concept", "mdp5.score", policy="forward_fill")
    cond = _condition_body(
        "mdp5.cond", "mdp5.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp5.cond", "user-1",
        connector_data={},
        forward_fill_data={},  # no fill data → None propagates
    )

    # Must not trigger (null input is not less than 0.50)
    assert result.value is not True, (
        f"FORWARD FILL with no data: should not trigger; "
        f"got value={result.value!r}, reason={result.reason}"
    )
    # null_input reason expected when no fill data available
    assert result.reason == "null_input", (
        f"Expected reason='null_input', got reason={result.reason!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD FILL policy
# ══════════════════════════════════════════════════════════════════════════════

def test_policy_backward_fill_uses_next_known_triggers(db_pool, run):
    """
    BACKWARD FILL: primary returns None, backward_fill_data has 0.40.
    Threshold below 0.50: 0.40 < 0.50 → triggered.
    """
    concept = _float_concept("mdp6.concept", "mdp6.score", policy="backward_fill")
    cond = _condition_body(
        "mdp6.cond", "mdp6.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp6.cond", "user-1",
        connector_data={},
        backward_fill_data={("mdp6.score", "user-1"): 0.40},
    )

    assert result.value is True, (
        f"BACKWARD FILL 0.40: should trigger 'below 0.50', "
        f"got value={result.value!r}, reason={result.reason}"
    )


def test_policy_backward_fill_high_value_no_trigger(db_pool, run):
    """
    BACKWARD FILL: primary returns None, backward_fill_data has 0.80.
    Threshold below 0.50: 0.80 is NOT below 0.50 → not triggered.
    """
    concept = _float_concept("mdp7.concept", "mdp7.score", policy="backward_fill")
    cond = _condition_body(
        "mdp7.cond", "mdp7.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp7.cond", "user-1",
        connector_data={},
        backward_fill_data={("mdp7.score", "user-1"): 0.80},
    )

    assert result.value is False, (
        f"BACKWARD FILL 0.80: should NOT trigger 'below 0.50', "
        f"got value={result.value!r}, reason={result.reason}"
    )


def test_policy_backward_fill_no_data_propagates_null(db_pool, run):
    """
    BACKWARD FILL: primary returns None, no backward_fill_data available.

    Result: None propagates → concept_value=None → not triggered, reason='null_input'.
    """
    concept = _float_concept("mdp8.concept", "mdp8.score", policy="backward_fill")
    cond = _condition_body(
        "mdp8.cond", "mdp8.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp8.cond", "user-1",
        connector_data={},
        backward_fill_data={},  # no backward fill
    )

    assert result.value is not True, (
        f"BACKWARD FILL with no data: should not trigger; "
        f"got value={result.value!r}, reason={result.reason}"
    )
    assert result.reason == "null_input", (
        f"Expected reason='null_input', got reason={result.reason!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# NULL policy
# ══════════════════════════════════════════════════════════════════════════════

def test_policy_null_no_crash_returns_false(db_pool, run):
    """
    NULL policy: connector returns None → concept_value=None.
    ThresholdStrategy returns value=False, reason='null_input'.
    No exception raised.
    """
    concept = _float_concept("mdp9.concept", "mdp9.score", policy="null")
    cond = _condition_body(
        "mdp9.cond", "mdp9.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp9.cond", "user-1",
        connector_data={},  # absent → None → null → concept_value=None
    )

    assert result.value is False, (
        f"NULL policy: concept_value=None should produce value=False; "
        f"got value={result.value!r}, reason={result.reason}"
    )
    assert result.reason == "null_input", (
        f"NULL policy: expected reason='null_input', got reason={result.reason!r}"
    )


def test_policy_null_does_not_discard_real_value(db_pool, run):
    """
    NULL policy, but connector DOES return a real value → policy NOT applied.

    0.30 < 0.50 → triggered despite null policy (null policy is only
    applied when the connector returns None, not when data is present).
    """
    concept = _float_concept("mdp10.concept", "mdp10.score", policy="null")
    cond = _condition_body(
        "mdp10.cond", "mdp10.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp10.cond", "user-1",
        connector_data={("mdp10.score", "user-1", None): 0.30},
    )

    assert result.value is True, (
        f"Real value 0.30 should trigger 'below 0.50' regardless of null policy; "
        f"got value={result.value!r}, reason={result.reason}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Categorical + NULL policy
# ══════════════════════════════════════════════════════════════════════════════

def test_categorical_null_policy_no_crash_no_match(db_pool, run):
    """
    Categorical concept, policy=null, connector returns None.

    passthrough op propagates None → concept_value=None.
    EqualsStrategy._categorical_null_decision → value=None, reason='null_input'.
    No exception raised.
    """
    labels = ["high_risk", "low_risk"]
    concept = _categorical_concept("mdp11.concept", "mdp11.label", labels, policy="null")
    cond = _condition_body(
        "mdp11.cond", "mdp11.concept",
        strategy_type="equals",
        params={"value": "high_risk", "labels": labels},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp11.cond", "user-1",
        connector_data={},  # absent → None → null → concept_value=None
    )

    assert result.value is None, (
        f"Categorical null input should produce value=None; "
        f"got value={result.value!r}, reason={result.reason}"
    )
    assert result.reason == "null_input", (
        f"Expected reason='null_input', got reason={result.reason!r}"
    )


def test_categorical_real_value_matches_through_null_policy(db_pool, run):
    """
    Categorical concept, policy=null, connector returns 'high_risk'.

    Real value present → null policy not applied → equals match → value='high_risk'.
    """
    labels = ["high_risk", "low_risk"]
    concept = _categorical_concept("mdp12.concept", "mdp12.label", labels, policy="null")
    cond = _condition_body(
        "mdp12.cond", "mdp12.concept",
        strategy_type="equals",
        params={"value": "high_risk", "labels": labels},
    )
    build_condition(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "mdp12.cond", "user-1",
        connector_data={("mdp12.label", "user-1", None): "high_risk"},
    )

    assert result.value == "high_risk", (
        f"Real value 'high_risk' should match equals target; "
        f"got value={result.value!r}, reason={result.reason}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Policy does not cross-contaminate between entities
# ══════════════════════════════════════════════════════════════════════════════

def test_zero_policy_does_not_affect_entity_with_real_data(db_pool, run):
    """
    ZERO policy: entity-1 has real data (0.70), entity-2 has no data (→ 0.0).

    entity-1: 0.70 is NOT below 0.50 → not triggered.
    entity-2: 0.0  IS  below 0.50 → triggered.

    Verifies substitution is per-entity, not global.
    """
    concept = _float_concept("mdp13.concept", "mdp13.score", policy="zero")
    cond = _condition_body(
        "mdp13.cond", "mdp13.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    build_condition(run, db_pool, concept, cond)

    # entity-1 has real data
    result_e1 = evaluate_condition(
        run, db_pool, "mdp13.cond", "entity-1",
        connector_data={("mdp13.score", "entity-1", None): 0.70},
    )
    assert result_e1.value is False, (
        f"entity-1 (0.70): should NOT trigger; got {result_e1.value!r}"
    )

    # entity-2 has no data → zero → 0.0 < 0.50 → triggered
    result_e2 = evaluate_condition(
        run, db_pool, "mdp13.cond", "entity-2",
        connector_data={},  # no data for entity-2
    )
    assert result_e2.value is True, (
        f"entity-2 (zero→0.0): should trigger 'below 0.50'; got {result_e2.value!r}"
    )
