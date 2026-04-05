"""
tests/integration/test_v7_strategy_params.py
──────────────────────────────────────────────────────────────────────────────
T-5 Part 1 — Strategy parameter correctness.

For every strategy type, construct a real condition definition with known
parameters, evaluate it against a known input value, and assert the decision
outcome is correct.  "No exception raised" is not a sufficient assertion.

Approach
────────
  build_condition_with_strategy — stores concept + condition in the definitions
    table via DefinitionStore.  Returns (condition_id, condition_version).

  evaluate_condition — monkeypatches app.services.execute._make_connector so the
    ExecuteService uses a MockConnector whose data dict is caller-controlled.
    Then calls ExecuteService.evaluate_condition() which runs the full ψ → φ
    pipeline without persisting a decision record.

  For history-based strategies (z_score, percentile, change), history is seeded
  directly in concept_results via ConceptResultStore before evaluation.

  For composite, sub-conditions and their concepts are all registered in the DB;
  the MockConnector holds values for every operand primitive simultaneously.
"""
from __future__ import annotations

import types as _types
from unittest.mock import patch

import pytest

import app.services.execute as _execute_module
from app.models.condition import ConditionDefinition, OperandRef
from app.runtime.data_resolver import MockConnector
from app.services.execute import ExecuteService
from app.stores.concept_result import ConceptResultStore
from app.stores.definition import DefinitionStore


# ══════════════════════════════════════════════════════════════════════════════
# Concept / condition body factories
# ══════════════════════════════════════════════════════════════════════════════

def _float_concept(concept_id: str, primitive_name: str, policy: str = "null") -> dict:
    """Return a minimal float concept body.

    Uses z_score_op which is accepted by the TypeChecker for float inputs and
    is a transparent passthrough in the executor (returns the raw primitive value).
    """
    return {
        "concept_id": concept_id,
        "version": "v1",
        "namespace": "org",
        "output_type": "float",
        "description": f"Test float concept: {concept_id}",
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
    """Return a minimal categorical concept body using the passthrough op."""
    return {
        "concept_id": concept_id,
        "version": "v1",
        "namespace": "org",
        "output_type": "categorical",
        "labels": labels,
        "description": f"Test categorical concept: {concept_id}",
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
    """Store a definition body in the definitions table, version='v1'."""
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id=definition_id,
        version="v1",
        definition_type=def_type,
        namespace="org",
        body=body,
    ))


def build_condition_with_strategy(
    run,
    db_pool,
    concept_body: dict,
    condition_body: dict,
) -> tuple[str, str]:
    """Register concept + condition in the DB; return (condition_id, 'v1')."""
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
    Evaluate condition_id for entity, injecting known primitive values.

    Monkeypatches _make_connector so the service receives a MockConnector
    populated with connector_data (keyed on (primitive_name, entity, timestamp)).
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


def seed_history(run, db_pool, concept_id: str, entity: str, values: list[float]) -> None:
    """Insert concept_results rows to provide history for stateful strategies."""
    store = ConceptResultStore(db_pool)
    for val in values:
        run(store.store(
            concept_id=concept_id,
            version="v1",
            entity=entity,
            value=val,
            output_type="float",
        ))


# ══════════════════════════════════════════════════════════════════════════════
# Threshold strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def test_threshold_below_triggers(db_pool, run):
    """direction='below', value=0.80 → fires when input < 0.80."""
    concept = _float_concept("t1.concept", "t1.score")
    cond = _condition_body(
        "t1.cond", "t1.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, cver = build_condition_with_strategy(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, cid, "user-1",
        {("t1.score", "user-1", None): 0.65},
    )

    assert result.value is True, f"Expected triggered=True, got {result.value}"
    # No 'outcome' field on DecisionResult — fired is expressed as value=True.
    assert result.condition_id == cid
    assert result.condition_version == cver
    # Confirm strategy type is threshold via the condition we stored
    cond_def = ConditionDefinition(
        **run(DefinitionStore(db_pool).get(cid, cver))
    )
    assert cond_def.strategy.type.value == "threshold"


def test_threshold_above_no_trigger(db_pool, run):
    """direction='below', value=0.80 → does NOT fire when input > threshold."""
    concept = _float_concept("t2.concept", "t2.score")
    cond = _condition_body(
        "t2.cond", "t2.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "t2.cond", "user-1",
        {("t2.score", "user-1", None): 0.90},
    )
    assert result.value is False


def test_threshold_at_boundary(db_pool, run):
    """less_than is strict: value == threshold does NOT trigger."""
    concept = _float_concept("t3.concept", "t3.score")
    cond = _condition_body(
        "t3.cond", "t3.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "t3.cond", "user-1",
        {("t3.score", "user-1", None): 0.80},
    )
    assert result.value is False, (
        "Boundary (value == threshold) must NOT trigger for strict less_than."
    )


def test_threshold_greater_than_triggers(db_pool, run):
    """direction='above', value=30 → fires when input > 30, not when equal."""
    concept = _float_concept("t4.concept", "t4.metric")
    cond = _condition_body(
        "t4.cond", "t4.concept",
        strategy_type="threshold",
        params={"direction": "above", "value": 30.0},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)

    # 45 > 30 → triggered
    result = evaluate_condition(
        run, db_pool, "t4.cond", "user-1",
        {("t4.metric", "user-1", None): 45.0},
    )
    assert result.value is True

    # 30 == threshold → strict greater_than → NOT triggered
    result2 = evaluate_condition(
        run, db_pool, "t4.cond", "user-1",
        {("t4.metric", "user-1", None): 30.0},
    )
    assert result2.value is False


# ══════════════════════════════════════════════════════════════════════════════
# Percentile strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def test_percentile_strategy(db_pool, run):
    """
    direction='top', value=20 (top 20%) — verify correct ranking against history.

    History: [0.1, 0.2, 0.3, 0.4, 0.5].
    80th-percentile of history = 0.42.
    Current=0.45 > 0.42 → in top 20% → triggered.
    Current=0.35 < 0.42 → NOT in top 20% → not triggered.
    """
    concept = _float_concept("p1.concept", "p1.score")
    cond = _condition_body(
        "p1.cond", "p1.concept",
        strategy_type="percentile",
        params={"direction": "top", "value": 20.0},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)
    # Seed history (must be >= 3 for strategy to evaluate)
    seed_history(run, db_pool, "p1.concept", "user-1", [0.1, 0.2, 0.3, 0.4, 0.5])

    result_in_top = evaluate_condition(
        run, db_pool, "p1.cond", "user-1",
        {("p1.score", "user-1", None): 0.45},
    )
    assert result_in_top.value is True, (
        f"0.45 should be in top 20% of [0.1-0.5], got value={result_in_top.value}, "
        f"reason={result_in_top.reason}"
    )

    result_not_in_top = evaluate_condition(
        run, db_pool, "p1.cond", "user-1",
        {("p1.score", "user-1", None): 0.35},
    )
    assert result_not_in_top.value is False, (
        f"0.35 should NOT be in top 20% of [0.1-0.5], got value={result_not_in_top.value}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Z-score strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def test_zscore_strategy(db_pool, run):
    """
    threshold=2.0, direction='above'.

    History: [8, 10, 12] → mean=10, std=sqrt(8/3)≈1.633.
    current=16: z=(16-10)/1.633≈3.67 > 2.0 → triggered.
    current=11: z=(11-10)/1.633≈0.61 < 2.0 → NOT triggered.
    """
    concept = _float_concept("z1.concept", "z1.metric")
    cond = _condition_body(
        "z1.cond", "z1.concept",
        strategy_type="z_score",
        params={"threshold": 2.0, "direction": "above", "window": "30d"},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)
    seed_history(run, db_pool, "z1.concept", "user-1", [8.0, 10.0, 12.0])

    result_high = evaluate_condition(
        run, db_pool, "z1.cond", "user-1",
        {("z1.metric", "user-1", None): 16.0},
    )
    assert result_high.value is True, (
        f"z≈3.67 should exceed threshold 2.0, got value={result_high.value}, "
        f"reason={result_high.reason}"
    )

    result_low = evaluate_condition(
        run, db_pool, "z1.cond", "user-1",
        {("z1.metric", "user-1", None): 11.0},
    )
    assert result_low.value is False, (
        f"z≈0.61 should NOT exceed threshold 2.0, got value={result_low.value}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Change strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def test_change_strategy(db_pool, run):
    """
    direction='increase', value=0.40 (40% increase threshold).

    Previous (history[-1]) = 10.0.
    current=15.0: pct_change=(15-10)/10=0.50 > 0.40 → triggered.
    current=12.0: pct_change=(12-10)/10=0.20 < 0.40 → NOT triggered.
    """
    concept = _float_concept("c1.concept", "c1.metric")
    cond = _condition_body(
        "c1.cond", "c1.concept",
        strategy_type="change",
        params={"direction": "increase", "value": 0.40, "window": "1d"},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)
    # history[-1] is the most recent → use 10.0 as the reference value
    seed_history(run, db_pool, "c1.concept", "user-1", [8.0, 9.0, 10.0])

    result_trigger = evaluate_condition(
        run, db_pool, "c1.cond", "user-1",
        {("c1.metric", "user-1", None): 15.0},
    )
    assert result_trigger.value is True, (
        f"50% increase should exceed 40% threshold, got value={result_trigger.value}, "
        f"reason={result_trigger.reason}"
    )

    result_no_trigger = evaluate_condition(
        run, db_pool, "c1.cond", "user-1",
        {("c1.metric", "user-1", None): 12.0},
    )
    assert result_no_trigger.value is False, (
        f"20% increase should NOT exceed 40% threshold, got {result_no_trigger.value}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Equals strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def test_equals_strategy_match(db_pool, run):
    """Equals strategy fires when concept value matches the target label."""
    labels = ["high_risk", "medium_risk", "low_risk"]
    concept = _categorical_concept("eq1.concept", "eq1.label", labels)
    cond = _condition_body(
        "eq1.cond", "eq1.concept",
        strategy_type="equals",
        params={"value": "high_risk", "labels": labels},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "eq1.cond", "user-1",
        {("eq1.label", "user-1", None): "high_risk"},
    )
    assert result.value == "high_risk", (
        f"Equals match should return the matched label, got value={result.value!r}, "
        f"reason={result.reason}"
    )


def test_equals_strategy_no_match(db_pool, run):
    """Equals strategy does NOT fire when concept value does not match the target."""
    labels = ["high_risk", "medium_risk", "low_risk"]
    concept = _categorical_concept("eq2.concept", "eq2.label", labels)
    cond = _condition_body(
        "eq2.cond", "eq2.concept",
        strategy_type="equals",
        params={"value": "high_risk", "labels": labels},
    )
    build_condition_with_strategy(run, db_pool, concept, cond)

    result = evaluate_condition(
        run, db_pool, "eq2.cond", "user-1",
        {("eq2.label", "user-1", None): "low_risk"},
    )
    assert result.value is None, (
        f"No match should return None, got value={result.value!r}"
    )
    assert result.reason == "no_match"


# ══════════════════════════════════════════════════════════════════════════════
# Composite strategy tests
# ══════════════════════════════════════════════════════════════════════════════

def _register_composite_setup(run, db_pool):
    """
    Register two sub-conditions (A: score < 0.80, B: level < 0.50) and their
    concepts, plus a dummy float concept for the composite condition's own
    concept_id reference.

    Returns (cond_a_id, cond_b_id, dummy_concept_id).
    """
    # Sub-condition A: fires when score < 0.80
    concept_a = _float_concept("comp.concept_a", "comp.score_a")
    cond_a = _condition_body(
        "comp.cond_a", "comp.concept_a",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    _register(run, db_pool, "comp.concept_a", concept_a, "concept")
    _register(run, db_pool, "comp.cond_a", cond_a, "condition")

    # Sub-condition B: fires when level < 0.50
    concept_b = _float_concept("comp.concept_b", "comp.level_b")
    cond_b = _condition_body(
        "comp.cond_b", "comp.concept_b",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.50},
    )
    _register(run, db_pool, "comp.concept_b", concept_b, "concept")
    _register(run, db_pool, "comp.cond_b", cond_b, "condition")

    # Dummy float concept for the composite condition itself
    # (composite strategy ignores concept_result, but evaluate_condition
    #  still fetches and executes the linked concept)
    dummy = _float_concept("comp.dummy_concept", "comp.dummy_score")
    _register(run, db_pool, "comp.dummy_concept", dummy, "concept")

    return "comp.cond_a", "comp.cond_b", "comp.dummy_concept"


def test_composite_strategy_and_logic(db_pool, run):
    """
    AND composite fires only when ALL operands fire.

    Setup:
      cond_a fires when score_a < 0.80 (below).
      cond_b fires when level_b < 0.50 (below).

    Case 1: score_a=0.65 (<0.80 → A fires), level_b=0.30 (<0.50 → B fires)
            → AND → True.
    Case 2: score_a=0.65 (<0.80 → A fires), level_b=0.60 (≥0.50 → B does not fire)
            → AND → False.
    """
    cond_a_id, cond_b_id, dummy_cid = _register_composite_setup(run, db_pool)

    # Register the composite condition
    comp_body = _condition_body(
        "comp.and_cond", dummy_cid,
        strategy_type="composite",
        params={
            "operator": "AND",
            "operands": [
                {"condition_id": cond_a_id, "condition_version": "v1"},
                {"condition_id": cond_b_id, "condition_version": "v1"},
            ],
        },
    )
    _register(run, db_pool, "comp.and_cond", comp_body, "condition")

    entity = "user-1"
    # Case 1: both fire → AND → True
    result_both = evaluate_condition(
        run, db_pool, "comp.and_cond", entity,
        connector_data={
            ("comp.score_a",    entity, None): 0.65,
            ("comp.level_b",    entity, None): 0.30,
            ("comp.dummy_score", entity, None): 0.50,
        },
    )
    assert result_both.value is True, (
        f"AND(A triggers, B triggers) should be True, got {result_both.value}, "
        f"reason={result_both.reason}"
    )

    # Case 2: only A fires → AND → False
    result_one = evaluate_condition(
        run, db_pool, "comp.and_cond", entity,
        connector_data={
            ("comp.score_a",    entity, None): 0.65,
            ("comp.level_b",    entity, None): 0.60,
            ("comp.dummy_score", entity, None): 0.50,
        },
    )
    assert result_one.value is False, (
        f"AND(A triggers, B does not) should be False, got {result_one.value}"
    )


def test_composite_strategy_or_logic(db_pool, run):
    """
    OR composite fires when ANY operand fires.

    Case 1: only A fires → OR → True.
    Case 2: neither fires → OR → False.
    """
    # The composite setup registers concepts/conditions under 'comp.*' but since
    # clean_tables truncates before each test, re-register from scratch.
    cond_a_id, cond_b_id, dummy_cid = _register_composite_setup(run, db_pool)

    comp_body = _condition_body(
        "comp.or_cond", dummy_cid,
        strategy_type="composite",
        params={
            "operator": "OR",
            "operands": [
                {"condition_id": cond_a_id, "condition_version": "v1"},
                {"condition_id": cond_b_id, "condition_version": "v1"},
            ],
        },
    )
    _register(run, db_pool, "comp.or_cond", comp_body, "condition")

    entity = "user-2"
    # Case 1: only A fires (score_a=0.65 < 0.80; level_b=0.60 ≥ 0.50)
    result_one = evaluate_condition(
        run, db_pool, "comp.or_cond", entity,
        connector_data={
            ("comp.score_a",    entity, None): 0.65,
            ("comp.level_b",    entity, None): 0.60,
            ("comp.dummy_score", entity, None): 0.50,
        },
    )
    assert result_one.value is True, (
        f"OR(A triggers, B does not) should be True, got {result_one.value}, "
        f"reason={result_one.reason}"
    )

    # Case 2: neither fires (score_a=0.90 ≥ 0.80; level_b=0.60 ≥ 0.50)
    result_none = evaluate_condition(
        run, db_pool, "comp.or_cond", entity,
        connector_data={
            ("comp.score_a",    entity, None): 0.90,
            ("comp.level_b",    entity, None): 0.60,
            ("comp.dummy_score", entity, None): 0.50,
        },
    )
    assert result_none.value is False, (
        f"OR(neither fires) should be False, got {result_none.value}"
    )


def test_composite_operand_ref_types(db_pool, run):
    """
    Regression test for the OperandRef bug.

    When a composite condition is stored in the DB and fetched back, each
    operand in strategy.params.operands must be an OperandRef object (not a
    raw string or dict), with accessible .condition_id and .condition_version
    attributes.

    This test MUST fail if someone reverts the OperandRef fix — the bug caused
    condition_id access to raise TypeError because operands were stored as dicts.
    """
    # Build and register a composite condition (operand conditions don't need to
    # exist for this structural test — we're only checking the parsed model).
    comp_body = {
        "condition_id": "ref.composite_check",
        "version": "v1",
        "concept_id": "ref.dummy_concept",
        "concept_version": "v1",
        "namespace": "org",
        "strategy": {
            "type": "composite",
            "params": {
                "operator": "AND",
                "operands": [
                    {"condition_id": "ref.sub_a", "condition_version": "v1"},
                    {"condition_id": "ref.sub_b", "condition_version": "v1"},
                ],
            },
        },
    }
    store = DefinitionStore(db_pool)
    run(store.register(
        definition_id="ref.composite_check",
        version="v1",
        definition_type="condition",
        namespace="org",
        body=comp_body,
    ))

    # Fetch back and parse as ConditionDefinition
    raw_body = run(store.get("ref.composite_check", "v1"))
    cond_def = ConditionDefinition(**raw_body)

    operands = cond_def.strategy.params.operands

    # Each operand MUST be an OperandRef, not a raw dict or string.
    for i, op in enumerate(operands):
        assert isinstance(op, OperandRef), (
            f"operands[{i}] is {type(op).__name__!r}, expected OperandRef. "
            "This indicates the OperandRef fix was reverted — composite operands "
            "are being left as raw dicts instead of parsed OperandRef objects."
        )

    # .condition_id and .condition_version must be accessible without TypeError.
    try:
        ids = [op.condition_id for op in operands]
        versions = [op.condition_version for op in operands]
    except (TypeError, AttributeError) as exc:
        pytest.fail(
            f"Accessing .condition_id on operands raised {type(exc).__name__}: {exc}\n"
            "This is the OperandRef bug — operands are not properly typed."
        )

    assert ids == ["ref.sub_a", "ref.sub_b"]
    assert versions == ["v1", "v1"]

    # condition_version must be a string (not int, not None).
    for i, ver in enumerate(versions):
        assert isinstance(ver, str), (
            f"operands[{i}].condition_version is {type(ver).__name__!r}, expected str."
        )
