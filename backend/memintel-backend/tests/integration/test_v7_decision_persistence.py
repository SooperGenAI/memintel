"""
tests/integration/test_v7_decision_persistence.py
──────────────────────────────────────────────────────────────────────────────
T-5 Part 3 — Decision persistence.

Asserts that after evaluate_full() the decisions table contains the expected
row with correct field values.  Coverage:

  1. Row is written at all.
  2. fired=True when condition triggered; fired=False when not.
  3. concept_value reflects the actual evaluated primitive value.
  4. threshold_applied contains the strategy params dict.
  5. entity_id, concept_id, condition_id, condition_version fields correct.
  6. Separate rows per entity (no cross-entity contamination).
  7. dry_run=True → row NOT written (fire-and-forget skipped).
  8. list_for_entity returns newest first (ordering).
  9. decision_id is a non-empty UUID string.
 10. input_primitives captures the primitive value used in evaluation.

Key behaviour
─────────────
evaluate_full() persists the decision via asyncio.create_task() (fire-and-forget).
The task runs on the next event-loop iteration.  Tests call:

    run(asyncio.sleep(0.05))

after evaluate_full() to allow the background task to complete before querying
the decisions table.

Approach
────────
Uses DefinitionStore to register concept + condition + (optionally) action.
Monkeypatches _make_connector to inject controlled primitive values.
Uses DecisionStore.list_for_entity() to retrieve persisted records.
"""
from __future__ import annotations

import asyncio
import types as _types
from unittest.mock import patch

import pytest

import app.services.execute as _execute_module
from app.runtime.data_resolver import MockConnector
from app.services.execute import ExecuteService
from app.stores.decision import DecisionStore
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
        "description": f"Persistence test concept: {concept_id}",
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


def setup_concept_condition(
    run,
    db_pool,
    concept_body: dict,
    condition_body_: dict,
) -> tuple[str, str, str, str]:
    """
    Register concept + condition; return (concept_id, concept_ver, cond_id, cond_ver).
    """
    _register(run, db_pool, concept_body["concept_id"], concept_body, "concept")
    _register(run, db_pool, condition_body_["condition_id"], condition_body_, "condition")
    return (
        concept_body["concept_id"],
        "v1",
        condition_body_["condition_id"],
        "v1",
    )


def evaluate_full(
    run,
    db_pool,
    concept_id: str,
    condition_id: str,
    entity: str,
    connector_data: dict,
    *,
    timestamp: str | None = None,
    dry_run: bool = False,
):
    """
    Run the full ψ → φ → α → δ pipeline with a monkeypatched MockConnector.

    After returning, callers must wait for the fire-and-forget persistence task:
        run(asyncio.sleep(0.05))
    """
    mock_conn = MockConnector(data=connector_data)
    svc = ExecuteService(pool=db_pool)
    req = _types.SimpleNamespace(
        concept_id=concept_id,
        concept_version="v1",
        condition_id=condition_id,
        condition_version="v1",
        entity=entity,
        timestamp=timestamp,
        dry_run=dry_run,
        explain=False,
    )
    with patch.object(_execute_module, "_make_connector", return_value=mock_conn):
        return run(svc.evaluate_full(req))


def fetch_decisions(run, db_pool, entity: str, concept_id: str) -> list:
    """Retrieve persisted decisions for (entity, concept_id), newest first."""
    store = DecisionStore(db_pool)
    return run(store.list_for_entity(entity_id=entity, concept_id=concept_id))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Row is written after evaluate_full
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_row_written(db_pool, run):
    """
    evaluate_full() → decisions table contains at least one row for the entity.
    """
    concept = _float_concept("dp1.concept", "dp1.score")
    cond = _condition_body(
        "dp1.cond", "dp1.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-1",
        {("dp1.score", "user-1", None): 0.65},
    )
    run(asyncio.sleep(0.05))  # let fire-and-forget task complete

    records = fetch_decisions(run, db_pool, "user-1", cid)
    assert len(records) >= 1, (
        "No decision record found after evaluate_full(). "
        "The fire-and-forget persistence task may have silently failed."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. fired=True when condition triggers
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_fired_true_when_triggered(db_pool, run):
    """
    fired=True in the decisions table when the condition fires.

    0.65 < 0.80 (below threshold) → fired=True.
    """
    concept = _float_concept("dp2.concept", "dp2.score")
    cond = _condition_body(
        "dp2.cond", "dp2.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    result = evaluate_full(
        run, db_pool, cid, condid, "user-1",
        {("dp2.score", "user-1", None): 0.65},
    )
    # API-level assertion first
    assert result.decision.value is True, (
        f"evaluate_full returned value={result.decision.value!r} (expected True)"
    )

    run(asyncio.sleep(0.05))
    records = fetch_decisions(run, db_pool, "user-1", cid)
    assert records, "No decision row persisted"
    rec = records[0]
    assert rec.fired is True, (
        f"decisions.fired should be True for a triggered condition; got {rec.fired!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. fired=False when condition does not trigger
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_fired_false_when_not_triggered(db_pool, run):
    """
    fired=False in the decisions table when the condition does not fire.

    0.90 > 0.80, direction=below → not triggered → fired=False.
    """
    concept = _float_concept("dp3.concept", "dp3.score")
    cond = _condition_body(
        "dp3.cond", "dp3.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-1",
        {("dp3.score", "user-1", None): 0.90},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-1", cid)
    assert records, "No decision row persisted"
    assert records[0].fired is False, (
        f"decisions.fired should be False (not triggered); got {records[0].fired!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. concept_value stored correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_concept_value_stored(db_pool, run):
    """
    decisions.concept_value reflects the primitive value evaluated by the executor.

    z_score_op is a transparent passthrough: concept_value == input primitive.
    """
    concept = _float_concept("dp4.concept", "dp4.score")
    cond = _condition_body(
        "dp4.cond", "dp4.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-1",
        {("dp4.score", "user-1", None): 0.65},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-1", cid)
    assert records, "No decision row persisted"
    stored_val = records[0].concept_value
    assert stored_val == pytest.approx(0.65, abs=1e-6), (
        f"decisions.concept_value should be ≈0.65; got {stored_val!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. threshold_applied stores strategy params
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_threshold_applied_stored(db_pool, run):
    """
    decisions.threshold_applied is the strategy params dict from the condition.

    For threshold strategy: {"direction": "below", "value": 0.80}.
    """
    concept = _float_concept("dp5.concept", "dp5.score")
    params = {"direction": "below", "value": 0.80}
    cond = _condition_body(
        "dp5.cond", "dp5.concept",
        strategy_type="threshold",
        params=params,
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-1",
        {("dp5.score", "user-1", None): 0.65},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-1", cid)
    assert records, "No decision row persisted"
    applied = records[0].threshold_applied
    assert applied is not None, "threshold_applied should not be None"
    assert applied.get("direction") == "below", (
        f"threshold_applied.direction should be 'below'; got {applied!r}"
    )
    assert applied.get("value") == pytest.approx(0.80, abs=1e-6), (
        f"threshold_applied.value should be 0.80; got {applied!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. entity_id, concept_id, condition_id fields
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_identity_fields_correct(db_pool, run):
    """
    decisions table fields entity_id, concept_id, condition_id, condition_version
    match the values passed to evaluate_full().
    """
    concept = _float_concept("dp6.concept", "dp6.score")
    cond = _condition_body(
        "dp6.cond", "dp6.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, condver = setup_concept_condition(run, db_pool, concept, cond)

    entity = "entity-dp6"
    evaluate_full(
        run, db_pool, cid, condid, entity,
        {("dp6.score", entity, None): 0.65},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, entity, cid)
    assert records, "No decision row persisted"
    rec = records[0]
    assert rec.entity_id == entity, (
        f"entity_id should be '{entity}'; got {rec.entity_id!r}"
    )
    assert rec.concept_id == cid, (
        f"concept_id should be '{cid}'; got {rec.concept_id!r}"
    )
    assert rec.condition_id == condid, (
        f"condition_id should be '{condid}'; got {rec.condition_id!r}"
    )
    assert rec.condition_version == condver, (
        f"condition_version should be '{condver}'; got {rec.condition_version!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Separate rows per entity
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_separate_rows_per_entity(db_pool, run):
    """
    Each entity produces its own decision row — no cross-entity contamination.

    entity-A and entity-B both evaluated; each should have exactly one row and
    the correct entity_id in that row.
    """
    concept = _float_concept("dp7.concept", "dp7.score")
    cond = _condition_body(
        "dp7.cond", "dp7.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    ts_a = "2025-01-01T10:00:00Z"
    ts_b = "2025-01-01T10:00:01Z"

    # Evaluate entity-A first, then entity-B.
    # Connector keys must match the exact timestamp passed to evaluate_full so
    # that MockConnector.fetch(primitive, entity, timestamp) returns the value.
    evaluate_full(
        run, db_pool, cid, condid, "entity-A",
        {("dp7.score", "entity-A", ts_a): 0.65},
        timestamp=ts_a,
    )
    evaluate_full(
        run, db_pool, cid, condid, "entity-B",
        {("dp7.score", "entity-B", ts_b): 0.90},
        timestamp=ts_b,
    )
    run(asyncio.sleep(0.05))

    recs_a = fetch_decisions(run, db_pool, "entity-A", cid)
    recs_b = fetch_decisions(run, db_pool, "entity-B", cid)

    assert len(recs_a) >= 1, "No decision row for entity-A"
    assert len(recs_b) >= 1, "No decision row for entity-B"

    assert recs_a[0].entity_id == "entity-A", (
        f"entity-A row has entity_id={recs_a[0].entity_id!r}"
    )
    assert recs_b[0].entity_id == "entity-B", (
        f"entity-B row has entity_id={recs_b[0].entity_id!r}"
    )

    # entity-A: 0.65 < 0.80 → fired=True
    assert recs_a[0].fired is True, (
        f"entity-A (0.65 below 0.80): fired should be True; got {recs_a[0].fired!r}"
    )
    # entity-B: 0.90 >= 0.80 → fired=False
    assert recs_b[0].fired is False, (
        f"entity-B (0.90 not below 0.80): fired should be False; got {recs_b[0].fired!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. dry_run=True does NOT persist a decision
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_dry_run_not_persisted(db_pool, run):
    """
    dry_run=True → evaluate_full() runs ψ and φ but skips decision persistence.

    The decisions table must remain empty after a dry-run evaluation.
    """
    concept = _float_concept("dp8.concept", "dp8.score")
    cond = _condition_body(
        "dp8.cond", "dp8.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    result = evaluate_full(
        run, db_pool, cid, condid, "user-dry",
        {("dp8.score", "user-dry", None): 0.65},
        dry_run=True,
    )
    run(asyncio.sleep(0.05))

    # ψ and φ should still execute (condition evaluates to True)
    assert result.dry_run is True
    assert result.decision.value is True, (
        f"Dry-run ψ→φ should still evaluate correctly; got {result.decision.value!r}"
    )

    # No row in decisions table
    records = fetch_decisions(run, db_pool, "user-dry", cid)
    assert len(records) == 0, (
        f"dry_run=True should NOT write to decisions; found {len(records)} row(s)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9. list_for_entity ordering — newest first
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_ordering_newest_first(db_pool, run):
    """
    list_for_entity() returns decisions for an entity newest-first.

    Two evaluations with explicit timestamps; the later timestamp must appear
    first in the returned list.
    """
    concept = _float_concept("dp9.concept", "dp9.score")
    cond = _condition_body(
        "dp9.cond", "dp9.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    ts_earlier = "2025-06-01T10:00:00Z"
    ts_later   = "2025-06-02T10:00:00Z"

    # Connector keys must match the explicit timestamps.
    evaluate_full(
        run, db_pool, cid, condid, "user-order",
        {("dp9.score", "user-order", ts_earlier): 0.65},
        timestamp=ts_earlier,
    )
    evaluate_full(
        run, db_pool, cid, condid, "user-order",
        {("dp9.score", "user-order", ts_later): 0.75},
        timestamp=ts_later,
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-order", cid)
    assert len(records) >= 2, f"Expected ≥2 records; got {len(records)}"

    # Newest should appear first
    t0 = records[0].evaluated_at
    t1 = records[1].evaluated_at
    assert t0 >= t1, (
        f"list_for_entity should be newest first: records[0].evaluated_at={t0} "
        f"should be >= records[1].evaluated_at={t1}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10. decision_id is a non-empty UUID string
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_id_is_uuid_string(db_pool, run):
    """
    decisions.decision_id is a non-empty UUID string (not None, not integer).
    """
    import re
    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    concept = _float_concept("dp10.concept", "dp10.score")
    cond = _condition_body(
        "dp10.cond", "dp10.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-uuid",
        {("dp10.score", "user-uuid", None): 0.65},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-uuid", cid)
    assert records, "No decision row persisted"
    decision_id = records[0].decision_id
    assert decision_id and isinstance(decision_id, str), (
        f"decision_id should be a non-empty string; got {decision_id!r}"
    )
    assert _UUID_RE.match(decision_id), (
        f"decision_id should be a UUID string; got {decision_id!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 11. input_primitives captures evaluated primitive value
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_input_primitives_captured(db_pool, run):
    """
    decisions.input_primitives is a dict mapping primitive names to their values.

    After evaluation with primitive dp11.score=0.72, input_primitives should
    contain at least {'dp11.score': 0.72}.
    """
    concept = _float_concept("dp11.concept", "dp11.score")
    cond = _condition_body(
        "dp11.cond", "dp11.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    evaluate_full(
        run, db_pool, cid, condid, "user-prim",
        {("dp11.score", "user-prim", None): 0.72},
    )
    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-prim", cid)
    assert records, "No decision row persisted"
    primitives = records[0].input_primitives

    assert primitives is not None, (
        "decisions.input_primitives should not be None when primitives were fetched"
    )
    assert isinstance(primitives, dict), (
        f"input_primitives should be a dict; got {type(primitives).__name__}"
    )
    prim_key = "dp11.score"
    assert prim_key in primitives, (
        f"input_primitives should contain '{prim_key}'; keys: {list(primitives.keys())}"
    )
    assert primitives[prim_key] == pytest.approx(0.72, abs=1e-6), (
        f"input_primitives['{prim_key}'] should be ≈0.72; got {primitives[prim_key]!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 12. Consecutive evaluations — ON CONFLICT prevents duplicate rows
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_deduplication_on_conflict(db_pool, run):
    """
    ON CONFLICT (condition_id, condition_version, entity_id, evaluated_at) DO NOTHING.

    Two evaluate_full() calls with the SAME explicit timestamp for the same entity
    must produce exactly ONE decision row (the second write is silently discarded).
    """
    concept = _float_concept("dp12.concept", "dp12.score")
    cond = _condition_body(
        "dp12.cond", "dp12.concept",
        strategy_type="threshold",
        params={"direction": "below", "value": 0.80},
    )
    cid, _, condid, _ = setup_concept_condition(run, db_pool, concept, cond)

    ts = "2025-07-01T12:00:00Z"
    for _ in range(2):
        # Connector key must match the explicit timestamp so MockConnector returns the value.
        evaluate_full(
            run, db_pool, cid, condid, "user-dedup",
            {("dp12.score", "user-dedup", ts): 0.65},
            timestamp=ts,
        )

    run(asyncio.sleep(0.05))

    records = fetch_decisions(run, db_pool, "user-dedup", cid)
    assert len(records) == 1, (
        f"ON CONFLICT should deduplicate: expected 1 row, got {len(records)}"
    )
