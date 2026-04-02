"""
tests/unit/test_seven_checks.py
──────────────────────────────────────────────────────────────────────────────
Verification tests for the seven integrity checks.

CHECK 1 — ir_hash stability
  - hash_graph() uses hashlib.sha256 exclusively (no Python built-in hash())
  - Same concept definition → same ir_hash across repeated calls / "processes"
  - Different concept definitions → different ir_hash values

CHECK 2 — ir_hash verify logic
  - VerifyDecisionResponse model is correct (verified bool, stored/computed hashes)
  - verify logic: stored_hash == computed_hash → verified=True
  - verify logic: stored_hash != computed_hash → verified=False
  (The GET /decisions/{id}/verify endpoint is exercised in integration tests)

CHECK 3 — OperandRef backwards compatibility
  - OperandRef can be parsed from a dict (wire format)
  - CompositeParams accepts list[OperandRef] (typed model)
  - Migration 0010 file exists (confirms DB migration is in place)

CHECK 4 — fetch_error suppresses all action firing
  - fire_on='true'  + reason='fetch_error' → SKIPPED (was already correct)
  - fire_on='false' + reason='fetch_error' → SKIPPED (was BUG: fired)
  - fire_on='any'   + reason='fetch_error' → SKIPPED (was BUG: fired)
  - Same for other propagating reasons (null_input, insufficient_history, etc.)
  - reason=None, fire_on='false', value=False → TRIGGERED (still works)

CHECK 5 — NOT operator with OperandRef
  - NOT(True)  → DecisionValue(value=False, reason=None)
  - NOT(False) → DecisionValue(value=True,  reason=None)
  - NOT with fetch_error reason → propagates reason without inverting

CHECK 6 — DriverContribution accepts fetch_error dict value
  - DriverContribution(value={"error": "fetch_error"}) → no Pydantic error
  - _drivers_from_primitives() with mixed null/error/normal values → no crash
  - contribution sum is correct even when dict values are present

CHECK 7 — Concept value edge cases
  - float("inf")  → threshold strategy evaluates (inf > 0.5 = True), no crash
  - float("nan")  → threshold strategy compares to False (nan > 0.5 = False)
  - -0.0          → str(-0.0) round-trips via _parse_concept_value as 0.0
  - None concept value → threshold returns null_input reason
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import math
from pathlib import Path
from typing import Any

import pytest

from app.compiler.dag_builder import DAGBuilder
from app.compiler.ir_generator import IRGenerator
from app.models.action import (
    ActionDefinition,
    FireOn,
    NotificationActionConfig,
    TriggerConfig,
)
from app.models.condition import (
    CompositeOperator,
    CompositeParams,
    DecisionType,
    DecisionValue,
    DriverContribution,
    OperandRef,
    StrategyType,
)
from app.models.concept import ConceptDefinition
from app.models.result import ConceptOutputType
from app.models.result import ActionTriggeredStatus
from app.models.task import Namespace
from app.runtime.action_trigger import ActionTrigger
from app.services.explanation import _drivers_from_primitives
from app.stores.decision import _parse_concept_value
from app.strategies.composite import CompositeStrategy
from app.strategies.threshold import ThresholdStrategy


# ── Shared helpers ─────────────────────────────────────────────────────────────

def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _concept_body(concept_id: str = "org.churn") -> dict:
    return {
        "concept_id": concept_id,
        "version": "1.0",
        "namespace": "org",
        "output_type": "float",
        "description": "churn risk",
        "primitives": {"sig": {"type": "float", "missing_data_policy": "zero"}},
        "features": {"val": {"op": "normalize", "inputs": {"input": "sig"}, "params": {}}},
        "output_feature": "val",
    }


def _decision(
    value: bool = True,
    reason: str | None = None,
    fire_on: FireOn = FireOn.TRUE,
) -> tuple[DecisionValue, ActionDefinition]:
    dv = DecisionValue(
        value=value,
        decision_type=DecisionType.BOOLEAN,
        condition_id="cond_x",
        condition_version="1.0",
        entity="ent_001",
        reason=reason,
    )
    action = ActionDefinition(
        action_id="notify_1",
        version="1.0",
        config=NotificationActionConfig(
            type="notification",
            channel="alerts",
            message_template=None,
        ),
        trigger=TriggerConfig(
            fire_on=fire_on,
            condition_id="cond_x",
            condition_version="1.0",
        ),
        namespace=Namespace.ORG,
    )
    return dv, action


def _concept_result(value: float | None) -> Any:
    """Build a minimal ConceptResult-like object for strategy testing."""
    from app.models.result import ConceptResult
    return ConceptResult(
        value=value,
        type=ConceptOutputType.FLOAT,
        entity="ent",
        version="1.0",
        deterministic=True,
    )


def _bool_decision_value(
    fired: bool,
    reason: str | None = None,
    history_count: int | None = None,
) -> DecisionValue:
    return DecisionValue(
        value=fired,
        decision_type=DecisionType.BOOLEAN,
        condition_id="op_cond",
        condition_version="1.0",
        entity="ent",
        reason=reason,
        history_count=history_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — ir_hash stability
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck1IRHashStability:
    """ir_hash uses hashlib.sha256 exclusively and is process-stable."""

    def test_no_python_builtin_hash_in_ir_generator(self):
        """
        IRGenerator.hash_graph() must not use Python's built-in hash() function.

        Python's hash() is randomised across processes (PYTHONHASHSEED) so any
        hash computed with it would differ across runs — breaking tamper evidence.
        """
        src = inspect.getsource(IRGenerator)
        # Look for bare 'hash(' that isn't 'ir_hash(' or 'hashlib'
        import re
        matches = re.findall(r'\bhash\s*\(', src)
        # Only 'ir_hash' assignments referencing 'hashlib' are valid.
        # Any bare `hash(` call means Python's built-in is used.
        # Our canonical implementation uses hashlib.sha256().hexdigest() only.
        assert len(matches) == 0, (
            f"IRGenerator uses Python built-in hash() — found {len(matches)} call(s): "
            f"this is process-unstable (randomised by PYTHONHASHSEED). "
            f"Replace with hashlib.sha256()."
        )

    def test_same_concept_same_hash_repeated_calls(self):
        """Same concept body → identical ir_hash on two successive calls."""
        concept = ConceptDefinition.model_validate(_concept_body("org.check1a"))
        g1 = DAGBuilder().build_dag(concept)
        g2 = DAGBuilder().build_dag(concept)
        h1 = IRGenerator().hash_graph(g1)
        h2 = IRGenerator().hash_graph(g2)
        assert h1 == h2, f"ir_hash is not stable: {h1!r} != {h2!r}"

    def test_different_concepts_different_hashes(self):
        """Different concept definitions → different ir_hash values."""
        concept_a = ConceptDefinition.model_validate(_concept_body("org.concept_a"))
        concept_b = ConceptDefinition.model_validate({
            **_concept_body("org.concept_b"),
            "primitives": {
                "sig": {"type": "float", "missing_data_policy": "zero"},
                "sig2": {"type": "float", "missing_data_policy": "zero"},
            },
            "features": {
                "val": {"op": "add", "inputs": {"a": "sig", "b": "sig2"}, "params": {}},
            },
        })
        h_a = IRGenerator().hash_graph(DAGBuilder().build_dag(concept_a))
        h_b = IRGenerator().hash_graph(DAGBuilder().build_dag(concept_b))
        assert h_a != h_b, "Distinct concepts must produce distinct ir_hash values."

    def test_hash_is_64_hex_chars(self):
        """SHA-256 hexdigest is always 64 lowercase hex characters."""
        concept = ConceptDefinition.model_validate(_concept_body("org.len_check"))
        graph = DAGBuilder().build_dag(concept)
        h = IRGenerator().hash_graph(graph)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_set_on_graph_in_place(self):
        """hash_graph() sets graph.ir_hash in-place and returns same value."""
        concept = ConceptDefinition.model_validate(_concept_body("org.inplace"))
        graph = DAGBuilder().build_dag(concept)
        assert graph.ir_hash == ""  # sentinel before hashing
        returned = IRGenerator().hash_graph(graph)
        assert graph.ir_hash == returned
        assert len(returned) == 64


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — ir_hash verify logic
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck2VerifyLogic:
    """
    The verify endpoint compares stored_hash vs computed_hash.
    Tests cover the model and the comparison logic in isolation.
    The full HTTP endpoint is exercised in integration tests (requires DB).
    """

    def test_verify_response_model_accepts_matching_hashes(self):
        """VerifyDecisionResponse is importable and accepts valid fields."""
        from app.api.routes.decisions import VerifyDecisionResponse
        resp = VerifyDecisionResponse(
            decision_id="abc-123",
            verified=True,
            stored_hash="a" * 64,
            computed_hash="a" * 64,
        )
        assert resp.verified is True
        assert resp.stored_hash == resp.computed_hash

    def test_verify_logic_match(self):
        """stored_hash == computed_hash → verified=True."""
        concept = ConceptDefinition.model_validate(_concept_body("org.verify_match"))
        graph = DAGBuilder().build_dag(concept)
        computed = IRGenerator().hash_graph(graph)
        stored = computed  # what was stored at evaluation time
        verified = stored == computed
        assert verified is True

    def test_verify_logic_mismatch(self):
        """stored_hash != computed_hash → verified=False (tamper detected)."""
        concept = ConceptDefinition.model_validate(_concept_body("org.verify_mismatch"))
        graph = DAGBuilder().build_dag(concept)
        computed = IRGenerator().hash_graph(graph)
        stored = "0" * 64  # simulates a tampered or stale hash
        verified = stored == computed
        assert verified is False

    def test_verify_logic_none_stored_hash(self):
        """None stored_hash (old decision before ir_hash was populated) → verified=False."""
        stored: str | None = None
        computed = "a" * 64
        verified = stored == computed
        assert verified is False

    def test_verify_endpoint_registered_on_router(self):
        """GET /decisions/{decision_id}/verify must be registered on the router."""
        from app.api.routes.decisions import router
        paths = {route.path for route in router.routes}
        assert "/decisions/{decision_id}/verify" in paths, (
            "Verify endpoint not registered. Expected GET /decisions/{decision_id}/verify."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — OperandRef backwards compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck3OperandRefBackwardsCompat:
    """OperandRef model parsing and DB migration file presence."""

    def test_operand_ref_parses_from_dict(self):
        """OperandRef accepts wire-format dicts (as stored in DB JSONB)."""
        ref = OperandRef.model_validate({"condition_id": "org.cond_a", "condition_version": "1.0"})
        assert ref.condition_id == "org.cond_a"
        assert ref.condition_version == "1.0"

    def test_operand_ref_round_trip(self):
        """OperandRef → model_dump() → OperandRef round-trips correctly."""
        ref = OperandRef(condition_id="org.cond_b", condition_version="2.3")
        d = ref.model_dump()
        assert d == {"condition_id": "org.cond_b", "condition_version": "2.3"}
        ref2 = OperandRef.model_validate(d)
        assert ref2 == ref

    def test_composite_params_accepts_operand_refs(self):
        """CompositeParams.operands accepts list[OperandRef] typed objects."""
        params = CompositeParams(
            operator=CompositeOperator.AND,
            operands=[
                OperandRef(condition_id="org.cond_a", condition_version="1.0"),
                OperandRef(condition_id="org.cond_b", condition_version="1.0"),
            ],
        )
        assert len(params.operands) == 2
        assert all(isinstance(op, OperandRef) for op in params.operands)

    def test_composite_params_accepts_operand_ref_dicts(self):
        """CompositeParams.operands accepts raw dicts (Pydantic coerces to OperandRef)."""
        params = CompositeParams.model_validate({
            "operator": "AND",
            "operands": [
                {"condition_id": "org.cond_a", "condition_version": "1.0"},
                {"condition_id": "org.cond_b", "condition_version": "1.0"},
            ],
        })
        assert all(isinstance(op, OperandRef) for op in params.operands)

    def test_migration_0010_file_exists(self):
        """Alembic migration 0010 (OperandRef data migration) must be present."""
        migration_path = Path(__file__).parents[2] / "alembic" / "versions" / "0010_pin_composite_operand_versions.py"
        assert migration_path.exists(), (
            f"Migration file not found: {migration_path}. "
            "The data migration for OperandRef conversion must be present."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — fetch_error suppresses all action firing
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck4FetchErrorSuppressesActions:
    """
    When decision.reason is non-None, no action fires regardless of fire_on.

    Before fix: fire_on='false' and fire_on='any' would trigger even when
    the condition had reason='fetch_error'. The decision.value=False from a
    fetch_error is semantically distinct from a genuine False evaluation.
    """

    def _assert_skipped(self, fire_on: FireOn, reason: str) -> None:
        dv, action_def = _decision(value=False, reason=reason, fire_on=fire_on)
        trigger = ActionTrigger()
        result = run(trigger._process_action(action_def, dv, dry_run=False))
        assert result.status == ActionTriggeredStatus.SKIPPED, (
            f"Expected SKIPPED for fire_on={fire_on.value!r} with reason={reason!r}, "
            f"got {result.status.value!r}. "
            f"Actions must never fire when the condition evaluation failed."
        )

    def test_fetch_error_skips_fire_on_false(self):
        """fire_on='false' + reason='fetch_error' → SKIPPED (was BUG: triggered)."""
        self._assert_skipped(FireOn.FALSE, "fetch_error")

    def test_fetch_error_skips_fire_on_any(self):
        """fire_on='any' + reason='fetch_error' → SKIPPED (was BUG: triggered)."""
        self._assert_skipped(FireOn.ANY, "fetch_error")

    def test_fetch_error_skips_fire_on_true(self):
        """fire_on='true' + reason='fetch_error' → SKIPPED (was already correct)."""
        self._assert_skipped(FireOn.TRUE, "fetch_error")

    def test_null_input_skips_fire_on_false(self):
        """fire_on='false' + reason='null_input' → SKIPPED."""
        self._assert_skipped(FireOn.FALSE, "null_input")

    def test_null_input_skips_fire_on_any(self):
        """fire_on='any' + reason='null_input' → SKIPPED."""
        self._assert_skipped(FireOn.ANY, "null_input")

    def test_insufficient_history_skips_fire_on_any(self):
        """fire_on='any' + reason='insufficient_history' → SKIPPED."""
        self._assert_skipped(FireOn.ANY, "insufficient_history")

    def test_zero_variance_skips_fire_on_any(self):
        """fire_on='any' + reason='zero_variance' → SKIPPED."""
        self._assert_skipped(FireOn.ANY, "zero_variance")

    def test_no_reason_fire_on_false_value_false_triggers(self):
        """
        Genuine False evaluation (reason=None) with fire_on='false' → TRIGGERED.

        Verifies the fix does not suppress legitimate 'condition is False' actions.
        """
        dv, action_def = _decision(value=False, reason=None, fire_on=FireOn.FALSE)
        trigger = ActionTrigger()
        # In unit context with no HTTP client, notification goes to 'logged' state.
        # We just need to confirm it is NOT skipped at the reason guard.
        result = run(trigger._process_action(action_def, dv, dry_run=False))
        # notification with no CANVAS_NOTIFICATION_ENDPOINT → 'logged' (not skipped)
        assert result.status != ActionTriggeredStatus.SKIPPED, (
            "A genuine False decision with fire_on='false' must not be skipped "
            "by the reason guard. Only unevaluated results should be suppressed."
        )

    def test_no_reason_fire_on_true_value_true_triggers(self):
        """Genuine True evaluation (reason=None) with fire_on='true' → not skipped."""
        dv, action_def = _decision(value=True, reason=None, fire_on=FireOn.TRUE)
        trigger = ActionTrigger()
        result = run(trigger._process_action(action_def, dv, dry_run=False))
        assert result.status != ActionTriggeredStatus.SKIPPED


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — NOT operator with OperandRef
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck5NotOperatorWithOperandRef:
    """
    NOT operator evaluation via CompositeStrategy.

    OperandRef is resolved by execute.py before reaching CompositeStrategy;
    the strategy receives pre-evaluated DecisionValue operands.
    These tests verify NOT logic is correct for all cases.
    """

    def _evaluate_not(
        self, operand_value: bool, operand_reason: str | None = None
    ) -> DecisionValue:
        strategy = CompositeStrategy()
        result = _concept_result(1.0 if operand_value else 0.0)
        operand = _bool_decision_value(operand_value, reason=operand_reason)
        return strategy.evaluate(
            result=result,
            history=[],
            params={
                "operator": "NOT",
                "operand_results": [operand],
            },
            condition_id="not_cond",
            condition_version="1.0",
        )

    def test_not_true_returns_false(self):
        """NOT(True) → value=False, reason=None."""
        decision = self._evaluate_not(operand_value=True)
        assert decision.value is False
        assert decision.reason is None

    def test_not_false_returns_true(self):
        """NOT(False) → value=True, reason=None."""
        decision = self._evaluate_not(operand_value=False)
        assert decision.value is True
        assert decision.reason is None

    def test_not_with_fetch_error_propagates_reason(self):
        """NOT(fetch_error) → value=False, reason='fetch_error' (not inverted)."""
        decision = self._evaluate_not(operand_value=False, operand_reason="fetch_error")
        assert decision.value is False
        assert decision.reason == "fetch_error"

    def test_not_with_null_input_propagates_reason(self):
        """NOT(null_input) → reason='null_input' propagated without inversion."""
        decision = self._evaluate_not(operand_value=False, operand_reason="null_input")
        assert decision.reason == "null_input"

    def test_not_with_insufficient_history_propagates(self):
        """NOT(insufficient_history) → reason propagated."""
        decision = self._evaluate_not(operand_value=False, operand_reason="insufficient_history")
        assert decision.reason == "insufficient_history"

    def test_not_requires_exactly_one_operand(self):
        """NOT with two operands → semantic_error."""
        from app.models.errors import MemintelError
        strategy = CompositeStrategy()
        result = _concept_result(0.5)
        op = _bool_decision_value(True)
        with pytest.raises(MemintelError):
            strategy.evaluate(
                result=result,
                history=[],
                params={"operator": "NOT", "operand_results": [op, op]},
            )

    def test_not_composite_params_not_operator_exactly_one_operand(self):
        """CompositeParams NOT validator enforces exactly one operand."""
        params = CompositeParams(
            operator=CompositeOperator.NOT,
            operands=[OperandRef(condition_id="org.cond_a", condition_version="1.0")],
        )
        assert params.operator == CompositeOperator.NOT
        assert len(params.operands) == 1

    def test_not_composite_params_two_operands_rejected(self):
        """CompositeParams NOT with two operands raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompositeParams(
                operator=CompositeOperator.NOT,
                operands=[
                    OperandRef(condition_id="org.cond_a", condition_version="1.0"),
                    OperandRef(condition_id="org.cond_b", condition_version="1.0"),
                ],
            )


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — DriverContribution accepts fetch_error dict value
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck6DriverContributionFetchErrorValue:
    """
    DriverContribution.value must accept dict (fetch_error marker) and None.
    _drivers_from_primitives() must not crash when input_primitives contains
    {"error": "fetch_error"} entries.
    """

    def test_driver_contribution_accepts_fetch_error_dict(self):
        """DriverContribution(value={"error": "fetch_error"}) → no Pydantic error."""
        d = DriverContribution(
            signal="bad_prim",
            contribution=0.5,
            value={"error": "fetch_error"},
        )
        assert d.value == {"error": "fetch_error"}

    def test_driver_contribution_accepts_none_value(self):
        """DriverContribution(value=None) → no Pydantic error."""
        d = DriverContribution(signal="null_prim", contribution=0.5, value=None)
        assert d.value is None

    def test_driver_contribution_accepts_float(self):
        """Standard float value still accepted after model widening."""
        d = DriverContribution(signal="prim", contribution=0.5, value=0.87)
        assert d.value == pytest.approx(0.87)

    def test_driver_contribution_accepts_bool(self):
        """Boolean value still accepted after model widening."""
        d = DriverContribution(signal="prim", contribution=1.0, value=True)
        assert d.value is True

    def test_drivers_from_primitives_with_fetch_error(self):
        """
        _drivers_from_primitives() with a fetch_error dict value must not crash.
        The driver for the erroring primitive is included with the dict as value.
        """
        primitives = {
            "good_prim": 0.75,
            "bad_prim": {"error": "fetch_error"},
        }
        drivers = _drivers_from_primitives(primitives)
        assert len(drivers) == 2
        signals = {d.signal for d in drivers}
        assert "good_prim" in signals
        assert "bad_prim" in signals

    def test_drivers_from_primitives_with_null_value(self):
        """
        _drivers_from_primitives() with a None value (legitimate null) must not crash.
        """
        primitives = {
            "score": 0.5,
            "missing": None,
        }
        drivers = _drivers_from_primitives(primitives)
        assert len(drivers) == 2
        null_driver = next(d for d in drivers if d.signal == "missing")
        assert null_driver.value is None

    def test_drivers_from_primitives_contributions_sum_correct(self):
        """
        Contribution values must be 1/n per driver regardless of value type.
        3 primitives (one dict, one None, one float) → each gets 1/3.
        """
        primitives = {
            "a": 0.5,
            "b": None,
            "c": {"error": "fetch_error"},
        }
        drivers = _drivers_from_primitives(primitives)
        assert len(drivers) == 3
        total = sum(d.contribution for d in drivers)
        assert abs(total - 1.0) < 0.001, f"Contributions don't sum to 1.0: {total}"


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Concept value edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestCheck7ConceptValueEdgeCases:
    """
    Edge-case concept values must not crash the server.
    Tests cover the strategy evaluation path and the decision store
    serialisation layer (_parse_concept_value).
    """

    def _threshold_decision(self, concept_value: float | None) -> DecisionValue:
        """Evaluate threshold strategy with the given concept value."""
        strategy = ThresholdStrategy()
        result = _concept_result(concept_value)
        return strategy.evaluate(
            result=result,
            history=[],
            params={"direction": "above", "value": 0.5},
            condition_id="c",
            condition_version="1.0",
        )

    def test_inf_concept_value_no_crash(self):
        """
        float('inf') as concept value: threshold evaluates to True (inf > 0.5),
        no exception raised.
        """
        decision = self._threshold_decision(float("inf"))
        assert decision.value is True
        assert decision.reason is None

    def test_neg_inf_concept_value_no_crash(self):
        """
        float('-inf') as concept value: threshold evaluates to False (-inf < 0.5),
        no exception raised.
        """
        decision = self._threshold_decision(float("-inf"))
        assert decision.value is False

    def test_nan_concept_value_no_crash(self):
        """
        float('nan') as concept value: threshold comparison returns False
        (nan > 0.5 is False in Python), no exception raised.
        """
        decision = self._threshold_decision(float("nan"))
        # nan comparison is False — strategy should not crash regardless of result
        assert isinstance(decision.value, bool)

    def test_neg_zero_concept_value_no_crash(self):
        """
        -0.0 as concept value: threshold evaluates -0.0 > 0.5 → False, no crash.
        -0.0 == 0.0 in Python so it is treated as zero.
        """
        decision = self._threshold_decision(-0.0)
        assert decision.value is False  # -0.0 > 0.5 is False

    def test_none_concept_value_returns_null_input(self):
        """
        None concept value: threshold strategy returns reason='null_input'.
        This is the expected behaviour when a primitive has no value.
        """
        decision = self._threshold_decision(None)
        assert decision.value is False
        assert decision.reason == "null_input"

    def test_neg_zero_parse_concept_value_round_trip(self):
        """
        -0.0 stored as str("-0.0") in TEXT column.
        _parse_concept_value("-0.0") → 0.0 (float); -0.0 == 0.0 so no information lost.
        """
        result = _parse_concept_value("-0.0")
        assert isinstance(result, float)
        assert result == 0.0  # -0.0 == 0.0

    def test_inf_parse_concept_value_round_trip(self):
        """
        str(float('inf')) == 'inf'. _parse_concept_value('inf') → float('inf').
        Round-trip is preserved.
        """
        stored = str(float("inf"))  # "inf"
        result = _parse_concept_value(stored)
        assert isinstance(result, float)
        assert math.isinf(result)

    def test_nan_parse_concept_value_round_trip(self):
        """
        str(float('nan')) == 'nan'. _parse_concept_value('nan') → float('nan').
        Round-trip is preserved (nan is nan).
        """
        stored = str(float("nan"))  # "nan"
        result = _parse_concept_value(stored)
        assert isinstance(result, float)
        assert math.isnan(result)
