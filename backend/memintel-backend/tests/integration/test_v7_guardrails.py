"""
tests/integration/test_v7_guardrails.py
──────────────────────────────────────────────────────────────────────────────
T-6 — Guardrails boundary enforcement.

Answers to the six pre-test questions (derived from source — NOT assumptions):
────────────────────────────────────────────────────────────────────────────

Q1: Where exactly does guardrails validation fire in the pipeline?
    AFTER the LLM call (Step 2), BEFORE any persistence, in Steps 3 and 4
    of _run_cor_pipeline():
      Step 3 — _validate_strategy_presence(condition_dict)
               Hard fail if strategy.type or strategy.params is absent.
      Step 4 — _validate_strategy_allowed(condition)
               Checks constraints.disallowed_strategies; z_score IS listed.
      Step 4 — _apply_bias_rules(condition, concept, intent)
               Applied only if parameter_bias_rules non-empty (see Q5).
      Step 4 — _validate_primitives_registered(concept)
               Checks every concept primitive against guardrails.primitives.
    All four run BEFORE the dry_run check, so dry_run=True still triggers them.

Q2: What happens when LLM produces a strategy not in the registry?
    Two distinct cases:
      a) Strategy name not in StrategyType enum (e.g. "neural_score"):
         ConditionDefinition(**condition_dict) raises a Pydantic ValueError.
         TaskAuthoringService wraps it in MemintelError(SEMANTIC_ERROR).
      b) Strategy in enum but in constraints.disallowed_strategies (z_score):
         _validate_strategy_allowed() raises MemintelError(SEMANTIC_ERROR).
         NOTE: memintel_guardrails.yaml lists z_score in disallowed_strategies.
    In both cases no task is persisted.

Q3: What happens when LLM produces a param value outside bounds?
    GAP — bounds are NOT enforced during task authoring.
    constraints.threshold_bounds defines [min, max] for change (0.01–0.25),
    percentile (1–50), z_score (0.5–4.0). on_bounds_exceeded: clamp.
    However, TaskAuthoringService has NO bounds-clamping or bounds-rejection
    logic. GuardrailConstraints.threshold_bounds is read ONLY by
    CalibrationService. Out-of-bounds values pass through to persistence.
    Tests in Section 3 confirm this gap.

Q4: What happens when LLM produces a strategy incompatible with the primitive type?
    GAP — type-strategy compatibility is NOT enforced at the guardrails/service
    layer. TYPE_STRATEGY_COMPATIBILITY is imported by the compiler (DAGBuilder /
    TypeChecker) and enforced only when definition_registry.register() is called
    with a REAL DefinitionRegistryService. With dry_run=True or a mock registry,
    the incompatible combination passes through silently.
    Tests in Section 2 confirm this gap.

Q5: How are bias rules applied — do they modify the LLM output or reject it?
    _apply_bias_rules() MODIFIES the condition (returns a new ConditionDefinition
    with an overridden threshold value). It does NOT reject.
    Applies only when ALL hold:
      1. guardrails.parameter_bias_rules is non-empty
      2. condition strategy is THRESHOLD
      3. A bias rule keyword matches the intent (first-match, case-insensitive)
      4. The concept's primitive has a threshold_prior at the resolved severity tier
    IMPORTANT: memintel_guardrails.yaml has NO parameter_bias_rules section.
    The severity_vocabulary.resolution_rules are LLM guidelines, not server
    enforcement. Bias rules in this codebase must be added programmatically.
    Section 4 tests use a custom Guardrails object with bias rules added.

Q6: What is the strategy selection priority order and where is it enforced?
    The priorities.order from the YAML:
      user_explicit → primitive_hint → mapping_rule →
      application_context → global_preferred → global_default
    This is injected into the LLM context as advisory guidance. It is NOT
    enforced server-side in TaskAuthoringService. Server-side enforcement is
    limited to:
      - Strategy presence check (SEMANTIC_ERROR if missing)
      - constraints.disallowed_strategies check (SEMANTIC_ERROR if violated)
      - constraints.disallowed_primitives enforcement (via _validate_primitives_registered,
        which raises REFERENCE_ERROR for any primitive not in guardrails.primitives)
    Section 5 tests verify the actual enforced behaviour, not the advisory order.

Approach
────────
All tests use service-level testing (TaskAuthoringService instantiated directly,
no HTTP layer). This gives precise control over:
  - LLM output (FixedOutputLLM — returns exactly the dict we configure)
  - Guardrails (loaded from memintel_guardrails.yaml; extended for bias tests)
  - dry_run=True — skips _register_and_persist so no DB/mocks needed for
    acceptance tests; all four guardrails checks still run.

Event loop: a fresh loop per test (no db_pool dependency). Rejection tests
raise before any async I/O, so event loop overhead is minimal.

Only primitives registered in memintel_guardrails.yaml are used in concepts
(because _validate_primitives_registered enforces this):
  float:              user.feature_adoption_score, account.seats_used_pct
  categorical:        account.plan_tier
  time_series<float>: user.daily_active_minutes, user.activity_count
"""
from __future__ import annotations

import asyncio
import copy
import os
from typing import Any

import pytest

from app.config.config_loader import ConfigLoader
from app.models.errors import ErrorType, MemintelError
from app.models.guardrails import BiasEffect, ParameterBiasRule
from app.models.result import DryRunResult
from app.models.task import CreateTaskRequest, DeliveryConfig, DeliveryType
from app.services.task_authoring import TaskAuthoringService

# ── Guardrails path ────────────────────────────────────────────────────────────
# Relative to the project root where pytest is invoked.
_GUARDRAILS_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "memintel_guardrails.yaml"
)


# ══════════════════════════════════════════════════════════════════════════════
# Infrastructure helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_guardrails():
    """Load the real Guardrails object from memintel_guardrails.yaml."""
    return ConfigLoader().load_guardrails(_GUARDRAILS_YAML)


def _run(coro):
    """
    Run a coroutine in a fresh event loop.

    No db_pool needed — these are service-level tests with dry_run=True.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class FixedOutputLLM:
    """
    Minimal LLM mock that always returns the same pre-configured dict.

    LLMClientBase.generate_task(intent, context) → dict.
    No network call, no latency, fully deterministic.
    """
    def __init__(self, output: dict) -> None:
        self._output = output

    def generate_task(self, intent: str, context: dict) -> dict:
        return self._output


class MockDefinitionRegistry:
    """
    Minimal definition registry mock for non-dry-run paths.

    For dry_run=True tests this is never called (register+persist is skipped).
    For non-dry-run tests it returns stub responses so the service doesn't fail.
    """
    async def register(self, body: Any, namespace: str = "org",
                       definition_type: str | None = None) -> Any:
        import types
        return types.SimpleNamespace(
            definition_id=getattr(body, "concept_id", "mock"),
            version="v1",
        )

    async def versions(self, definition_id: str) -> Any:
        import types
        return types.SimpleNamespace(
            versions=[types.SimpleNamespace(version="v1")]
        )

    async def get(self, definition_id: str, version: str) -> dict:
        return {}


class MockTaskStore:
    """Minimal task store that records created tasks."""
    def __init__(self) -> None:
        self.tasks_created: list = []

    async def create(self, task: Any) -> Any:
        import uuid
        task.task_id = str(uuid.uuid4())
        self.tasks_created.append(task)
        return task


def _make_service(
    llm_output: dict,
    guardrails=None,
    *,
    task_store: Any = None,
    definition_registry: Any = None,
) -> TaskAuthoringService:
    """
    Build a TaskAuthoringService with a FixedOutputLLM and given guardrails.

    task_store and definition_registry are only needed for non-dry-run tests.
    guardrails=None means no guardrails validation (only strategy_presence runs).
    """
    return TaskAuthoringService(
        task_store=task_store or MockTaskStore(),
        definition_registry=definition_registry or MockDefinitionRegistry(),
        guardrails=guardrails,
        llm_client=FixedOutputLLM(llm_output),
        max_retries=1,  # one attempt only — no retry logic needed in tests
    )


def _make_request(
    intent: str = "alert on low feature adoption",
    *,
    entity_scope: str = "user",
    dry_run: bool = True,
    return_reasoning: bool = False,
) -> CreateTaskRequest:
    """Build a minimal CreateTaskRequest."""
    return CreateTaskRequest(
        intent=intent,
        entity_scope=entity_scope,
        delivery=DeliveryConfig(
            type=DeliveryType.WEBHOOK,
            endpoint="https://guardrail-test.example.com/hook",
        ),
        stream=False,
        return_reasoning=return_reasoning,
        dry_run=dry_run,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM output builders
# ══════════════════════════════════════════════════════════════════════════════
#
# Primitives used — ALL are registered in memintel_guardrails.yaml:
#   float:              user.feature_adoption_score
#   time_series<float>: user.daily_active_minutes
#   categorical:        account.plan_tier
#
# Concept output_type=float always uses z_score_op (transparent passthrough).
# Concept output_type=categorical uses passthrough op.
# ──────────────────────────────────────────────────────────────────────────────

_FLOAT_PRIMITIVE = "user.feature_adoption_score"
_TIME_SERIES_PRIMITIVE = "user.daily_active_minutes"
_CATEGORICAL_PRIMITIVE = "account.plan_tier"

_FLOAT_CONCEPT = {
    "concept_id": "gr_test.adoption_score",
    "version": "v1",
    "namespace": "org",
    "output_type": "float",
    "description": "Feature adoption score for guardrail tests",
    "primitives": {
        _FLOAT_PRIMITIVE: {"type": "float", "missing_data_policy": "null"}
    },
    "features": {
        "output": {
            "op": "z_score_op",
            "inputs": {"input": _FLOAT_PRIMITIVE},
            "params": {},
        }
    },
    "output_feature": "output",
}

_TIME_SERIES_CONCEPT = {
    "concept_id": "gr_test.daily_minutes",
    "version": "v1",
    "namespace": "org",
    "output_type": "float",
    "description": "Daily active minutes for guardrail tests",
    "primitives": {
        _TIME_SERIES_PRIMITIVE: {"type": "float", "missing_data_policy": "zero"}
    },
    "features": {
        "output": {
            "op": "z_score_op",
            "inputs": {"input": _TIME_SERIES_PRIMITIVE},
            "params": {},
        }
    },
    "output_feature": "output",
}

_CATEGORICAL_CONCEPT = {
    "concept_id": "gr_test.plan_tier",
    "version": "v1",
    "namespace": "org",
    "output_type": "categorical",
    "labels": ["starter", "professional", "enterprise"],
    "description": "Account plan tier for guardrail tests",
    "primitives": {
        _CATEGORICAL_PRIMITIVE: {
            "type": "categorical",
            "labels": ["starter", "professional", "enterprise"],
            "missing_data_policy": "null",
        }
    },
    "features": {
        "output": {
            "op": "passthrough",
            "inputs": {"input": _CATEGORICAL_PRIMITIVE},
            "params": {},
        }
    },
    "output_feature": "output",
}


def _action(condition_id: str) -> dict:
    """Minimal action dict bound to the given condition_id."""
    return {
        "action_id": f"{condition_id}_action",
        "version": "v1",
        "namespace": "org",
        "config": {
            "type": "webhook",
            "endpoint": "https://guardrail-test.example.com/hook",
        },
        "trigger": {
            "fire_on": "true",
            "condition_id": condition_id,
            "condition_version": "v1",
        },
    }


def _llm_output(concept: dict, strategy_type: str, params: dict) -> dict:
    """
    Build a complete LLM output dict for a given concept and strategy.

    condition_id is derived from the concept_id to be unique per test.
    """
    cond_id = f"{concept['concept_id']}.cond_{strategy_type}"
    condition = {
        "condition_id": cond_id,
        "version": "v1",
        "concept_id": concept["concept_id"],
        "concept_version": "v1",
        "namespace": "org",
        "strategy": {"type": strategy_type, "params": params},
    }
    return {
        "concept":   concept,
        "condition": condition,
        "action":    _action(cond_id),
    }


def _float_threshold_output(direction: str = "below", value: float = 0.5) -> dict:
    return _llm_output(_FLOAT_CONCEPT, "threshold", {"direction": direction, "value": value})


def _float_percentile_output(direction: str = "bottom", value: float = 20.0) -> dict:
    return _llm_output(_FLOAT_CONCEPT, "percentile", {"direction": direction, "value": value})


def _timeseries_change_output(direction: str = "decrease", value: float = 0.10) -> dict:
    return _llm_output(_TIME_SERIES_CONCEPT, "change",
                       {"direction": direction, "value": value, "window": "7d"})


def _timeseries_zscore_output(threshold: float = 2.0) -> dict:
    return _llm_output(_TIME_SERIES_CONCEPT, "z_score",
                       {"threshold": threshold, "direction": "above", "window": "30d"})


def _categorical_equals_output(value: str = "starter") -> dict:
    return _llm_output(
        _CATEGORICAL_CONCEPT, "equals",
        {"value": value, "labels": ["starter", "professional", "enterprise"]},
    )


def _output_with_custom_strategy(strategy_type: str, params: dict) -> dict:
    """Build LLM output using the float concept with an arbitrary strategy_type string."""
    return _llm_output(_FLOAT_CONCEPT, strategy_type, params)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Strategy registry enforcement
# ══════════════════════════════════════════════════════════════════════════════

def test_registered_strategy_threshold_accepted():
    """
    threshold is registered in guardrails and NOT in disallowed_strategies.
    LLM output using threshold on a float concept → DryRunResult (accepted).
    """
    gr = _load_guardrails()
    svc = _make_service(_float_threshold_output(), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        f"threshold strategy should be accepted; got {type(result).__name__}"
    )
    assert result.condition.strategy.type.value == "threshold"


def test_registered_strategy_percentile_accepted():
    """
    percentile is registered in guardrails and NOT in disallowed_strategies.
    LLM output using percentile on a float concept → DryRunResult (accepted).
    """
    gr = _load_guardrails()
    svc = _make_service(_float_percentile_output(), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        f"percentile strategy should be accepted; got {type(result).__name__}"
    )
    assert result.condition.strategy.type.value == "percentile"


def test_registered_strategy_change_accepted():
    """
    change is registered in guardrails and NOT in disallowed_strategies.
    LLM output using change on a time_series<float> concept → accepted.
    """
    gr = _load_guardrails()
    svc = _make_service(_timeseries_change_output(), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        f"change strategy should be accepted; got {type(result).__name__}"
    )
    assert result.condition.strategy.type.value == "change"


def test_registered_strategy_equals_accepted():
    """
    equals is registered in guardrails and NOT in disallowed_strategies.
    LLM output using equals on a categorical concept → accepted.
    """
    gr = _load_guardrails()
    svc = _make_service(_categorical_equals_output(), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        f"equals strategy should be accepted; got {type(result).__name__}"
    )
    assert result.condition.strategy.type.value == "equals"


def test_unregistered_strategy_neural_score_rejected():
    """
    'neural_score' is not in StrategyType enum.
    ConditionDefinition(**condition_dict) raises ValueError → MemintelError(SEMANTIC_ERROR).
    No task persisted.
    """
    gr = _load_guardrails()
    svc = _make_service(
        _output_with_custom_strategy("neural_score", {"value": 0.9}),
        gr,
    )

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    err = exc_info.value
    assert err.error_type == ErrorType.SEMANTIC_ERROR, (
        f"Expected SEMANTIC_ERROR for unknown strategy 'neural_score'; "
        f"got {err.error_type}"
    )


def test_unregistered_strategy_custom_rule_rejected():
    """
    'custom_rule' is not in StrategyType enum → rejected with SEMANTIC_ERROR.
    """
    gr = _load_guardrails()
    svc = _make_service(
        _output_with_custom_strategy("custom_rule", {"threshold": 0.5}),
        gr,
    )

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR


def test_strategy_name_case_sensitivity_capitalised_rejected():
    """
    'Threshold' (PascalCase) is not a valid StrategyType enum value.
    StrategyType enum uses lowercase wire values — mismatch is rejected.

    Implemented behaviour: REJECTED with SEMANTIC_ERROR.
    The StrategyType enum is case-sensitive; 'Threshold' != 'threshold'.
    """
    gr = _load_guardrails()
    svc = _make_service(
        _output_with_custom_strategy("Threshold", {"direction": "below", "value": 0.5}),
        gr,
    )

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR, (
        "Capitalised strategy name 'Threshold' must be rejected — "
        "StrategyType enum is case-sensitive."
    )


def test_disallowed_strategy_z_score_rejected():
    """
    z_score is in StrategyType enum (valid syntax) but is listed in
    constraints.disallowed_strategies in memintel_guardrails.yaml.
    _validate_strategy_allowed() raises MemintelError(SEMANTIC_ERROR).

    This is the primary disallowed-strategy test. The error is SEMANTIC_ERROR,
    not TYPE_ERROR or BOUNDS_EXCEEDED — enforced in Step 4 of the pipeline.
    """
    gr = _load_guardrails()
    assert "z_score" in gr.constraints.disallowed_strategies, (
        "Test precondition: z_score must be in disallowed_strategies in guardrails YAML."
    )
    svc = _make_service(_timeseries_zscore_output(), gr)

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    err = exc_info.value
    assert err.error_type == ErrorType.SEMANTIC_ERROR, (
        f"Disallowed strategy z_score must raise SEMANTIC_ERROR; got {err.error_type}"
    )
    assert "z_score" in str(err).lower(), (
        f"Error message should mention 'z_score'; got: {err}"
    )


def test_strategy_missing_entirely_rejected():
    """
    LLM output with condition dict lacking the 'strategy' key entirely.
    _validate_strategy_presence() raises MemintelError(SEMANTIC_ERROR) in Step 3.
    """
    # Build output manually — omit strategy key from condition
    output = {
        "concept":   _FLOAT_CONCEPT,
        "condition": {
            "condition_id": "gr_test.no_strategy",
            "version":      "v1",
            "concept_id":   "gr_test.adoption_score",
            "concept_version": "v1",
            "namespace":    "org",
            # NO strategy key
        },
        "action": _action("gr_test.no_strategy"),
    }
    gr = _load_guardrails()
    svc = _make_service(output, gr)

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR
    assert "strategy" in str(exc_info.value).lower()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Type-strategy compatibility
# ══════════════════════════════════════════════════════════════════════════════
#
# GAP: Type-strategy compatibility (from guardrails.type_compatibility) is NOT
# enforced at the TaskAuthoringService guardrails layer. It is enforced only
# by the compiler (DAGBuilder/TypeChecker) during definition_registry.register(),
# which is called inside _register_and_persist. With dry_run=True or a mock
# registry, incompatible combinations pass through silently.
#
# What IS enforced: _validate_strategy_allowed() (disallowed_strategies) and
# _validate_primitives_registered() (primitive in registry). Neither checks
# type-strategy compatibility.
#
# The tests below document ACTUAL behaviour, not the spec intent.
# ──────────────────────────────────────────────────────────────────────────────

def test_threshold_on_float_accepted():
    """
    threshold + float primitive: compatible (valid_strategies for float includes threshold).
    No disallowed_strategies violation.
    → Accepted. DryRunResult returned.
    """
    gr = _load_guardrails()
    svc = _make_service(_float_threshold_output(), gr)
    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.type.value == "threshold"


def test_threshold_on_categorical_not_enforced_at_service_layer():
    """
    GAP: threshold + categorical primitive is invalid per type_compatibility
    (float.invalid_strategies = [change, z_score, equals, composite]; but
    categorical.invalid_strategies = [threshold, percentile, change, z_score, composite]).

    At the TaskAuthoringService guardrails layer this IS NOT rejected because:
      _validate_strategy_allowed  — only checks disallowed_strategies (z_score),
                                    NOT type-strategy pairs.
      _validate_primitives_registered — checks primitive exists in registry,
                                        NOT type compatibility with strategy.

    ACTUAL BEHAVIOUR: DryRunResult returned (not rejected).
    Type-strategy enforcement would only happen in the compiler during
    definition_registry.register(concept, ...) which is skipped for dry_run.
    """
    # categorical concept with threshold strategy (type-incompatible per spec)
    cond_id = "gr_test.categorical_threshold"
    output = {
        "concept":   _CATEGORICAL_CONCEPT,
        "condition": {
            "condition_id":   cond_id,
            "version":        "v1",
            "concept_id":     "gr_test.plan_tier",
            "concept_version": "v1",
            "namespace":      "org",
            "strategy": {
                "type": "threshold",
                "params": {"direction": "above", "value": 0.5},
            },
        },
        "action": _action(cond_id),
    }
    gr = _load_guardrails()
    svc = _make_service(output, gr)

    # GAP: dry_run skips compiler; incompatible combination NOT rejected.
    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult), (
        "GAP CONFIRMED: threshold on categorical is NOT rejected at the "
        "guardrails service layer (dry_run=True skips compiler validation). "
        "Type-strategy enforcement requires the full compiler path."
    )


def test_equals_on_categorical_accepted():
    """
    equals + categorical primitive: compatible (valid_strategies for categorical = [equals]).
    → Accepted. DryRunResult returned.
    """
    gr = _load_guardrails()
    svc = _make_service(_categorical_equals_output(), gr)
    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.type.value == "equals"


def test_equals_on_float_not_enforced_at_service_layer():
    """
    GAP: equals + float primitive is invalid per type_compatibility
    (float.invalid_strategies includes equals). But type-strategy is NOT
    enforced at the guardrails layer — equals on float passes through.

    ACTUAL BEHAVIOUR: DryRunResult returned (not rejected at service layer).
    """
    cond_id = "gr_test.float_equals"
    output = {
        "concept":   _FLOAT_CONCEPT,
        "condition": {
            "condition_id":   cond_id,
            "version":        "v1",
            "concept_id":     "gr_test.adoption_score",
            "concept_version": "v1",
            "namespace":      "org",
            "strategy": {
                "type": "equals",
                "params": {"value": "high"},
            },
        },
        "action": _action(cond_id),
    }
    gr = _load_guardrails()
    svc = _make_service(output, gr)

    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult), (
        "GAP CONFIRMED: equals on float NOT rejected at guardrails layer."
    )


def test_zscore_disallowed_regardless_of_type():
    """
    z_score is in disallowed_strategies — rejected regardless of primitive type.
    Even for time_series<float> (which is type-compatible with z_score per spec),
    z_score is disallowed by the constraints block.
    """
    gr = _load_guardrails()
    svc = _make_service(_timeseries_zscore_output(), gr)

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR


def test_change_on_timeseries_accepted():
    """
    change + time_series<float> primitive: compatible AND not disallowed.
    → Accepted. DryRunResult returned.
    """
    gr = _load_guardrails()
    svc = _make_service(_timeseries_change_output(), gr)
    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.type.value == "change"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Parameter bounds enforcement
# ══════════════════════════════════════════════════════════════════════════════
#
# GAP: constraints.threshold_bounds are NOT enforced during task authoring.
# The YAML defines:
#   change:    min: 0.01, max: 0.25
#   percentile: min: 1, max: 50
#   z_score:   min: 0.5, max: 4.0
#   on_bounds_exceeded: clamp
#
# However, TaskAuthoringService has no _clamp_params() or _validate_bounds()
# method. GuardrailConstraints.threshold_bounds is only consumed by
# CalibrationService. Out-of-bounds values pass through unchanged.
#
# Tests below confirm this gap and assert the ACTUAL behaviour.
# ──────────────────────────────────────────────────────────────────────────────

def test_threshold_value_within_normal_range_accepted():
    """
    threshold value=0.5: no defined bounds for threshold strategy in constraints.
    → Accepted without any bounds check.
    """
    gr = _load_guardrails()
    # Confirm: no threshold_bounds for 'threshold' strategy in the YAML
    assert "threshold" not in gr.constraints.threshold_bounds, (
        "Test precondition: 'threshold' strategy should have no bounds in YAML."
    )
    svc = _make_service(_float_threshold_output(value=0.5), gr)
    result = _run(svc.create_task(_make_request()))
    assert isinstance(result, DryRunResult)


def test_percentile_value_above_maximum_not_clamped_at_service_layer():
    """
    GAP: percentile value=75.0 exceeds constraints.threshold_bounds.percentile.max=50.
    Expected (per spec): clamped to 50.
    Actual: passes through unchanged (no clamping in TaskAuthoringService).

    NOTE: Values >100 or <0 are rejected by PercentileStrategy Pydantic validation
    (model-level bounds), not by guardrails clamping.  We use 75.0 which is inside
    Pydantic range (0–100) but above the guardrails max (50).

    ACTUAL BEHAVIOUR: DryRunResult returned with value=75.0 (unclamped).
    """
    gr = _load_guardrails()
    # Confirm bounds are defined in guardrails
    percentile_bounds = gr.constraints.threshold_bounds.get("percentile")
    assert percentile_bounds is not None and percentile_bounds.max == 50.0, (
        "Test precondition: percentile max bound should be 50."
    )

    svc = _make_service(_float_percentile_output(value=75.0), gr)
    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        "GAP CONFIRMED: percentile value=75.0 (above guardrails max=50) is NOT "
        "rejected or clamped at the TaskAuthoringService level. on_bounds_exceeded: "
        "clamp is only enforced by CalibrationService, not during task authoring."
    )
    # Confirm unclamped value is in the result
    stored_val = result.condition.strategy.params.value
    assert stored_val == 75.0, (
        f"GAP: value should be unclamped 75.0; got {stored_val}"
    )


def test_percentile_value_below_minimum_not_clamped_at_service_layer():
    """
    GAP: percentile value=0.5 is below constraints.threshold_bounds.percentile.min=1.
    ACTUAL BEHAVIOUR: passes through unchanged (not clamped to 1.0).

    NOTE: Pydantic allows ge=0, so 0.5 is valid at model level but below guardrails
    min=1. This correctly isolates the clamping-gap without triggering model validation.
    """
    gr = _load_guardrails()
    svc = _make_service(_float_percentile_output(value=0.5), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        "GAP CONFIRMED: percentile value=0.5 (below guardrails min=1) not clamped."
    )
    stored_val = result.condition.strategy.params.value
    assert stored_val == 0.5, f"GAP: value should be unclamped 0.5; got {stored_val}"


def test_change_value_above_maximum_not_clamped_at_service_layer():
    """
    GAP: change value=0.99 exceeds constraints.threshold_bounds.change.max=0.25.
    ACTUAL BEHAVIOUR: passes through unchanged at the TaskAuthoringService level.
    """
    gr = _load_guardrails()
    change_bounds = gr.constraints.threshold_bounds.get("change")
    assert change_bounds is not None and change_bounds.max == 0.25, (
        "Test precondition: change max bound should be 0.25."
    )

    svc = _make_service(_timeseries_change_output(value=0.99), gr)
    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        "GAP CONFIRMED: change value=0.99 (above max=0.25) not clamped."
    )
    stored_val = result.condition.strategy.params.value
    assert stored_val == 0.99, f"GAP: unclamped value expected; got {stored_val}"


def test_change_value_below_minimum_not_clamped_at_service_layer():
    """
    GAP: change value=0.001 is below constraints.threshold_bounds.change.min=0.01.
    ACTUAL BEHAVIOUR: passes through unchanged.
    """
    gr = _load_guardrails()
    svc = _make_service(_timeseries_change_output(value=0.001), gr)

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.params.value == 0.001


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Bias rules
# ══════════════════════════════════════════════════════════════════════════════
#
# memintel_guardrails.yaml does NOT define parameter_bias_rules.
# The severity_vocabulary.resolution_rules are guidelines injected into the
# LLM context, NOT server-side enforcement rules.
#
# To test the bias-rule mechanism (_apply_bias_rules), a custom Guardrails
# object is built that adds ParameterBiasRule entries programmatically.
#
# For user.feature_adoption_score, threshold_priors from the YAML are:
#   {"low": 0.3, "medium": 0.5, "high": 0.7}
#
# Bias rule with severity_shift=+1 on intent keyword "urgent":
#   Base tier = medium (index 1) → shifted to high (index 2) → prior = 0.7
#
# So if LLM produces threshold value=0.3 and intent contains "urgent",
# _apply_bias_rules replaces value with 0.7.
# ──────────────────────────────────────────────────────────────────────────────

def _guardrails_with_bias_rule(keyword: str, severity_shift: int):
    """
    Return a Guardrails object extended with a single ParameterBiasRule.

    The rule triggers when `keyword` appears in the intent (case-insensitive)
    and shifts the severity tier by `severity_shift` (-1, 0, +1).
    """
    base_gr = _load_guardrails()
    rule = ParameterBiasRule(
        if_instruction_contains=keyword,
        effect=BiasEffect(direction="tighten_threshold", severity_shift=severity_shift),
    )
    return base_gr.model_copy(update={"parameter_bias_rules": [rule]})


def test_no_bias_rules_in_yaml_confirmed():
    """
    Confirm that memintel_guardrails.yaml has NO parameter_bias_rules.
    This means _apply_bias_rules is a no-op with the file-based guardrails.
    """
    gr = _load_guardrails()
    assert len(gr.parameter_bias_rules) == 0, (
        "memintel_guardrails.yaml should have no parameter_bias_rules. "
        "If this fails, the YAML has been updated — review Section 4 tests."
    )


def test_bias_rule_high_urgency_shifts_threshold_up():
    """
    Bias rule: if intent contains 'urgent', severity_shift=+1 → high tier.
    user.feature_adoption_score threshold_priors.threshold.high = 0.7.
    LLM produces value=0.3 (low prior). After bias → value becomes 0.7.
    """
    gr = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)
    svc = _make_service(
        _float_threshold_output(value=0.3),  # LLM produces low-tier value
        gr,
    )

    result = _run(svc.create_task(
        _make_request(intent="urgent alert when feature adoption drops"),
    ))

    assert isinstance(result, DryRunResult)
    biased_value = result.condition.strategy.params.value
    assert biased_value == pytest.approx(0.7, abs=1e-9), (
        f"Bias rule (shift +1 → high tier): expected 0.7, got {biased_value}. "
        "The threshold prior for user.feature_adoption_score at 'high' is 0.7."
    )


def test_bias_rule_low_urgency_shifts_threshold_down():
    """
    Bias rule: if intent contains 'minor', severity_shift=-1 → low tier.
    user.feature_adoption_score threshold_priors.threshold.low = 0.3.
    LLM produces value=0.5 (medium prior). After bias → value becomes 0.3.
    """
    gr = _guardrails_with_bias_rule(keyword="minor", severity_shift=-1)
    svc = _make_service(
        _float_threshold_output(value=0.5),  # LLM produces medium-tier value
        gr,
    )

    result = _run(svc.create_task(
        _make_request(intent="alert on minor drop in feature adoption"),
    ))

    assert isinstance(result, DryRunResult)
    biased_value = result.condition.strategy.params.value
    assert biased_value == pytest.approx(0.3, abs=1e-9), (
        f"Bias rule (shift -1 → low tier): expected 0.3, got {biased_value}."
    )


def test_bias_rule_no_match_preserves_llm_value():
    """
    Intent does NOT contain the bias rule keyword → no bias applied.
    LLM-produced value is preserved unchanged in the dry-run result.
    """
    gr = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)
    svc = _make_service(
        _float_threshold_output(value=0.3),  # LLM produces 0.3
        gr,
    )

    result = _run(svc.create_task(
        _make_request(intent="alert when feature adoption drops below threshold"),
        # "urgent" NOT in intent
    ))

    assert isinstance(result, DryRunResult)
    # No bias match — LLM value 0.3 preserved as-is
    val = result.condition.strategy.params.value
    assert val == pytest.approx(0.3, abs=1e-9), (
        f"No bias rule matched: LLM value 0.3 should be unchanged; got {val}."
    )


def test_bias_rule_is_deterministic():
    """
    Bias rules are deterministic: same intent → same shifted value, every time.
    Submit the same request twice and assert identical outcomes.
    """
    gr = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)

    def run_once():
        svc = _make_service(_float_threshold_output(value=0.3), gr)
        return _run(svc.create_task(
            _make_request(intent="urgent alert when feature adoption drops"),
        ))

    result1 = run_once()
    result2 = run_once()

    val1 = result1.condition.strategy.params.value
    val2 = result2.condition.strategy.params.value

    assert val1 == val2, (
        f"Bias rule must be deterministic: run1={val1}, run2={val2} differ."
    )
    assert val1 == pytest.approx(0.7, abs=1e-9), (
        f"Both runs should produce biased value 0.7; got {val1}"
    )


def test_bias_rule_only_applies_to_threshold_strategy():
    """
    _apply_bias_rules() is a no-op for non-threshold strategies.
    A matching bias keyword in the intent has no effect on a change strategy.

    change strategy does NOT have an overrideable 'value' via threshold_priors
    in the bias mechanism (which only overrides threshold strategy params).
    """
    gr = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)
    svc = _make_service(
        _timeseries_change_output(value=0.10),  # change, not threshold
        gr,
    )

    result = _run(svc.create_task(
        _make_request(intent="urgent alert when daily minutes change"),
    ))

    assert isinstance(result, DryRunResult)
    # change strategy — bias rule skipped (strategy type check in _apply_bias_rules)
    assert result.condition.strategy.type.value == "change"
    assert result.condition.strategy.params.value == pytest.approx(0.10, abs=1e-9), (
        "Bias rule must NOT affect change strategy — only threshold is supported."
    )


def test_bias_rule_applied_before_dry_run():
    """
    Bias modification happens in Step 4 BEFORE the dry_run branch.
    The DryRunResult.condition must already contain the biased value —
    it is NOT a post-processing step.
    """
    gr = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)
    svc = _make_service(_float_threshold_output(value=0.3), gr)

    result = _run(svc.create_task(
        _make_request(
            intent="urgent alert when feature adoption drops",
            dry_run=True,
        ),
    ))

    # Bias happens before dry_run branch, so DryRunResult has biased condition.
    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.params.value == pytest.approx(0.7, abs=1e-9)


def test_severity_at_high_tier_clamped_no_error():
    """
    severity_shift=+1 from base tier=medium → high (index 2).
    A second shift of +1 from high would go to index 3, but is clamped to 2.
    Test with shift=+2 to confirm clamp: both +1 and +2 resolve to high tier.
    """
    # shift=2: tier_idx = max(0, min(2, 1+2)) = min(2, 3) = 2 → high → 0.7
    gr1 = _guardrails_with_bias_rule(keyword="urgent", severity_shift=1)
    gr2 = _guardrails_with_bias_rule(keyword="urgent", severity_shift=2)

    intent = "urgent alert when feature adoption drops"

    def _val(gr):
        svc = _make_service(_float_threshold_output(value=0.3), gr)
        result = _run(svc.create_task(_make_request(intent=intent)))
        return result.condition.strategy.params.value

    val_shift1 = _val(gr1)
    val_shift2 = _val(gr2)

    # Both shift +1 and +2 resolve to 'high' tier (clamped) → prior = 0.7
    assert val_shift1 == pytest.approx(0.7, abs=1e-9)
    assert val_shift2 == pytest.approx(0.7, abs=1e-9), (
        "Severity tier index is clamped to [0, 2]: shift+2 from medium should "
        "still resolve to 'high' (not overflow)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Strategy selection priority order
# ══════════════════════════════════════════════════════════════════════════════
#
# The priorities.order from the YAML:
#   user_explicit → primitive_hint → mapping_rule → application_context →
#   global_preferred → global_default
#
# This is ADVISORY — injected into LLM context, NOT server-side enforced.
# Server-side enforcement in TaskAuthoringService is limited to:
#   1. _validate_strategy_presence  (SEMANTIC_ERROR if strategy missing)
#   2. _validate_strategy_allowed   (SEMANTIC_ERROR if in disallowed_strategies)
#   3. _validate_primitives_registered (REFERENCE_ERROR if primitive not in registry)
#
# "user_explicit overrides global_default" is enforced by the LLM itself
# (via context injection), not by a server-side check that verifies priority.
#
# Tests here verify the enforced behaviours, not the advisory priority order.
# ──────────────────────────────────────────────────────────────────────────────

def test_priority_order_defined_in_guardrails():
    """
    Confirm the priority order is correctly defined in the guardrails YAML.
    The order itself is NOT enforced server-side but should match the spec.
    """
    gr = _load_guardrails()
    expected_order = [
        "user_explicit",
        "primitive_hint",
        "mapping_rule",
        "application_context",
        "global_preferred",
        "global_default",
    ]
    assert gr.priorities.order == expected_order, (
        f"Priority order mismatch: {gr.priorities.order}"
    )


def test_user_explicit_allowed_strategy_accepted():
    """
    When the LLM returns an allowed strategy (percentile), it is accepted
    regardless of other priority hints. No server-side priority check needed.
    """
    gr = _load_guardrails()
    svc = _make_service(_float_percentile_output(value=20.0), gr)

    result = _run(svc.create_task(
        _make_request(intent="alert when adoption score is in bottom 20 percentile"),
    ))

    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.type.value == "percentile"


def test_global_default_threshold_accepted_without_hint():
    """
    When LLM returns threshold (the strategy for most float signals),
    it is accepted even without an explicit user hint in the intent.
    """
    gr = _load_guardrails()
    svc = _make_service(_float_threshold_output(), gr)

    result = _run(svc.create_task(
        _make_request(intent="alert when adoption score changes"),
        # No explicit strategy hint
    ))

    assert isinstance(result, DryRunResult)
    assert result.condition.strategy.type.value == "threshold"


def test_disallowed_strategy_rejected_regardless_of_user_explicit():
    """
    Even if the user explicitly requests z_score, the server rejects it
    because z_score is in disallowed_strategies.
    user_explicit priority cannot override a hard disallow constraint.
    """
    gr = _load_guardrails()
    svc = _make_service(_timeseries_zscore_output(), gr)

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(
            _make_request(
                intent="use z-score based anomaly detection for daily active minutes",
            ),
        ))

    assert exc_info.value.error_type == ErrorType.SEMANTIC_ERROR


def test_primitive_not_in_registry_rejected_regardless_of_strategy():
    """
    _validate_primitives_registered raises REFERENCE_ERROR when a concept
    primitive is not in guardrails.primitives, regardless of strategy type.

    'internal.test_metric' is listed in disallowed_primitives in the YAML
    but is ALSO not in guardrails.primitives (which is the enforced check).
    Either way: REFERENCE_ERROR raised.
    """
    cond_id = "gr_test.unregistered_primitive_cond"
    output = {
        "concept": {
            "concept_id": "gr_test.unregistered_prim",
            "version": "v1",
            "namespace": "org",
            "output_type": "float",
            "description": "Test with unregistered primitive",
            "primitives": {
                "internal.test_metric": {"type": "float", "missing_data_policy": "null"}
            },
            "features": {
                "output": {
                    "op": "z_score_op",
                    "inputs": {"input": "internal.test_metric"},
                    "params": {},
                }
            },
            "output_feature": "output",
        },
        "condition": {
            "condition_id":   cond_id,
            "version":        "v1",
            "concept_id":     "gr_test.unregistered_prim",
            "concept_version": "v1",
            "namespace":      "org",
            "strategy": {"type": "threshold", "params": {"direction": "below", "value": 0.5}},
        },
        "action": _action(cond_id),
    }
    gr = _load_guardrails()
    # Confirm guardrails has a primitives registry (non-empty)
    assert gr.primitives, "Test precondition: guardrails must have a non-empty primitives dict."
    assert "internal.test_metric" not in gr.primitives

    svc = _make_service(output, gr)

    with pytest.raises(MemintelError) as exc_info:
        _run(svc.create_task(_make_request()))

    assert exc_info.value.error_type == ErrorType.REFERENCE_ERROR, (
        f"Unregistered primitive should raise REFERENCE_ERROR; got {exc_info.value.error_type}"
    )
    assert "internal.test_metric" in str(exc_info.value)


def test_no_guardrails_only_strategy_presence_enforced():
    """
    When guardrails=None, only _validate_strategy_presence runs.
    _validate_strategy_allowed and _validate_primitives_registered are skipped
    (both are guarded by `if self._guardrails is not None`).

    Unregistered primitive and non-disallowed strategy both pass through.
    """
    svc = _make_service(
        _float_threshold_output(),
        guardrails=None,  # no guardrails enforcement
    )

    result = _run(svc.create_task(_make_request()))

    assert isinstance(result, DryRunResult), (
        "With guardrails=None, threshold on float should always be accepted."
    )


def test_disallowed_strategy_correctly_listed_in_guardrails():
    """
    Validate the guardrails YAML constraints match the spec:
    - disallowed_strategies must include z_score
    - threshold_bounds must include change, percentile, z_score
    - on_bounds_exceeded must be 'clamp'
    """
    gr = _load_guardrails()

    assert "z_score" in gr.constraints.disallowed_strategies, (
        "z_score must be in disallowed_strategies per guardrails spec."
    )
    assert "change" in gr.constraints.threshold_bounds
    assert "percentile" in gr.constraints.threshold_bounds
    assert gr.constraints.on_bounds_exceeded == "clamp", (
        "on_bounds_exceeded must be 'clamp' per guardrails spec."
    )

    # Strategy registry must contain all 6 registered strategies
    registered = set(gr.strategy_registry.keys())
    for strategy in ("threshold", "percentile", "z_score", "change", "equals", "composite"):
        assert strategy in registered, f"{strategy} must be in strategy_registry."
